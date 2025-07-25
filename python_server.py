import logging
import os
import sys
import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import tempfile
import shutil
import pandas as pd
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse, Response
import uvicorn
from dotenv import load_dotenv
import torch
from pyannote.audio import Pipeline
from pydantic import BaseModel
import subprocess

# --- Environment and Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    logger.warning("Hugging Face token not found. Diarization will be disabled.")

# --- Model and Pipeline Initialization ---
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This server requires a GPU.")

logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"Device: {torch.cuda.get_device_name(0)}")

# Whisper Model
logger.info("Loading Whisper model (large-v3)...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
logger.info("Whisper model loaded successfully.")

# Diarization Pipeline
diarization_pipeline = None
if HUGGINGFACE_TOKEN:
    try:
        logger.info("Loading Pyannote Diarization pipeline...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_TOKEN
        ).to(torch.device("cuda"))
        logger.info("Diarization pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load diarization pipeline: {e}")
        diarization_pipeline = None

app = FastAPI(title="Whisper Transcription API", version="2.0")

# --- Helper Functions ---

def format_timestamp(seconds: float) -> str:
    dt = datetime.utcfromtimestamp(seconds)
    return dt.strftime('%H:%M:%S,%f')[:-3]

def to_srt(segments: List[Dict]) -> str:
    srt_content = ""
    for i, seg in enumerate(segments):
        start_time = format_timestamp(seg['start'])
        end_time = format_timestamp(seg['end'])
        speaker = f"[{seg['speaker']}] " if 'speaker' in seg else ""
        srt_content += f"{i + 1}\n{start_time} --> {end_time}\n{speaker}{seg['text'].strip()}\n\n"
    return srt_content

def to_vtt(segments: List[Dict]) -> str:
    vtt_content = "WEBVTT\n\n"
    for seg in segments:
        start_time = format_timestamp(seg['start']).replace(',', '.')
        end_time = format_timestamp(seg['end']).replace(',', '.')
        speaker = f"[{seg['speaker']}] " if 'speaker' in seg else ""
        vtt_content += f"{start_time} --> {end_time}\n{speaker}{seg['text'].strip()}\n\n"
    return vtt_content

def transcribe_and_diarize(
    audio_path: str,
    language: Optional[str],
    enable_diarization: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    # Added parameters
    beam_size: int,
    vad_filter: bool,
    vad_parameters: Dict[str, Any],
    condition_on_previous_text: bool,
    temperature: float,
    prompt: Optional[str],
    enable_profanity_filter: bool
) -> (List[Dict], Dict, int, List[Dict]):
    # 1. Transcription with word-level timestamps
    transcribe_args = {
        "language": language,
        "word_timestamps": True,
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "vad_parameters": vad_parameters,
        "condition_on_previous_text": condition_on_previous_text,
        "temperature": temperature,
        "initial_prompt": prompt
    }
    if enable_profanity_filter:
        transcribe_args["suppress_tokens"] = [-1]

    segments_gen, info = model.transcribe(audio_path, **transcribe_args)

    # Materialize the generator immediately to allow multiple iterations
    materialized_segments = list(segments_gen)

    word_segments = []
    for segment in materialized_segments:
        if segment.words:
            for word in segment.words:
                word_segments.append({
                    "start": word.start, "end": word.end, "text": word.word, "word_prob": word.probability
                })

    if not enable_diarization or not diarization_pipeline:
        # Return transcription without diarization
        final_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in materialized_segments]
        return final_segments, info, 0, word_segments

    # 2. Speaker Diarization
    logger.info("Performing speaker diarization...")
    
    diarization_params = {}
    if min_speakers is not None:
        diarization_params["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarization_params["max_speakers"] = max_speakers

    diarization = diarization_pipeline(audio_path, **diarization_params)
    speaker_turns = pd.DataFrame(
        [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)],
        columns=["start", "end", "speaker"]
    )
    logger.info(f"Found {len(speaker_turns['speaker'].unique())} speakers.")

    # 3. Align transcription and diarization
    for word in word_segments:
        word_center = (word["start"] + word["end"]) / 2
        speaker = speaker_turns[
            (speaker_turns["start"] <= word_center) & (speaker_turns["end"] >= word_center)
        ]
        if not speaker.empty:
            word["speaker"] = speaker.iloc[0]["speaker"]

    # 3.5. Fill in UNKNOWN speakers by propagating the last known speaker
    last_speaker = None
    for word in word_segments:
        if 'speaker' in word:
            last_speaker = word['speaker']
        elif last_speaker:
            word['speaker'] = last_speaker

    # 4. Combine words into speaker-aware segments
    final_segments = []
    current_segment = None
    for word in word_segments:
        if "speaker" not in word:
            word["speaker"] = "UNKNOWN"

        if current_segment and current_segment["speaker"] == word["speaker"]:
            current_segment["text"] += " " + word["text"]
            current_segment["end"] = word["end"]
        else:
            if current_segment:
                final_segments.append(current_segment)
            current_segment = {
                "start": word["start"],
                "end": word["end"],
                "text": word["text"],
                "speaker": word["speaker"]
            }
    if current_segment:
        final_segments.append(current_segment)

    num_speakers = len(speaker_turns['speaker'].unique())
    return final_segments, info, num_speakers, word_segments


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Whisper Transcription API is running."}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": {
            "transcription": model is not None,
            "diarization": diarization_pipeline is not None
        }
    }

@app.post("/v1/audio/transcriptions")
async def main_transcription_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form("whisper-1"), # Compatibility, not used
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: List[str] = Form(["segment"]), # Compatibility
    enable_diarization: bool = Form(False),
    min_speakers: Optional[str] = Form(None),
    max_speakers: Optional[str] = Form(None),
    # Restored quality and feature parameters
    beam_size: int = Form(8, description="Beam size for transcription (higher = more accurate but slower)."),
    output_content: str = Form("both", description="Control response content: 'text_only', 'timestamps_only', or 'both'."),
    vad_filter: bool = Form(True, description="Enable Voice Activity Detection (VAD) filter."),
    vad_threshold: float = Form(0.2, description="VAD threshold for speech detection sensitivity."),
    min_silence_duration_ms: int = Form(800, description="Minimum silence duration for VAD."),
    condition_on_previous_text: bool = Form(False, description="Condition on previous text to prevent repetition."),
    enable_profanity_filter: bool = Form(False, description="Enable the profanity filter to censorResults.")
):
    start_time = datetime.now()
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    _, extension = os.path.splitext(file.filename.lower())
    if extension not in ['.wav', '.mp3', '.flac', '.m4a']:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {extension}")

    min_speakers_int = int(min_speakers) if min_speakers and min_speakers.isdigit() else None
    max_speakers_int = int(max_speakers) if max_speakers and max_speakers.isdigit() else None

    # Treat empty string for language as None to trigger automatic language detection
    if not language:
        language = None

    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            temp_audio_path = tmp_file.name

        # The audio is converted to a standardized format to avoid issues with pyannote
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as converted_file:
            converted_audio_path = converted_file.name

        command = ["ffmpeg", "-i", temp_audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-y", converted_audio_path]
        
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {e.stderr}")
        

        vad_parameters = {
            "threshold": vad_threshold,
            "min_silence_duration_ms": min_silence_duration_ms
        }

        segments, info, num_speakers, words = await run_in_threadpool(
            transcribe_and_diarize,
            audio_path=converted_audio_path,
            language=language,
            enable_diarization=enable_diarization,
            min_speakers=min_speakers_int,
            max_speakers=max_speakers_int,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            condition_on_previous_text=condition_on_previous_text,
            temperature=temperature,
            prompt=prompt,
            enable_profanity_filter=enable_profanity_filter
        )

        full_text = " ".join(s['text'].strip() for s in segments)

        # Format response based on 'response_format'
        if response_format in ["json", "verbose_json"]:
            content = {}
            if response_format == "verbose_json":
                content.update({
                    "task": "transcribe",
                    "language": info.language,
                    "duration": info.duration,
                })

            if enable_diarization:
                content["speakers"] = num_speakers
            
            if output_content in ["text_only", "both"]:
                content["text"] = full_text

            if output_content in ["timestamps_only", "both"]:
                # For compatibility, return segments, not words, unless specified
                if "word" in timestamp_granularities:
                    content["words"] = words
                if "segment" in timestamp_granularities or "words" not in content:
                    content["segments"] = segments


            return JSONResponse(content=content)

        elif response_format == "text":
            return PlainTextResponse(full_text)
        
        elif response_format == "srt":
            return PlainTextResponse(to_srt(segments), media_type="text/plain")
            
        elif response_format == "vtt":
            return PlainTextResponse(to_vtt(segments), media_type="text/plain")

        else:
            raise HTTPException(status_code=400, detail="Invalid response_format")

    except Exception as e:
        logger.error(f"Error during transcription: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if 'converted_audio_path' in locals() and os.path.exists(converted_audio_path):
            os.remove(converted_audio_path)
        torch.cuda.empty_cache()

# Alias for backwards compatibility
@app.post("/transcribe")
async def transcribe_alias(
    file: UploadFile = File(...),
    model_name: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: List[str] = Form(["segment"]),
    enable_diarization: bool = Form(False),
    min_speakers: Optional[str] = Form(None),
    max_speakers: Optional[str] = Form(None),
    beam_size: int = Form(8),
    output_content: str = Form("both"),
    vad_filter: bool = Form(True),
    vad_threshold: float = Form(0.2),
    min_silence_duration_ms: int = Form(800),
    condition_on_previous_text: bool = Form(False),
    enable_profanity_filter: bool = Form(False)
):
    # This alias maintains backward compatibility by simply calling the new endpoint.
    # It now accepts all the same parameters as the main endpoint for full functionality.

    # Treat empty string for language as None to trigger automatic language detection
    if not language:
        language = None
        
    return await main_transcription_endpoint(
        file=file,
        model_name=model_name,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
        enable_diarization=enable_diarization,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        beam_size=beam_size,
        output_content=output_content,
        vad_filter=vad_filter,
        vad_threshold=vad_threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        condition_on_previous_text=condition_on_previous_text,
        enable_profanity_filter=enable_profanity_filter
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3333)