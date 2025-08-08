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
import re

import whisperx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv
import torch
import subprocess

# --- Environment and Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    logger.warning("Hugging Face token not found. Diarization will be disabled.")

# --- Text Processing Functions ---
try:
    import contractions
    from nemo_text_processing.text_normalization.normalize import Normalizer
    NORMALIZER = Normalizer(input_case='lower_cased', lang='en')
    TEXT_PROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Text processing libraries not available: {e}")
    logger.warning("Falling back to basic text normalization")
    TEXT_PROCESSING_AVAILABLE = False

def normalize_text_for_display(text: str) -> str:
    """
    Normalize text for proper display formatting using established NLP libraries.
    Converts uppercase emotional text to proper capitalization and punctuation.
    
    Args:
        text: Raw transcribed text (may contain emotional uppercase)
    
    Returns:
        Properly formatted text with correct capitalization and punctuation
    """
    if not text or not isinstance(text, str):
        return text
    
    text = text.strip()
    if not text:
        return text
    
    if TEXT_PROCESSING_AVAILABLE:
        try:
            # Step 1: Convert to lowercase and expand contractions
            normalized = text.lower()
            normalized = contractions.fix(normalized)
            
            # Step 2: Use NeMo text normalizer for proper capitalization and formatting
            # This handles proper nouns, sentence beginnings, etc.
            normalized = NORMALIZER.normalize(normalized, verbose=False)
            
            return normalized
        except Exception as e:
            logger.warning(f"Text normalization failed: {e}, falling back to basic normalization")
            
    # Fallback: Basic normalization if libraries aren't available
    normalized = text.lower()
    
    # Basic contractions
    basic_contractions = {
        " im ": " I'm ", " youre ": " you're ", " hes ": " he's ", " shes ": " she's ",
        " its ": " it's ", " were ": " we're ", " theyre ": " they're ", " dont ": " don't ",
        " cant ": " can't ", " wont ": " won't ", " isnt ": " isn't ", " arent ": " aren't "
    }
    
    normalized_spaced = f" {normalized} "
    for old, new in basic_contractions.items():
        normalized_spaced = normalized_spaced.replace(old, new)
    normalized = normalized_spaced.strip()
    
    # Capitalize first letter and after periods
    if normalized:
        normalized = normalized[0].upper() + normalized[1:]
    normalized = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), normalized)
    
    # Always capitalize "I"
    normalized = re.sub(r'\bi\b', 'I', normalized)
    
    return normalized

# Customer Feedback Optimized Settings for Accuracy
WHISPERX_CONFIG = {
    # Core accuracy settings (based on customer feedback)
    "compute_type": os.getenv("WHISPERX_COMPUTE_TYPE", "float32"),  # float32 for max accuracy on punctuation
    "batch_size": int(os.getenv("WHISPERX_BATCH_SIZE", "8")),      # Smaller batch for accuracy
    "chunk_length_s": int(os.getenv("WHISPERX_CHUNK_LENGTH", "15")), # Shorter for better speaker turns
    "return_char_alignments": os.getenv("WHISPERX_CHAR_ALIGN", "true").lower() == "true", # For punctuation
    
    # Enhanced VAD Configuration (addressing customer speaker attribution issues)
    "vad_onset": float(os.getenv("VAD_ONSET", "0.35")),           # Enhanced: More sensitive boundary detection
    "vad_offset": float(os.getenv("VAD_OFFSET", "0.25")),         # Enhanced: Tighter boundaries
    "min_segment_length": float(os.getenv("MIN_SEGMENT_LENGTH", "0.5")), # Prevents over-segmentation
    "max_segment_length": float(os.getenv("MAX_SEGMENT_LENGTH", "30.0")), # Maximum segment duration
    "speech_threshold": float(os.getenv("SPEECH_THRESHOLD", "0.6")), # Confidence threshold for speech detection
    
    # Advanced settings
    "interpolate_method": os.getenv("INTERPOLATE_METHOD", "linear"), # Better word timing
    "align_model": os.getenv("ALIGN_MODEL", "WAV2VEC2_ASR_LARGE_LV60K_960H"), # Best alignment model
    "segment_resolution": os.getenv("SEGMENT_RESOLUTION", "sentence"), # Better punctuation context
}

# Enhanced Diarization Configuration (Customer Feedback Addressing)
DIARIZATION_CONFIG = {
    "speaker_smoothing_enabled": os.getenv("SPEAKER_SMOOTHING_ENABLED", "true").lower() == "true",
    "speaker_confidence_threshold": float(os.getenv("SPEAKER_CONFIDENCE_THRESHOLD", "0.8")),
    "text_normalization_mode": os.getenv("TEXT_NORMALIZATION_MODE", "enhanced"),
    "sentiment_analysis_enabled": os.getenv("SENTIMENT_ANALYSIS_ENABLED", "false").lower() == "true",
}

# Concurrent Processing Configuration
CONCURRENT_CONFIG = {
    "max_concurrent_requests": int(os.getenv("MAX_CONCURRENT_REQUESTS", "4")), # RTX 3090: 4 requests with diarization
    "queue_timeout": int(os.getenv("QUEUE_TIMEOUT", "300")),  # 5 minute queue timeout
    "memory_per_request_gb": float(os.getenv("MEMORY_PER_REQUEST_GB", "6.0")), # 4GB + 2GB diarization
    "enable_request_queuing": os.getenv("ENABLE_REQUEST_QUEUING", "true").lower() == "true",
}

# Initialize concurrent processing
import asyncio
from asyncio import Semaphore, Queue
from typing import NamedTuple
import time
import psutil
import torch

class RequestState(NamedTuple):
    request_id: str
    start_time: float
    audio_duration: Optional[float] = None

# Global concurrent processing state
processing_semaphore = Semaphore(CONCURRENT_CONFIG["max_concurrent_requests"])
request_queue = Queue()
active_requests = {}
request_counter = 0

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved, 
            "free_gb": total - reserved,
            "total_gb": total
        }
    return {"error": "CUDA not available"}

async def monitor_memory():
    """Background task to monitor memory usage and log warnings"""
    while True:
        try:
            memory_info = get_gpu_memory_info()
            if "error" not in memory_info:
                if memory_info["free_gb"] < CONCURRENT_CONFIG["memory_per_request_gb"]:
                    logger.warning(f"Low GPU memory: {memory_info['free_gb']:.1f}GB free, need {CONCURRENT_CONFIG['memory_per_request_gb']}GB per request")
        except Exception as e:
            logger.error(f"Memory monitoring error: {e}")
        await asyncio.sleep(30)  # Check every 30 seconds

# Memory monitor will be started in app startup

# --- Model and Pipeline Initialization ---
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This server requires a GPU.")

logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"Device: {torch.cuda.get_device_name(0)}")

# Whisper Model
logger.info(f"Loading WhisperX model (large-v3) with compute_type={WHISPERX_CONFIG['compute_type']}...")
model = whisperx.load_model("large-v3", device="cuda", compute_type=WHISPERX_CONFIG["compute_type"])
logger.info("WhisperX model loaded successfully.")

# Diarization will be loaded on demand if a token is present.
if not HUGGINGFACE_TOKEN:
    logger.warning("Hugging Face token not found. Diarization will be disabled.")

# Background tasks
background_tasks = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup
    task = asyncio.create_task(monitor_memory())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    logger.info(f"Started concurrent processing with max {CONCURRENT_CONFIG['max_concurrent_requests']} requests")
    
    yield  # Application runs here
    
    # Shutdown
    for task in background_tasks:
        task.cancel()
    logger.info("Shutdown complete")

app = FastAPI(
    title="WhisperX Transcription API", 
    version="2.0",
    description="OpenAI-compatible transcription API using WhisperX large-v3 model with concurrent processing",
    lifespan=lifespan
)

# Add CORS middleware to allow browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Swagger UI and browser requests
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Event handlers are now handled by the lifespan context manager above

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

async def transcribe_with_concurrency_control(
    audio_path: str,
    language: Optional[str],
    enable_diarization: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    enable_profanity_filter: bool,
    request_id: str
) -> (List[Dict], Any, int, List[Dict]):
    """Wrapper for transcribe_and_diarize with concurrency control and queuing"""
    
    # Log when waiting for semaphore (queued state)
    queue_start_time = time.time()
    logger.info(f"Request {request_id}: Waiting for processing slot...")
    
    async with processing_semaphore:
        # Log start of processing and queue time
        start_time = time.time()
        queue_wait_time = start_time - queue_start_time
        if queue_wait_time > 0.1:  # Only log if there was significant queue time
            logger.info(f"Request {request_id}: Waited {queue_wait_time:.2f}s in queue, now processing...")
        else:
            logger.info(f"Request {request_id}: Processing immediately...")
        
        active_requests[request_id] = RequestState(request_id, start_time)
        
        try:
            # Check GPU memory before processing
            memory_info = get_gpu_memory_info()
            if "error" not in memory_info:
                logger.info(f"Request {request_id}: Starting processing. GPU Memory: {memory_info['free_gb']:.1f}GB free")
            
            # Call the actual transcription function
            result = await run_in_threadpool(
                transcribe_and_diarize,
                audio_path=audio_path,
                language=language,
                enable_diarization=enable_diarization,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                enable_profanity_filter=enable_profanity_filter
            )
            
            # Log completion
            processing_time = time.time() - start_time
            logger.info(f"Request {request_id}: Completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Request {request_id}: Failed with error: {e}")
            raise
        finally:
            # Clean up
            if request_id in active_requests:
                del active_requests[request_id]
            torch.cuda.empty_cache()

def transcribe_and_diarize(
    audio_path: str,
    language: Optional[str],
    enable_diarization: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    # Removed unsupported parameters
    enable_profanity_filter: bool
) -> (List[Dict], Any, int, List[Dict]):
    # 1. Transcribe with faster-whisper
    transcribe_args = {
        "language": language,
    }
    # Note: Profanity filtering not directly supported by WhisperX transcribe
    # This parameter is kept for API compatibility but doesn't affect transcription

    audio = whisperx.load_audio(audio_path)
    # Use optimized batch_size from config
    result = model.transcribe(
        audio, 
        batch_size=WHISPERX_CONFIG["batch_size"],
        **transcribe_args
    )
    
    # WhisperX returns a result dict with 'segments' key
    logger.info(f"Transcribe result type: {type(result)}")
    logger.info(f"Transcribe result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
    
    segments = result.get("segments", [])
    transcript_for_alignment = segments  # WhisperX segments are already in the correct format

    # 2. Align transcription
    logger.info("Aligning transcription...")
    detected_language = result.get("language", language or "en")
    
    # Load alignment model (use custom if specified)
    align_model_name = WHISPERX_CONFIG["align_model"] if WHISPERX_CONFIG["align_model"] else None
    if align_model_name:
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language, 
            device="cuda",
            model_name=align_model_name
        )
    else:
        model_a, metadata = whisperx.load_align_model(language_code=detected_language, device="cuda")
    
    aligned_result = whisperx.align(
        transcript_for_alignment, 
        model_a, 
        metadata, 
        audio, 
        "cuda", 
        return_char_alignments=WHISPERX_CONFIG["return_char_alignments"],
        interpolate_method=WHISPERX_CONFIG["interpolate_method"]
    )
    logger.info("Alignment complete.")

    # 3. Diarization and speaker assignment
    num_speakers = 0
    if enable_diarization and HUGGINGFACE_TOKEN:
        logger.info("Performing speaker diarization...")
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HUGGINGFACE_TOKEN, device="cuda")
        
        diarize_params = {}
        if min_speakers is not None:
            diarize_params["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_params["max_speakers"] = max_speakers
        
        diarization = diarize_model(audio, **diarize_params)
        
        result_with_speakers = whisperx.assign_word_speakers(diarization, aligned_result)
        final_segments = result_with_speakers.get("segments", [])
        raw_words = result_with_speakers.get("word_segments", [])
        
        # Extract number of speakers from diarization DataFrame
        if hasattr(diarization, 'speaker'):
            # diarization is a DataFrame with a 'speaker' column
            unique_speakers = diarization['speaker'].unique()
            num_speakers = len(unique_speakers)
        elif hasattr(diarization, 'df') and hasattr(diarization.df, 'speaker'):
            # Alternative: diarization might wrap a df attribute
            unique_speakers = diarization.df['speaker'].unique()
            num_speakers = len(unique_speakers)
        else:
            # Fallback: try to infer from the result segments
            speakers_in_segments = set()
            for seg in final_segments:
                if 'speaker' in seg:
                    speakers_in_segments.add(seg['speaker'])
            num_speakers = len(speakers_in_segments)
        
        logger.info(f"Found {num_speakers} speakers.")
    else:
        final_segments = aligned_result.get("segments", [])
        raw_words = aligned_result.get("word_segments", [])

    # Format words for API response compatibility
    word_segments = []
    if raw_words:
        for word in raw_words:
            raw_text = word.get("word", "")
            word_segments.append({
                "start": word.get("start"),
                "end": word.get("end"),
                "text": raw_text,
                "decorated_text": normalize_text_for_display(raw_text),
                "word_prob": word.get("score"),
                "speaker": word.get("speaker")
            })

    return final_segments, result, num_speakers, word_segments


# --- API Endpoints ---
@app.get("/")
async def root():
    return {
        "message": "WhisperX Transcription API is running.",
        "model": "whisperx-large-v3",
        "openai_compatible": True,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    memory_info = get_gpu_memory_info()
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "actual_model": "whisperx-large-v3",
        "models_loaded": {
            "transcription": model is not None,
            "diarization": HUGGINGFACE_TOKEN is not None
        },
        "concurrent_processing": {
            "max_concurrent_requests": CONCURRENT_CONFIG["max_concurrent_requests"],
            "active_requests": len(active_requests),
            "available_slots": CONCURRENT_CONFIG["max_concurrent_requests"] - len(active_requests),
            "queue_enabled": CONCURRENT_CONFIG["enable_request_queuing"]
        },
        "gpu_memory": memory_info,
        "optimization_profile": "customer_feedback_accuracy",
        "compute_type": WHISPERX_CONFIG["compute_type"],
        "note": "Model parameter is ignored - always uses WhisperX large-v3"
    }

@app.post("/v1/audio/transcriptions")
async def main_transcription_endpoint(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"), # OpenAI compatibility - ignored (always uses large-v3)
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None), # Not used by whisperx transcribe
    response_format: str = Form("json"),
    temperature: float = Form(0.0), # Not used by whisperx transcribe
    timestamp_granularities: List[str] = Form(["segment"]), # Compatibility
    enable_diarization: bool = Form(False),
    min_speakers: Optional[str] = Form(None),
    max_speakers: Optional[str] = Form(None),
    # Parameters not directly used by whisperx transcribe are now removed from the call
    output_content: str = Form("both", description="Control response content: 'text_only', 'timestamps_only', or 'both'."),
    enable_profanity_filter: bool = Form(False, description="Enable the profanity filter to censorResults.")
):
    global request_counter
    request_counter += 1
    request_id = f"req_{request_counter}_{int(time.time())}"
    start_time = datetime.now()
    
    # Note: Semaphore will handle queuing automatically - no manual capacity check needed
    
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
        

        segments, result, num_speakers, words = await transcribe_with_concurrency_control(
            audio_path=converted_audio_path,
            language=language,
            enable_diarization=enable_diarization,
            min_speakers=min_speakers_int,
            max_speakers=max_speakers_int,
            enable_profanity_filter=enable_profanity_filter,
            request_id=request_id
        )

        full_text = " ".join(s['text'].strip() for s in segments)

        # Format response based on 'response_format'
        if response_format in ["json", "verbose_json"]:
            content = {}
            if response_format == "verbose_json":
                content.update({
                    "task": "transcribe",
                    "language": result.get("language", "en"),
                    "duration": result.get("duration", 0.0),
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
                    # Clean segments if word-level timestamps not requested
                    if "word" not in timestamp_granularities:
                        # Remove word-level data from segments
                        clean_segments = []
                        for seg in segments:
                            clean_seg = {
                                "start": seg["start"],
                                "end": seg["end"],
                                "text": seg["text"]
                            }
                            if "speaker" in seg:
                                clean_seg["speaker"] = seg["speaker"]
                            clean_segments.append(clean_seg)
                        content["segments"] = clean_segments
                    else:
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
    model: str = Form("whisper-1"), # OpenAI compatibility - ignored (always uses large-v3)
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: List[str] = Form(["segment"]),
    enable_diarization: bool = Form(False),
    min_speakers: Optional[str] = Form(None),
    max_speakers: Optional[str] = Form(None),
    output_content: str = Form("both"),
    enable_profanity_filter: bool = Form(False)
):
    # This alias maintains backward compatibility by simply calling the new endpoint.
    # It now accepts all the same parameters as the main endpoint for full functionality.

    # Treat empty string for language as None to trigger automatic language detection
    if not language:
        language = None
        
    return await main_transcription_endpoint(
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
        enable_diarization=enable_diarization,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        output_content=output_content,
        enable_profanity_filter=enable_profanity_filter
    )

# New endpoint for monitoring concurrent processing status
@app.get("/processing-status")
async def get_processing_status():
    """Get current processing status and queue information"""
    return {
        "timestamp": datetime.now().isoformat(),
        "concurrent_processing": {
            "max_concurrent_requests": CONCURRENT_CONFIG["max_concurrent_requests"],
            "active_requests": len(active_requests),
            "available_slots": CONCURRENT_CONFIG["max_concurrent_requests"] - len(active_requests),
            "queue_enabled": CONCURRENT_CONFIG["enable_request_queuing"],
            "memory_per_request_gb": CONCURRENT_CONFIG["memory_per_request_gb"]
        },
        "active_request_details": [
            {
                "request_id": req.request_id,
                "start_time": req.start_time,
                "duration_seconds": time.time() - req.start_time
            }
            for req in active_requests.values()
        ],
        "gpu_memory": get_gpu_memory_info(),
        "optimization_config": {
            "compute_type": WHISPERX_CONFIG["compute_type"],
            "batch_size": WHISPERX_CONFIG["batch_size"],
            "chunk_length_s": WHISPERX_CONFIG["chunk_length_s"],
            "return_char_alignments": WHISPERX_CONFIG["return_char_alignments"],
            "align_model": WHISPERX_CONFIG["align_model"],
            "segment_resolution": WHISPERX_CONFIG["segment_resolution"],
            "interpolate_method": WHISPERX_CONFIG["interpolate_method"]
        },
        "enhanced_vad_config": {
            "vad_onset": WHISPERX_CONFIG["vad_onset"],
            "vad_offset": WHISPERX_CONFIG["vad_offset"], 
            "min_segment_length": WHISPERX_CONFIG["min_segment_length"],
            "max_segment_length": WHISPERX_CONFIG["max_segment_length"],
            "speech_threshold": WHISPERX_CONFIG["speech_threshold"]
        },
        "diarization_config": {
            "speaker_smoothing_enabled": DIARIZATION_CONFIG["speaker_smoothing_enabled"],
            "speaker_confidence_threshold": DIARIZATION_CONFIG["speaker_confidence_threshold"],
            "text_normalization_mode": DIARIZATION_CONFIG["text_normalization_mode"],
            "sentiment_analysis_enabled": DIARIZATION_CONFIG["sentiment_analysis_enabled"]
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=3333,
        timeout_keep_alive=0,  # No timeout - keep connection alive as long as needed
        # Note: File size limits are handled by the deployment config (max_body_size: 4294967295 ≈ 4GB)
    )