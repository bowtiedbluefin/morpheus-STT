#https://github.com/SYSTRAN/faster-whisper

from fastapi import FastAPI, UploadFile, File, Form, Query, Request
from faster_whisper import WhisperModel, BatchedInferencePipeline
import uvicorn
import tempfile
import os
import torch
from typing import Optional, List, Union
from enum import Enum

app = FastAPI()

# Define enums for validation
class ResponseFormat(str, Enum):
    json = "json"
    text = "text"
    srt = "srt"
    vtt = "vtt"
    verbose_json = "verbose_json"

class TimestampGranularity(str, Enum):
    segment = "segment"
    word = "word"

class OutputContent(str, Enum):
    text_only = "text_only"
    timestamps_only = "timestamps_only"
    both = "both"

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Initialize model with explicit CUDA settings
model = WhisperModel("large-v3", device="cuda", compute_type="float16", download_root="./models")
# Initialize batched model for long audio processing
batched_model = BatchedInferencePipeline(model=model)

@app.get("/")
async def root():
    return {"message": "Whisper Transcription API", "docs": "Visit /docs for API documentation"}

@app.get("/transcribe/")
async def transcribe_info():
    return {"message": "POST audio files here for transcription", "docs": "Visit /docs to test the API"}

# OpenAI-compatible endpoint with all extended features
@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form("whisper-1", description="Model to use (ignored, handled by API gateway)"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'es', 'fr') or None for auto-detection"),
    prompt: Optional[str] = Form(None, description="Initial prompt to condition the model"),
    response_format: ResponseFormat = Form(ResponseFormat.json, description="Output format"),
    temperature: float = Form(0.0, ge=0.0, le=1.0, description="Sampling temperature (0.0-1.0) - default 0.0 for deterministic output"),
    output_content: OutputContent = Form(OutputContent.both, description="What to include in JSON response (extended feature)"),
    use_batched: bool = Form(True, description="Use batched processing for long audio (recommended, default: enabled)"),
    batch_size: int = Form(8, ge=1, le=64, description="Batch size for batched processing (default: 8 for quality)"),
    max_speech_duration: float = Form(20.0, ge=5.0, le=120.0, description="Maximum speech segment duration in seconds (default: 20s for quality)")
):
    """OpenAI-compatible transcription endpoint with extended features"""
    # Parse timestamp_granularities from form data (handles array format timestamp_granularities[])
    form_data = await request.form()
    
    # Handle timestamp_granularities[] array format
    timestamp_granularities = None
    for key in form_data.keys():
        if key.startswith('timestamp_granularities'):
            if key == 'timestamp_granularities[]':
                # Array format: timestamp_granularities[]
                values = form_data.getlist(key)
                if values:
                    timestamp_granularities = values[0]  # Use first value for now
            elif key == 'timestamp_granularities':
                # Single value format
                timestamp_granularities = form_data.get(key)
            break
    
    # Default to segment if not provided (OpenAI default behavior)
    if timestamp_granularities:
        try:
            selected_granularity = TimestampGranularity(timestamp_granularities)
        except ValueError:
            selected_granularity = TimestampGranularity.segment
    else:
        selected_granularity = TimestampGranularity.segment
    
    print(f"Starting transcription with params: language={language}, prompt={prompt}, format={response_format}, temp={temperature}, granularity={selected_granularity}, content={output_content}, batched={use_batched}, batch_size={batch_size}")
    
    #Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(await file.read())
    temp_file.close()
    print(f"Saved temporary file: {temp_file.name}")

    #Process with Faster-Whisper
    print("Processing with Whisper...")
    
    # Enhanced VAD parameters optimized for quality on long audio
    enhanced_vad_params = dict(
        threshold=0.2,                              # More sensitive VAD for better speech detection
        min_silence_duration_ms=800,                # Shorter silence threshold for better segmentation
        min_speech_duration_ms=200,                 # Shorter minimum speech for better granularity
        max_speech_duration_s=max_speech_duration,  # Maximum speech segment (chunking)
        speech_pad_ms=300                           # Moderate padding around speech
    )
    
    if use_batched:
        # Use batched processing for long audio
        segments, info = batched_model.transcribe(
            temp_file.name,
            language=language,
            batch_size=batch_size,
            beam_size=8,                            # Higher beam size for better quality
            best_of=8,                              # More candidates for better quality
            vad_filter=True,
            vad_parameters=enhanced_vad_params,
            initial_prompt=prompt,
            temperature=temperature,
            word_timestamps=(selected_granularity == TimestampGranularity.word)
        )
    else:
        # Use standard processing with quality optimizations
        segments, info = model.transcribe(
            temp_file.name,
            language=language,
            beam_size=8,                            # Higher beam size for better quality
            best_of=8,                              # More candidates for better quality
            vad_filter=True,
            vad_parameters=enhanced_vad_params,
            initial_prompt=prompt,
            temperature=temperature,
            word_timestamps=(selected_granularity == TimestampGranularity.word)
        )
    print("Transcription complete")

    #Collect results based on format and granularity
    if response_format in [ResponseFormat.json, ResponseFormat.verbose_json]:
        if selected_granularity == TimestampGranularity.word:
            # Word-level timestamps
            result = {}
            
            # Add verbose metadata if verbose_json format
            if response_format == ResponseFormat.verbose_json:
                result["task"] = "transcribe"
                result["language"] = info.language
                result["duration"] = info.duration
            
            # Add text if requested
            if output_content in [OutputContent.text_only, OutputContent.both]:
                result["text"] = ""
                for segment in segments:
                    result["text"] += segment.text + " "
                result["text"] = result["text"].strip()
            
            # Add word timestamps if requested
            if output_content in [OutputContent.timestamps_only, OutputContent.both]:
                result["words"] = []
                # Reset segments iterator
                if use_batched:
                    segments, _ = batched_model.transcribe(
                        temp_file.name,
                        language=language,
                        batch_size=batch_size,
                        beam_size=8,
                        best_of=8,
                        vad_filter=True,
                        vad_parameters=enhanced_vad_params,
                        initial_prompt=prompt,
                        temperature=temperature,
                        word_timestamps=True
                    )
                else:
                    segments, _ = model.transcribe(
                        temp_file.name,
                        language=language,
                        beam_size=8,
                        best_of=8,
                        vad_filter=True,
                        vad_parameters=enhanced_vad_params,
                        initial_prompt=prompt,
                        temperature=temperature,
                        word_timestamps=True
                    )
                
                for segment in segments:
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            result["words"].append({
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "probability": word.probability
                            })
                    else:
                        # Fallback if word timestamps not available
                        words = segment.text.split()
                        word_duration = (segment.end - segment.start) / len(words) if words else 0
                        for i, word in enumerate(words):
                            result["words"].append({
                                "word": word,
                                "start": segment.start + (i * word_duration),
                                "end": segment.start + ((i + 1) * word_duration),
                                "probability": None
                            })
        else:
            # Segment-level timestamps
            result = {}
            
            # Add verbose metadata if verbose_json format
            if response_format == ResponseFormat.verbose_json:
                result["task"] = "transcribe"
                result["language"] = info.language
                result["duration"] = info.duration
            
            # Add text if requested
            if output_content in [OutputContent.text_only, OutputContent.both]:
                result["text"] = ""
                for segment in segments:
                    result["text"] += segment.text + " "
                result["text"] = result["text"].strip()
            
            # Add segment timestamps if requested
            if output_content in [OutputContent.timestamps_only, OutputContent.both]:
                result["segments"] = []
                # Reset segments iterator if we already consumed it
                if output_content == OutputContent.timestamps_only:
                    if use_batched:
                        segments, _ = batched_model.transcribe(
                            temp_file.name,
                            language=language,
                            batch_size=batch_size,
                            beam_size=8,                            # Higher beam size for better quality
                            best_of=8,                              # More candidates for better quality
                            vad_filter=True,
                            vad_parameters=enhanced_vad_params,
                            initial_prompt=prompt,
                            temperature=temperature,
                            word_timestamps=False
                        )
                    else:
                        segments, _ = model.transcribe(
                            temp_file.name,
                            language=language,
                            beam_size=8,                            # Higher beam size for better quality
                            best_of=8,                              # More candidates for better quality
                            vad_filter=True,
                            vad_parameters=enhanced_vad_params,
                            initial_prompt=prompt,
                            temperature=temperature,
                            word_timestamps=False
                        )
                
                for segment in segments:
                    result["segments"].append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    })
    
    elif response_format == ResponseFormat.text:
        # Plain text format
        result = ""
        for segment in segments:
            result += segment.text + " "
        result = result.strip()
    
    elif response_format == ResponseFormat.srt:
        # SRT subtitle format
        result = ""
        for i, segment in enumerate(segments, 1):
            start_time = format_timestamp_srt(segment.start)
            end_time = format_timestamp_srt(segment.end)
            result += f"{i}\n{start_time} --> {end_time}\n{segment.text.strip()}\n\n"
    
    elif response_format == ResponseFormat.vtt:
        # WebVTT format
        result = "WEBVTT\n\n"
        for segment in segments:
            start_time = format_timestamp_vtt(segment.start)
            end_time = format_timestamp_vtt(segment.end)
            result += f"{start_time} --> {end_time}\n{segment.text.strip()}\n\n"
    
    #Clean up
    os.unlink(temp_file.name)
    
    # Return appropriate content type
    if response_format in [ResponseFormat.json, ResponseFormat.verbose_json]:
        return result
    else:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content=result, media_type="text/plain")

# Alias endpoint for backward compatibility and convenience
@app.post("/transcribe/")
async def transcribe_audio_alias(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form("whisper-1", description="Model to use (ignored, handled by API gateway)"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'es', 'fr') or None for auto-detection"),
    prompt: Optional[str] = Form(None, description="Initial prompt to condition the model"),
    response_format: ResponseFormat = Form(ResponseFormat.json, description="Output format"),
    temperature: float = Form(0.0, ge=0.0, le=1.0, description="Sampling temperature (0.0-1.0) - default 0.0 for deterministic output"),
    output_content: OutputContent = Form(OutputContent.both, description="What to include in JSON response (extended feature)"),
    use_batched: bool = Form(True, description="Use batched processing for long audio (recommended, default: enabled)"),
    batch_size: int = Form(8, ge=1, le=64, description="Batch size for batched processing (default: 8 for quality)"),
    max_speech_duration: float = Form(20.0, ge=5.0, le=120.0, description="Maximum speech segment duration in seconds (default: 20s for quality)")
):
    """Alias to the main transcription endpoint - same functionality"""
    return await transcribe_audio(
        request=request,
        file=file,
        model_name=model_name,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        output_content=output_content,
        use_batched=use_batched,
        batch_size=batch_size,
        max_speech_duration=max_speech_duration
    )

def format_timestamp_srt(seconds):
    """Format timestamp for SRT format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def format_timestamp_vtt(seconds):
    """Format timestamp for VTT format (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3333)