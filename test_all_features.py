#!/usr/bin/env python3

import requests
import json
import time
import os

BASE_URL = "http://localhost:3333"
AUDIO_FILE = "/home/kyle/Desktop/whisper/Morpheus Test-20250801T195813Z-1-001/Morpheus Test/1minute_1speaker.mp3"

def test_endpoint(name, url, method="GET", files=None, data=None, expected_status=200):
    """Test an endpoint and return the result."""
    print(f"\n🧪 Testing: {name}")
    print(f"   URL: {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, files=files, data=data, timeout=120)
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == expected_status:
            print("   ✅ SUCCESS")
            return response
        else:
            print(f"   ❌ FAILED - Expected {expected_status}, got {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return None
            
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        return None

def test_transcription_feature(name, additional_params=None, expected_keys=None):
    """Test a specific transcription feature."""
    base_params = {
        "model": "whisper-1"
    }
    
    if additional_params:
        base_params.update(additional_params)
    
    files = {"file": open(AUDIO_FILE, "rb")}
    
    response = test_endpoint(
        name,
        f"{BASE_URL}/v1/audio/transcriptions",
        method="POST",
        files=files,
        data=base_params
    )
    
    files["file"].close()
    
    if response and response.status_code == 200:
        try:
            if base_params.get("response_format") in ["text", "srt", "vtt"]:
                print(f"   Content preview: {response.text[:100]}...")
                if expected_keys:
                    for key in expected_keys:
                        if key in response.text:
                            print(f"   ✅ Contains expected content: {key}")
                        else:
                            print(f"   ⚠️  Missing expected content: {key}")
            else:
                result = response.json()
                print(f"   Response keys: {list(result.keys())}")
                
                if expected_keys:
                    for key in expected_keys:
                        if key in result:
                            print(f"   ✅ Has expected key: {key}")
                        else:
                            print(f"   ⚠️  Missing expected key: {key}")
                
                # Show some key information
                if "text" in result:
                    print(f"   Text preview: {result['text'][:50]}...")
                if "speakers" in result:
                    print(f"   Speakers detected: {result['speakers']}")
                if "language" in result:
                    print(f"   Language: {result['language']}")
                if "duration" in result:
                    print(f"   Duration: {result['duration']:.2f}s")
                if "segments" in result:
                    print(f"   Segments: {len(result['segments'])}")
                if "words" in result:
                    print(f"   Words: {len(result['words'])}")
                    
        except json.JSONDecodeError:
            print(f"   Response (non-JSON): {response.text[:100]}...")
    
    return response

def main():
    print("🚀 COMPREHENSIVE API FEATURE TEST")
    print("=" * 50)
    
    # Check if audio file exists
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Audio file not found: {AUDIO_FILE}")
        return
    
    print(f"Using audio file: {os.path.basename(AUDIO_FILE)}")
    
    # Test basic endpoints
    print("\n📍 BASIC ENDPOINTS")
    test_endpoint("Root endpoint", f"{BASE_URL}/")
    test_endpoint("Health check", f"{BASE_URL}/health")
    test_endpoint("API docs", f"{BASE_URL}/docs", expected_status=200)
    
    # Test core transcription features
    print("\n📍 CORE TRANSCRIPTION FEATURES")
    
    test_transcription_feature(
        "Basic transcription (JSON)",
        {"response_format": "json"},
        ["text", "segments"]
    )
    
    test_transcription_feature(
        "Verbose JSON (OpenAI format)",
        {"response_format": "verbose_json"},
        ["task", "language", "duration", "text", "segments"]
    )
    
    test_transcription_feature(
        "Plain text output",
        {"response_format": "text"},
        ["Welcome", "Phoenix"]
    )
    
    test_transcription_feature(
        "SRT subtitle format",
        {"response_format": "srt"},
        ["1", "00:00:", "-->", "Welcome"]
    )
    
    test_transcription_feature(
        "VTT subtitle format",
        {"response_format": "vtt"},
        ["WEBVTT", "00:00:", "-->", "Welcome"]
    )
    
    # Test timestamp granularities
    print("\n📍 TIMESTAMP GRANULARITIES")
    
    test_transcription_feature(
        "Segment timestamps",
        {"response_format": "json", "timestamp_granularities": "segment"},
        ["text", "segments"]
    )
    
    test_transcription_feature(
        "Word-level timestamps",
        {"response_format": "json", "timestamp_granularities": "word"},
        ["text", "words"]
    )
    
    # Test output content options
    print("\n📍 OUTPUT CONTENT OPTIONS")
    
    test_transcription_feature(
        "Text only",
        {"response_format": "json", "output_content": "text_only"},
        ["text"]
    )
    
    test_transcription_feature(
        "Timestamps only",
        {"response_format": "json", "output_content": "timestamps_only", "timestamp_granularities": "segment"},
        ["segments"]
    )
    
    test_transcription_feature(
        "Both text and timestamps",
        {"response_format": "json", "output_content": "both", "timestamp_granularities": "word"},
        ["text", "words"]
    )
    
    # Test language features
    print("\n📍 LANGUAGE FEATURES")
    
    test_transcription_feature(
        "Auto-detect language",
        {"response_format": "verbose_json"},
        ["language"]
    )
    
    test_transcription_feature(
        "Explicit English",
        {"response_format": "verbose_json", "language": "en"},
        ["language"]
    )
    
    # Test WhisperX enhanced features
    print("\n📍 WHISPERX ENHANCED FEATURES")
    
    test_transcription_feature(
        "Speaker diarization",
        {"response_format": "json", "enable_diarization": "true"},
        ["speakers", "segments"]
    )
    
    test_transcription_feature(
        "Diarization with speaker constraints",
        {
            "response_format": "json", 
            "enable_diarization": "true",
            "min_speakers": "1",
            "max_speakers": "2"
        },
        ["speakers", "segments"]
    )
    
    test_transcription_feature(
        "Profanity filter",
        {"response_format": "json", "enable_profanity_filter": "true"},
        ["text", "segments"]
    )
    
    test_transcription_feature(
        "Combined features (diarization + profanity filter)",
        {
            "response_format": "json",
            "enable_diarization": "true",
            "enable_profanity_filter": "true",
            "timestamp_granularities": "word"
        },
        ["text", "segments", "words", "speakers"]
    )
    
    # Test SRT/VTT with speakers
    print("\n📍 SUBTITLE FORMATS WITH SPEAKERS")
    
    test_transcription_feature(
        "SRT with speaker labels",
        {"response_format": "srt", "enable_diarization": "true"},
        ["SPEAKER_", "-->", "Welcome"]
    )
    
    test_transcription_feature(
        "VTT with speaker labels",
        {"response_format": "vtt", "enable_diarization": "true"},
        ["WEBVTT", "SPEAKER_", "-->", "Welcome"]
    )
    
    # Test legacy parameters (should be accepted but not used)
    print("\n📍 LEGACY PARAMETERS (COMPATIBILITY)")
    
    test_transcription_feature(
        "With legacy parameters",
        {
            "response_format": "json",
            "temperature": "0.2",
            "prompt": "This is a test prompt"
        },
        ["text", "segments"]
    )
    
    # Test alias endpoint
    print("\n📍 ALIAS ENDPOINT")
    
    files = {"file": open(AUDIO_FILE, "rb")}
    response = test_endpoint(
        "Transcribe alias endpoint",
        f"{BASE_URL}/transcribe",
        method="POST",
        files=files,
        data={"model": "whisper-1", "response_format": "json"}
    )
    files["file"].close()
    
    print("\n🎯 TEST SUMMARY")
    print("=" * 50)
    print("All major API features have been tested!")
    print("Check the results above for any issues.")

if __name__ == "__main__":
    main() 