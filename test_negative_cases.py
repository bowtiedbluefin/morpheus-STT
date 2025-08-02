#!/usr/bin/env python3

import requests
import json
import os

BASE_URL = "http://localhost:3333"
AUDIO_FILE = "/home/kyle/Desktop/whisper/Morpheus Test-20250801T195813Z-1-001/Morpheus Test/1minute_1speaker.mp3"

def test_feature_disabled(name, params, should_not_contain=None, should_contain=None):
    """Test that a feature is properly disabled."""
    print(f"\n🧪 Testing: {name}")
    
    files = {"file": open(AUDIO_FILE, "rb")}
    
    try:
        response = requests.post(f"{BASE_URL}/v1/audio/transcriptions", files=files, data=params, timeout=120)
        files["file"].close()
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ REQUEST SUCCESS")
            
            # Handle different response formats
            if params.get("response_format") in ["text", "srt", "vtt"]:
                content = response.text
                print(f"   Content preview: {content[:100]}...")
                
                if should_not_contain:
                    for item in should_not_contain:
                        if item in content:
                            print(f"   ❌ FAIL: Found disabled content '{item}'")
                        else:
                            print(f"   ✅ PASS: Correctly excluded '{item}'")
                
                if should_contain:
                    for item in should_contain:
                        if item in content:
                            print(f"   ✅ PASS: Contains expected '{item}'")
                        else:
                            print(f"   ❌ FAIL: Missing expected '{item}'")
            
            else:
                # JSON response
                try:
                    result = response.json()
                    print(f"   Response keys: {list(result.keys())}")
                    
                    if should_not_contain:
                        for key in should_not_contain:
                            if key in result:
                                print(f"   ❌ FAIL: Found disabled key '{key}': {result[key]}")
                            else:
                                print(f"   ✅ PASS: Correctly excluded key '{key}'")
                    
                    if should_contain:
                        for key in should_contain:
                            if key in result:
                                print(f"   ✅ PASS: Contains expected key '{key}'")
                            else:
                                print(f"   ❌ FAIL: Missing expected key '{key}'")
                    
                    # Show what we got
                    if "text" in result:
                        print(f"   Text: {result['text'][:50]}...")
                    if "speakers" in result:
                        print(f"   Speakers: {result['speakers']}")
                    if "segments" in result and len(result['segments']) > 0:
                        print(f"   First segment: {result['segments'][0]}")
                    if "words" in result:
                        print(f"   Words count: {len(result['words'])}")
                        
                except json.JSONDecodeError:
                    print(f"   Non-JSON response: {response.text[:100]}...")
        else:
            print(f"   ❌ REQUEST FAILED: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        if 'files' in locals():
            files["file"].close()

def main():
    print("🚀 NEGATIVE FEATURE TESTING")
    print("=" * 50)
    print("Testing that disabled features are properly excluded")
    
    # Check if audio file exists
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Audio file not found: {AUDIO_FILE}")
        return
    
    print(f"Using audio file: {os.path.basename(AUDIO_FILE)}")
    
    print("\n📍 DIARIZATION DISABLED TESTS")
    
    # Test diarization explicitly disabled
    test_feature_disabled(
        "Diarization explicitly disabled (false)",
        {
            "model": "whisper-1",
            "response_format": "json",
            "enable_diarization": "false"
        },
        should_not_contain=["speakers"],
        should_contain=["text", "segments"]
    )
    
    # Test diarization not specified (should default to disabled)
    test_feature_disabled(
        "Diarization not specified (default disabled)",
        {
            "model": "whisper-1",
            "response_format": "json"
        },
        should_not_contain=["speakers"],
        should_contain=["text", "segments"]
    )
    
    # Test SRT without diarization (no speaker labels)
    test_feature_disabled(
        "SRT without diarization (no speaker labels)",
        {
            "model": "whisper-1",
            "response_format": "srt",
            "enable_diarization": "false"
        },
        should_not_contain=["SPEAKER_", "[SPEAKER"],
        should_contain=["Welcome", "-->"]
    )
    
    print("\n📍 TIMESTAMP DISABLED TESTS")
    
    # Test text-only output (no timestamps)
    test_feature_disabled(
        "Text only output (no timestamps)",
        {
            "model": "whisper-1",
            "response_format": "json",
            "output_content": "text_only"
        },
        should_not_contain=["segments", "words"],
        should_contain=["text"]
    )
    
    # Test plain text format (inherently no timestamps)
    test_feature_disabled(
        "Plain text format (inherently no timestamps)",
        {
            "model": "whisper-1",
            "response_format": "text"
        },
        should_not_contain=["00:00:", "-->", "{", "}"],
        should_contain=["Welcome", "Phoenix"]
    )
    
    print("\n📍 WORD TIMESTAMPS DISABLED TESTS")
    
    # Test segment timestamps only (no word timestamps)
    test_feature_disabled(
        "Segment timestamps only (no word-level)",
        {
            "model": "whisper-1",
            "response_format": "json",
            "timestamp_granularities": "segment",
            "output_content": "both"
        },
        should_not_contain=["words"],
        should_contain=["text", "segments"]
    )
    
    # Test timestamps only without word granularity
    test_feature_disabled(
        "Timestamps only (segment level)",
        {
            "model": "whisper-1",
            "response_format": "json",
            "output_content": "timestamps_only",
            "timestamp_granularities": "segment"
        },
        should_not_contain=["text", "words"],
        should_contain=["segments"]
    )
    
    print("\n📍 COMBINED DISABLED FEATURES")
    
    # Test everything minimal (no diarization, no word timestamps, text only)
    test_feature_disabled(
        "Minimal response (no diarization, text only)",
        {
            "model": "whisper-1",
            "response_format": "json",
            "enable_diarization": "false",
            "output_content": "text_only"
        },
        should_not_contain=["speakers", "segments", "words"],
        should_contain=["text"]
    )
    
    # Test verbose JSON without diarization
    test_feature_disabled(
        "Verbose JSON without diarization",
        {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "enable_diarization": "false"
        },
        should_not_contain=["speakers"],
        should_contain=["task", "language", "duration", "text", "segments"]
    )
    
    print("\n📍 EDGE CASES")
    
    # Test contradictory parameters (should respect the disable)
    test_feature_disabled(
        "Contradictory params (diarization false but min_speakers set)",
        {
            "model": "whisper-1",
            "response_format": "json",
            "enable_diarization": "false",
            "min_speakers": "2",
            "max_speakers": "4"
        },
        should_not_contain=["speakers"],
        should_contain=["text", "segments"]
    )
    
    print("\n🎯 NEGATIVE TEST SUMMARY")
    print("=" * 50)
    print("Verified that disabled features are properly excluded!")
    print("Check results above for any unexpected behavior.")

if __name__ == "__main__":
    main() 