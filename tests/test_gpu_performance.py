#!/usr/bin/env python3
"""
GPU Diarization Performance Test Script
======================================
Test script to verify GPU acceleration is working and measure performance improvements
"""

import requests
import time
import json
from pathlib import Path

def test_server_health():
    """Test if server is running and GPU-optimized"""
    try:
        response = requests.get("http://localhost:3337/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("🔍 Server Health Check:")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   GPU Available: {health_data.get('gpu_available', False)}")
            print(f"   Device: {health_data.get('device', 'unknown')}")
            print(f"   Models Loaded:")
            for model, loaded in health_data.get('models_loaded', {}).items():
                print(f"     - {model}: {'✅' if loaded else '❌'}")
            print(f"   GPU Optimizations:")
            for fix in health_data.get('fixes_applied', []):
                if 'GPU' in fix:
                    print(f"     - ✅ {fix}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_transcription_performance(audio_file_path):
    """Test transcription with diarization and measure performance"""
    if not Path(audio_file_path).exists():
        print(f"❌ Audio file not found: {audio_file_path}")
        return False
        
    try:
        print(f"\n🚀 Testing GPU-accelerated diarization with: {audio_file_path}")
        start_time = time.time()
        
        with open(audio_file_path, 'rb') as f:
            files = {'file': f}
            data = {'enable_diarization': 'true', 'response_format': 'json'}
            
            response = requests.post(
                "http://localhost:3337/v1/audio/transcriptions",
                files=files,
                data=data,
                timeout=300  # 5 minute timeout
            )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Transcription completed in {total_time:.1f}s")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Device: {result.get('processing_info', {}).get('device', 'unknown')}")
            print(f"   Language: {result.get('processing_info', {}).get('language_detected', 'unknown')}")
            
            # Count speakers
            speakers = set()
            for segment in result.get('result', {}).get('segments', []):
                if 'speaker' in segment:
                    speakers.add(segment['speaker'])
            print(f"   Speakers detected: {len(speakers)}")
            print(f"   🎯 PERFORMANCE: {total_time:.1f}s total (should be much faster than 996s!)")
            
            return True
        else:
            print(f"❌ Transcription failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 GPU Diarization Performance Test")
    print("=" * 40)
    
    # Test server health
    if not test_server_health():
        print("\n❌ Server not ready. Please start the server first:")
        print("   HUGGINGFACE_TOKEN=your_token ./fixed_gpu_start.sh")
        exit(1)
    
    print("\n📁 Looking for test audio files...")
    
    # Look for any audio files in current directory
    audio_extensions = ['.wav', '.mp3', '.m4a', '.mp4', '.flac']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path('.').glob(f'*{ext}'))
    
    if not audio_files:
        print("❌ No audio files found in current directory")
        print("   Please add an audio file (.wav, .mp3, .m4a, .mp4, or .flac) to test")
        exit(1)
    
    # Test with first audio file found
    test_file = audio_files[0]
    print(f"📄 Using test file: {test_file}")
    
    success = test_transcription_performance(test_file)
    
    if success:
        print("\n🎉 GPU acceleration test completed successfully!")
        print("💡 Compare this time to your previous 996s - it should be MUCH faster!")
    else:
        print("\n❌ Test failed - check server logs for issues") 