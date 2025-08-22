#!/usr/bin/env python3
"""
Test Speaker Recognition Accuracy
=================================
Tests the improved speaker recognition against known speaker counts
in the Morpheus test dataset.
"""

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from working_gpu_diarization_server import WorkingGPUDiarizationServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test cases with known speaker counts
TEST_CASES = [
    {"file": "1minute_1speaker.mp3", "expected_speakers": 1, "name": "1min-1spk"},
    {"file": "10minutes_2speakers.mp3", "expected_speakers": 2, "name": "10min-2spk"}, 
    {"file": "5minutes_5speakers.mp3", "expected_speakers": 5, "name": "5min-5spk"},
    # Skip longer tests for now - {"file": "30minutes_7speakers.mp3", "expected_speakers": 7, "name": "30min-7spk"},
]

async def test_speaker_accuracy():
    """Test speaker recognition accuracy"""
    print("="*80)
    print("TESTING SPEAKER RECOGNITION IMPROVEMENTS")
    print("="*80)
    
    # Initialize server
    server = WorkingGPUDiarizationServer()
    server.load_models()
    
    test_dir = Path("Morpheus Test-20250801T195813Z-1-001/Morpheus Test")
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return
    
    results = []
    
    for test_case in TEST_CASES:
        audio_file = test_dir / test_case["file"]
        if not audio_file.exists():
            logger.warning(f"Test file not found: {audio_file}")
            continue
            
        print(f"\n📊 Testing: {test_case['name']}")
        print(f"   File: {test_case['file']}")
        print(f"   Expected speakers: {test_case['expected_speakers']}")
        
        try:
            start_time = time.time()
            
            # Process with diarization
            result = await server.transcribe_with_diarization(str(audio_file), enable_diarization=True)
            
            processing_time = time.time() - start_time
            
            # Extract speaker info
            segments = result.get("result", {}).get("segments", [])
            detected_speakers = set()
            confident_speakers = set()
            
            for segment in segments:
                if "speaker" in segment:
                    detected_speakers.add(segment["speaker"])
                    # Count only confident speakers
                    if segment.get("speaker_confidence", 0) >= server.speaker_confidence_threshold:
                        confident_speakers.add(segment["speaker"])
            
            # Results
            accuracy = "✅ PERFECT" if len(confident_speakers) == test_case["expected_speakers"] else "❌ INACCURATE"
            if len(confident_speakers) > test_case["expected_speakers"]:
                accuracy += " (Over-detection)"
            elif len(confident_speakers) < test_case["expected_speakers"]:
                accuracy += " (Under-detection)"
            
            print(f"   Detected speakers (all): {len(detected_speakers)}")
            print(f"   Confident speakers: {len(confident_speakers)}")
            print(f"   Accuracy: {accuracy}")
            print(f"   Processing time: {processing_time:.1f}s")
            
            results.append({
                "test": test_case["name"],
                "expected": test_case["expected_speakers"],
                "detected_all": len(detected_speakers),
                "detected_confident": len(confident_speakers),
                "accurate": len(confident_speakers) == test_case["expected_speakers"],
                "processing_time": processing_time
            })
            
        except Exception as e:
            logger.error(f"Test failed for {test_case['name']}: {e}")
            results.append({
                "test": test_case["name"],
                "expected": test_case["expected_speakers"],
                "detected_all": 0,
                "detected_confident": 0,
                "accurate": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    accurate_tests = sum(1 for r in results if r.get("accurate", False))
    total_tests = len(results)
    accuracy_rate = (accurate_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Tests passed: {accurate_tests}/{total_tests} ({accuracy_rate:.1f}%)")
    print(f"Speaker confidence threshold: {server.speaker_confidence_threshold}")
    print(f"Clustering threshold: {server.clustering_threshold}")
    print(f"Segmentation threshold: {server.segmentation_threshold}")
    print(f"Min speaker duration: {server.min_speaker_duration}s")
    
    # Detailed results
    print("\nDetailed Results:")
    for result in results:
        status = "✅" if result.get("accurate", False) else "❌"
        print(f"  {status} {result['test']}: {result['detected_confident']}/{result['expected']} speakers")

if __name__ == "__main__":
    asyncio.run(test_speaker_accuracy()) 