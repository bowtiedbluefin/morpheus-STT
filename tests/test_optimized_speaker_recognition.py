#!/usr/bin/env python3
"""
Test Optimized Speaker Recognition
=================================
Tests the fully optimized speaker recognition with all improvements:
- Speaker Confidence Threshold (SR-TH) = 0.8
- Clustering Threshold = 0.7
- Speaker Smoothing
- VAD Validation  
- Hierarchical Clustering
- Enhanced Spurious Speaker Filtering

Compares against Deepgram golden samples.
"""

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from working_gpu_diarization_server import WorkingGPUDiarizationServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_DIR = Path("Morpheus Test-20250801T195813Z-1-001/Morpheus Test")

TEST_CASES = [
    {
        "name": "5min-5speakers",
        "audio_file": "5minutes_5speakers.mp3",
        "golden_file": "5minutes_deepgram.json", 
        "expected_speakers": 5
    },
    {
        "name": "10min-2speakers", 
        "audio_file": "10minutes_2speakers.mp3",
        "golden_file": "10minutes_deepgram.json",
        "expected_speakers": 2
    },
    {
        "name": "30min-7speakers",
        "audio_file": "30minutes_7speakers.mp3", 
        "golden_file": "30minutes_deepgram.json",
        "expected_speakers": 7
    }
]

def load_deepgram_golden(golden_path: Path) -> dict:
    """Load Deepgram golden reference"""
    try:
        with open(golden_path, 'r') as f:
            data = json.load(f)
        
        # Extract speaker information from Deepgram format
        speakers = set()
        words = []
        
        if 'results' in data and 'channels' in data['results']:
            for channel in data['results']['channels']:
                for alternative in channel.get('alternatives', []):
                    for word_info in alternative.get('words', []):
                        speaker_id = word_info.get('speaker', 0)
                        speakers.add(f"SPEAKER_{speaker_id:02d}")
                        words.append({
                            'word': word_info.get('word', ''),
                            'speaker': f"SPEAKER_{speaker_id:02d}",
                            'confidence': word_info.get('speaker_confidence', 0.5),
                            'start': word_info.get('start', 0),
                            'end': word_info.get('end', 0)
                        })
        
        return {
            'speakers': list(speakers),
            'speaker_count': len(speakers),
            'words': words
        }
    except Exception as e:
        logger.error(f"Failed to load golden reference {golden_path}: {e}")
        return {'speakers': [], 'speaker_count': 0, 'words': []}

def analyze_speaker_distribution(result: dict) -> dict:
    """Analyze speaker distribution in result"""
    speaker_stats = {}
    segments = result.get('result', {}).get('segments', [])
    
    for segment in segments:
        speaker = segment.get('speaker', 'UNKNOWN')
        confidence = segment.get('speaker_confidence', 0.0)
        duration = segment.get('end', 0) - segment.get('start', 0)
        
        if 'words' in segment:
            word_count = len(segment['words'])
        else:
            word_count = 0
        
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                'total_duration': 0,
                'total_words': 0,
                'confidences': [],
                'segment_count': 0
            }
        
        speaker_stats[speaker]['total_duration'] += duration
        speaker_stats[speaker]['total_words'] += word_count
        speaker_stats[speaker]['confidences'].append(confidence)
        speaker_stats[speaker]['segment_count'] += 1
    
    # Calculate averages
    for speaker, stats in speaker_stats.items():
        if stats['confidences']:
            stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
        else:
            stats['avg_confidence'] = 0.0
    
    return speaker_stats

def compare_with_golden(result: dict, golden: dict, server_threshold: float) -> dict:
    """Compare result with golden reference using server's actual threshold"""
    result_speakers = set()
    confident_speakers = set()
    
    # Extract speakers from our result
    segments = result.get('result', {}).get('segments', [])
    for segment in segments:
        if 'speaker' in segment:
            result_speakers.add(segment['speaker'])
            # Use server's actual confidence threshold, not hardcoded 0.8
            if segment.get('speaker_confidence', 0) >= server_threshold:
                confident_speakers.add(segment['speaker'])
    
    golden_speakers = set(golden.get('speakers', []))
    
    comparison = {
        'golden_count': len(golden_speakers),
        'detected_all': len(result_speakers),
        'detected_confident': len(confident_speakers),
        'accuracy_all': len(result_speakers) == len(golden_speakers),
        'accuracy_confident': len(confident_speakers) == len(golden_speakers),
        'over_detection_all': len(result_speakers) > len(golden_speakers),
        'over_detection_confident': len(confident_speakers) > len(golden_speakers),
        'under_detection_all': len(result_speakers) < len(golden_speakers),
        'under_detection_confident': len(confident_speakers) < len(golden_speakers)
    }
    
    return comparison

async def test_optimized_recognition():
    """Test optimized speaker recognition"""
    print("="*80)
    print("TESTING OPTIMIZED SPEAKER RECOGNITION")
    print("="*80)
    print("🔧 Optimizations Applied:")
    print("   ✅ Speaker Confidence Threshold (SR-TH) = 0.8")
    print("   ✅ Clustering Threshold = 0.7")
    print("   ✅ Speaker Smoothing (reduce A→B→A switches)")
    print("   ✅ VAD Validation")
    print("   ✅ Hierarchical Clustering")
    print("   ✅ Enhanced Spurious Speaker Filtering")
    print("="*80)
    
    if not TEST_DIR.exists():
        logger.error(f"Test directory not found: {TEST_DIR}")
        return
    
    # Initialize server with optimized settings
    server = WorkingGPUDiarizationServer()
    server.load_models()
    
    print(f"📊 Configuration:")
    print(f"   Clustering Threshold: {server.clustering_threshold}")
    print(f"   Segmentation Threshold: {server.segmentation_threshold}")
    print(f"   Speaker Confidence Threshold: {server.speaker_confidence_threshold}")
    print(f"   Min Speaker Duration: {server.min_speaker_duration}s")
    print(f"   Speaker Smoothing: {server.speaker_smoothing_enabled}")
    print(f"   Min Switch Duration: {server.min_switch_duration}s")
    
    results = []
    
    for test_case in TEST_CASES:
        audio_path = TEST_DIR / test_case["audio_file"]
        golden_path = TEST_DIR / test_case["golden_file"]
        
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            continue
            
        if not golden_path.exists():
            logger.warning(f"Golden file not found: {golden_path}")
            continue
        
        print(f"\n🎯 Testing: {test_case['name']}")
        print(f"   Audio: {test_case['audio_file']}")
        print(f"   Expected speakers: {test_case['expected_speakers']}")
        
        try:
            # Load golden reference
            golden = load_deepgram_golden(golden_path)
            print(f"   Golden speakers: {golden['speaker_count']}")
            
            # Process with optimized diarization
            start_time = time.time()
            result = await server.transcribe_with_diarization(str(audio_path), enable_diarization=True)
            processing_time = time.time() - start_time
            
            # Analyze result
            speaker_stats = analyze_speaker_distribution(result)
            comparison = compare_with_golden(result, golden, server.speaker_confidence_threshold)
            
            # Results
            print(f"   Processing time: {processing_time:.1f}s")
            print(f"   Detected speakers (all): {comparison['detected_all']}")
            print(f"   Confident speakers (≥{server.speaker_confidence_threshold}): {comparison['detected_confident']}")
            
            if comparison['accuracy_confident']:
                accuracy_status = "✅ PERFECT"
            elif comparison['over_detection_confident']:
                accuracy_status = "⚠️  OVER-DETECTION"  
            elif comparison['under_detection_confident']:
                accuracy_status = "⚠️  UNDER-DETECTION"
            else:
                accuracy_status = "❌ ERROR"
            
            print(f"   Accuracy: {accuracy_status}")
            
            # Speaker breakdown
            print(f"   Speaker Distribution:")
            for speaker, stats in sorted(speaker_stats.items()):
                print(f"     {speaker}: {stats['total_duration']:.1f}s, "
                     f"{stats['total_words']} words, "
                     f"conf: {stats['avg_confidence']:.3f}")
            
            results.append({
                'test': test_case['name'],
                'expected': test_case['expected_speakers'],
                'golden_count': golden['speaker_count'],
                'detected_all': comparison['detected_all'],
                'detected_confident': comparison['detected_confident'],
                'accurate': comparison['accuracy_confident'],
                'processing_time': processing_time,
                'speaker_stats': speaker_stats
            })
            
        except Exception as e:
            logger.error(f"Test failed for {test_case['name']}: {e}")
            results.append({
                'test': test_case['name'],
                'expected': test_case['expected_speakers'],
                'detected_all': 0,
                'detected_confident': 0,
                'accurate': False,
                'error': str(e)
            })
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    accurate_tests = sum(1 for r in results if r.get('accurate', False))
    total_tests = len([r for r in results if 'error' not in r])
    accuracy_rate = (accurate_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"✅ Tests passed: {accurate_tests}/{total_tests} ({accuracy_rate:.1f}%)")
    print(f"📊 Average processing time: {sum(r.get('processing_time', 0) for r in results) / len(results):.1f}s")
    
    print(f"\nDetailed Results:")
    for result in results:
        if 'error' in result:
            print(f"  ❌ {result['test']}: ERROR - {result['error']}")
        else:
            status = "✅" if result.get('accurate', False) else "❌"
            print(f"  {status} {result['test']}: "
                 f"Detected {result['detected_confident']}/{result['expected']} speakers "
                 f"({result['processing_time']:.1f}s)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"optimized_speaker_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'configuration': {
                'clustering_threshold': server.clustering_threshold,
                'segmentation_threshold': server.segmentation_threshold,
                'speaker_confidence_threshold': server.speaker_confidence_threshold,
                'min_speaker_duration': server.min_speaker_duration,
                'speaker_smoothing_enabled': server.speaker_smoothing_enabled
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(test_optimized_recognition()) 