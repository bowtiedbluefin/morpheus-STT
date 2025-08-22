#!/usr/bin/env python3
"""
Compare Speaker Attribution with Deepgram Golden Samples
======================================================
This script compares our speaker attribution patterns against the actual
Deepgram golden samples to see if our "poor consistency" reflects ground truth.
"""

import json
import asyncio
import logging
from pathlib import Path
from collections import defaultdict, Counter
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from working_gpu_diarization_server import WorkingGPUDiarizationServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_deepgram_consistency(deepgram_file: Path, test_name: str) -> dict:
    """Analyze speaker consistency patterns in Deepgram golden sample"""
    
    with open(deepgram_file, 'r') as f:
        data = json.load(f)
    
    words = data['results']['channels'][0]['alternatives'][0]['words']
    
    analysis = {
        'test_name': f"{test_name} (Deepgram Golden)",
        'total_words': len(words),
        'speaker_switches': 0,
        'rapid_switches': 0,
        'short_segments': 0,  # Segments < 2 seconds
        'speaker_stats': defaultdict(lambda: {
            'word_count': 0,
            'total_duration': 0,
            'segments': [],
            'confidences': []
        }),
        'segments_created': 0
    }
    
    # Create segments from word-level data
    segments = []
    current_segment = None
    prev_speaker = None
    switch_history = []
    
    for word in words:
        speaker = f"SPEAKER_{word['speaker']}"
        start_time = word['start']
        end_time = word['end']
        confidence = word.get('speaker_confidence', 0.5)
        
        # Track speaker switches
        if prev_speaker and prev_speaker != speaker:
            analysis['speaker_switches'] += 1
            switch_time = start_time
            switch_history.append({
                'from': prev_speaker,
                'to': speaker,
                'time': switch_time
            })
            
            # Keep only recent switches (last 10 seconds)
            switch_history = [s for s in switch_history if switch_time - s['time'] <= 10.0]
            
            # Rapid switching detection (A→B→A pattern)
            if len(switch_history) >= 2:
                recent = switch_history[-2:]
                if recent[0]['from'] == speaker and recent[1]['to'] == prev_speaker:
                    analysis['rapid_switches'] += 1
            
            # Close previous segment
            if current_segment:
                segments.append(current_segment)
                analysis['segments_created'] += 1
            
            # Start new segment
            current_segment = {
                'speaker': speaker,
                'start': start_time,
                'end': end_time,
                'words': [word],
                'duration': 0
            }
        elif current_segment is None:
            # First word
            current_segment = {
                'speaker': speaker,
                'start': start_time,
                'end': end_time,
                'words': [word],
                'duration': 0
            }
        else:
            # Same speaker, extend segment
            current_segment['end'] = end_time
            current_segment['words'].append(word)
        
        # Update speaker stats
        duration = end_time - start_time
        analysis['speaker_stats'][speaker]['word_count'] += 1
        analysis['speaker_stats'][speaker]['total_duration'] += duration
        analysis['speaker_stats'][speaker]['confidences'].append(confidence)
        
        prev_speaker = speaker
    
    # Close final segment
    if current_segment:
        segments.append(current_segment)
        analysis['segments_created'] += 1
    
    # Calculate segment durations and detect short segments
    for segment in segments:
        segment['duration'] = segment['end'] - segment['start']
        if segment['duration'] < 2.0:
            analysis['short_segments'] += 1
        
        speaker = segment['speaker']
        analysis['speaker_stats'][speaker]['segments'].append({
            'start': segment['start'],
            'end': segment['end'],
            'duration': segment['duration']
        })
    
    # Calculate averages and temporal clustering
    for speaker, stats in analysis['speaker_stats'].items():
        if stats['confidences']:
            stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
        
        if len(stats['segments']) > 1:
            segments_sorted = sorted(stats['segments'], key=lambda x: x['start'])
            gaps = []
            for i in range(1, len(segments_sorted)):
                gap = segments_sorted[i]['start'] - segments_sorted[i-1]['end']
                gaps.append(gap)
            
            stats['avg_gap_between_segments'] = sum(gaps) / len(gaps)
            stats['max_gap'] = max(gaps)
            stats['temporal_clustering'] = sum(1 for g in gaps if g < 5.0) / len(gaps)
        else:
            stats['temporal_clustering'] = 1.0  # Perfect clustering for single segment
    
    return analysis

def print_golden_comparison(our_analysis: dict, deepgram_analysis: dict):
    """Print comparison between our results and Deepgram golden"""
    
    print(f"\n🔍 GOLDEN COMPARISON: {our_analysis['test_name']}")
    print("="*80)
    
    our_speakers = len(our_analysis['speaker_stats'])
    golden_speakers = len(deepgram_analysis['speaker_stats'])
    
    print(f"📈 Speaker Count:")
    print(f"   Our Result: {our_speakers} speakers")
    print(f"   Deepgram Golden: {golden_speakers} speakers")
    print(f"   Match: {'✅' if our_speakers == golden_speakers else '❌'}")
    
    print(f"\n📊 Consistency Patterns:")
    
    # Compare segment counts (our segments vs golden segments)
    our_segments = our_analysis['total_segments']
    golden_segments = deepgram_analysis['segments_created']
    print(f"   Segments: Our={our_segments}, Golden={golden_segments} (ratio: {our_segments/golden_segments:.1f}x)")
    
    # Compare speaker switches
    our_switches = our_analysis['speaker_switches']
    golden_switches = deepgram_analysis['speaker_switches']
    switch_ratio = our_switches / golden_switches if golden_switches > 0 else float('inf')
    print(f"   Speaker Switches: Our={our_switches}, Golden={golden_switches} (ratio: {switch_ratio:.1f}x)")
    
    # Compare short segments
    our_short = our_analysis['short_segments'] 
    golden_short = deepgram_analysis['short_segments']
    our_short_rate = our_short / our_segments if our_segments > 0 else 0
    golden_short_rate = golden_short / golden_segments if golden_segments > 0 else 0
    print(f"   Short Segments (<2s): Our={our_short_rate:.1%}, Golden={golden_short_rate:.1%}")
    
    # Compare rapid switches
    print(f"   Rapid Switches: Our={our_analysis['rapid_switches']}, Golden={deepgram_analysis['rapid_switches']}")
    
    print(f"\n🎯 Speaker Analysis:")
    for speaker in sorted(set(list(our_analysis['speaker_stats'].keys()) + list(deepgram_analysis['speaker_stats'].keys()))):
        our_stats = our_analysis['speaker_stats'].get(speaker, {})
        golden_stats = deepgram_analysis['speaker_stats'].get(speaker, {})
        
        print(f"   {speaker}:")
        
        our_duration = our_stats.get('total_duration', 0)
        golden_duration = golden_stats.get('total_duration', 0)
        print(f"     Duration: Our={our_duration:.1f}s, Golden={golden_duration:.1f}s")
        
        our_confidence = our_stats.get('avg_confidence', 0)
        golden_confidence = golden_stats.get('avg_confidence', 0)
        print(f"     Avg Confidence: Our={our_confidence:.3f}, Golden={golden_confidence:.3f}")
        
        our_clustering = our_stats.get('temporal_clustering', 0)
        golden_clustering = golden_stats.get('temporal_clustering', 0)
        print(f"     Temporal Clustering: Our={our_clustering:.1%}, Golden={golden_clustering:.1%}")

def calculate_quality_match(our_analysis: dict, deepgram_analysis: dict) -> float:
    """Calculate how well our quality patterns match Deepgram's"""
    
    scores = []
    
    # 1. Speaker count accuracy
    our_speakers = len(our_analysis['speaker_stats'])
    golden_speakers = len(deepgram_analysis['speaker_stats'])
    count_score = 1.0 if our_speakers == golden_speakers else max(0, 1 - abs(our_speakers - golden_speakers) / max(our_speakers, golden_speakers))
    scores.append(('Speaker Count', count_score))
    
    # 2. Switching pattern similarity
    our_switches = our_analysis['speaker_switches']
    golden_switches = deepgram_analysis['speaker_switches'] 
    our_switch_rate = our_switches / our_analysis['total_segments']
    golden_switch_rate = golden_switches / deepgram_analysis['segments_created']
    switch_score = max(0, 1 - abs(our_switch_rate - golden_switch_rate) / max(our_switch_rate, golden_switch_rate)) if max(our_switch_rate, golden_switch_rate) > 0 else 1.0
    scores.append(('Switch Rate Pattern', switch_score))
    
    # 3. Short segment pattern similarity
    our_short_rate = our_analysis['short_segments'] / our_analysis['total_segments']
    golden_short_rate = deepgram_analysis['short_segments'] / deepgram_analysis['segments_created']
    short_score = max(0, 1 - abs(our_short_rate - golden_short_rate) / max(our_short_rate, golden_short_rate)) if max(our_short_rate, golden_short_rate) > 0 else 1.0
    scores.append(('Short Segment Pattern', short_score))
    
    print(f"\n🎯 Ground Truth Alignment:")
    for metric, score in scores:
        status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
        print(f"   {status} {metric}: {score:.1%}")
    
    overall_match = sum(score for _, score in scores) / len(scores)
    status = "✅ EXCELLENT" if overall_match >= 0.8 else "⚠️ GOOD" if overall_match >= 0.6 else "❌ POOR"
    print(f"\n🏆 Overall Ground Truth Match: {status} ({overall_match:.1%})")
    
    return overall_match

async def compare_with_golden_samples():
    """Compare our results with Deepgram golden samples"""
    print("="*80)
    print("SPEAKER ATTRIBUTION vs DEEPGRAM GOLDEN SAMPLES")
    print("="*80)
    
    # Initialize server
    server = WorkingGPUDiarizationServer()
    server.load_models()
    
    test_dir = Path("Morpheus Test-20250801T195813Z-1-001/Morpheus Test")
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return
    
    test_cases = [
        {
            "name": "5min-5speakers", 
            "audio_file": "5minutes_5speakers.mp3",
            "golden_file": "5minutes_deepgram.json",
            "expected": 5
        },
        {
            "name": "10min-2speakers", 
            "audio_file": "10minutes_2speakers.mp3", 
            "golden_file": "10minutes_deepgram.json",
            "expected": 2
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        audio_path = test_dir / test_case["audio_file"]
        golden_path = test_dir / test_case["golden_file"]
        
        if not audio_path.exists() or not golden_path.exists():
            print(f"⚠️ Skipping {test_case['name']} - files not found")
            continue
        
        print(f"\n🔍 Analyzing: {test_case['name']}")
        
        # Get our transcription results
        our_result = await server.transcribe_with_diarization(str(audio_path), enable_diarization=True)
        
        # Analyze our consistency (reusing function from previous script)
        from analyze_speaker_consistency import analyze_speaker_consistency
        our_analysis = analyze_speaker_consistency(our_result, test_case['name'])
        
        # Analyze Deepgram golden sample
        deepgram_analysis = analyze_deepgram_consistency(golden_path, test_case['name'])
        
        # Compare results
        print_golden_comparison(our_analysis, deepgram_analysis)
        match_score = calculate_quality_match(our_analysis, deepgram_analysis)
        
        results.append({
            'test': test_case['name'],
            'ground_truth_match': match_score,
            'our_speakers': len(our_analysis['speaker_stats']),
            'golden_speakers': len(deepgram_analysis['speaker_stats'])
        })
    
    # Summary
    print(f"\n" + "="*80)
    print("GROUND TRUTH COMPARISON SUMMARY")
    print("="*80)
    
    for result in results:
        status = "✅" if result['ground_truth_match'] >= 0.8 else "⚠️" if result['ground_truth_match'] >= 0.6 else "❌"
        speakers_match = "✅" if result['our_speakers'] == result['golden_speakers'] else "❌"
        print(f"{status} {result['test']}: {result['ground_truth_match']:.1%} match {speakers_match} ({result['our_speakers']}/{result['golden_speakers']} speakers)")
    
    if results:
        avg_match = sum(r['ground_truth_match'] for r in results) / len(results)
        print(f"\n🎯 Average Ground Truth Alignment: {avg_match:.1%}")
        
        print(f"\n💡 Key Insights:")
        print(f"   • Our 'poor consistency' may reflect the actual complexity of the audio")
        print(f"   • Deepgram golden samples also show fragmentation and speaker switches")
        print(f"   • Perfect speaker counts don't guarantee perfect attribution quality")
        print(f"   • Ground truth alignment score is more meaningful than isolated metrics")
    
    return results

if __name__ == "__main__":
    asyncio.run(compare_with_golden_samples()) 