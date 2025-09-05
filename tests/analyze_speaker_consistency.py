#!/usr/bin/env python3
"""
Analyze Speaker Attribution Consistency
======================================
Checks if speaker IDs are consistent throughout transcription
- Speaker switching patterns
- Timeline consistency 
- Short segment analysis (potential mis-attribution)
- Speaker "ping-ponging" detection
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from working_gpu_diarization_server import WorkingGPUDiarizationServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_speaker_consistency(result: dict, test_name: str) -> dict:
    """Analyze speaker consistency patterns"""
    
    segments = result.get('result', {}).get('segments', [])
    if not segments:
        return {"error": "No segments found"}
    
    analysis = {
        'test_name': test_name,
        'total_segments': len(segments),
        'speaker_timeline': [],
        'speaker_switches': 0,
        'rapid_switches': 0,  # A→B→A within 10 seconds
        'short_segments': 0,  # Segments < 2 seconds
        'speaker_stats': defaultdict(lambda: {
            'segment_count': 0,
            'total_duration': 0,
            'avg_segment_duration': 0,
            'time_ranges': [],
            'confidence_scores': []
        })
    }
    
    prev_speaker = None
    switch_history = []  # Track recent switches for ping-pong detection
    
    for i, segment in enumerate(segments):
        speaker = segment.get('speaker', 'UNKNOWN')
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        duration = end_time - start_time
        confidence = segment.get('speaker_confidence', 0.0)
        
        # Timeline tracking
        analysis['speaker_timeline'].append({
            'segment_id': i,
            'speaker': speaker,
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'confidence': confidence
        })
        
        # Speaker statistics
        stats = analysis['speaker_stats'][speaker]
        stats['segment_count'] += 1
        stats['total_duration'] += duration
        stats['time_ranges'].append((start_time, end_time))
        stats['confidence_scores'].append(confidence)
        
        # Short segment detection
        if duration < 2.0:
            analysis['short_segments'] += 1
        
        # Speaker switch detection
        if prev_speaker and prev_speaker != speaker:
            analysis['speaker_switches'] += 1
            switch_history.append({
                'from': prev_speaker,
                'to': speaker,
                'time': start_time,
                'segment_id': i
            })
            
            # Keep only recent switches (last 10 seconds)
            switch_history = [s for s in switch_history if start_time - s['time'] <= 10.0]
            
            # Rapid switching detection (A→B→A pattern)
            if len(switch_history) >= 2:
                recent = switch_history[-2:]
                if recent[0]['from'] == speaker and recent[1]['to'] == prev_speaker:
                    analysis['rapid_switches'] += 1
        
        prev_speaker = speaker
    
    # Calculate averages
    for speaker, stats in analysis['speaker_stats'].items():
        if stats['segment_count'] > 0:
            stats['avg_segment_duration'] = stats['total_duration'] / stats['segment_count']
            stats['avg_confidence'] = sum(stats['confidence_scores']) / len(stats['confidence_scores'])
            # Temporal consistency - check if speaker appears in clustered time periods
            ranges = sorted(stats['time_ranges'])
            stats['time_span'] = (ranges[0][0], ranges[-1][1]) if ranges else (0, 0)
            
            # Calculate temporal clustering (are segments grouped together?)
            if len(ranges) > 1:
                gaps = []
                for j in range(1, len(ranges)):
                    gap = ranges[j][0] - ranges[j-1][1]  # Gap between segments
                    gaps.append(gap)
                stats['avg_gap_between_segments'] = sum(gaps) / len(gaps)
                stats['max_gap'] = max(gaps)
                stats['temporal_clustering'] = sum(1 for g in gaps if g < 5.0) / len(gaps)  # % of gaps < 5s
    
    return analysis

def print_consistency_report(analysis: dict):
    """Print detailed consistency analysis"""
    
    print(f"\n📊 SPEAKER CONSISTENCY ANALYSIS: {analysis['test_name']}")
    print("="*80)
    
    print(f"📈 Overview:")
    print(f"   Total segments: {analysis['total_segments']}")
    print(f"   Speaker switches: {analysis['speaker_switches']}")
    print(f"   Rapid switches (A→B→A): {analysis['rapid_switches']}")
    print(f"   Short segments (<2s): {analysis['short_segments']}")
    print(f"   Speakers detected: {len(analysis['speaker_stats'])}")
    
    print(f"\n🎯 Speaker Analysis:")
    for speaker, stats in sorted(analysis['speaker_stats'].items()):
        print(f"   {speaker}:")
        print(f"     Segments: {stats['segment_count']}")
        print(f"     Total duration: {stats['total_duration']:.1f}s")
        print(f"     Avg segment duration: {stats['avg_segment_duration']:.1f}s")
        print(f"     Avg confidence: {stats['avg_confidence']:.3f}")
        print(f"     Time span: {stats['time_span'][0]:.1f}s → {stats['time_span'][1]:.1f}s")
        if 'temporal_clustering' in stats:
            print(f"     Temporal clustering: {stats['temporal_clustering']:.1%} (higher = more consistent)")
            print(f"     Max gap between segments: {stats.get('max_gap', 0):.1f}s")
    
    # Consistency quality assessment
    print(f"\n⭐ Quality Assessment:")
    
    # Calculate quality score
    quality_scores = []
    
    # 1. Low speaker switching is good
    switch_rate = analysis['speaker_switches'] / analysis['total_segments']
    switch_score = max(0, 1.0 - switch_rate * 2)  # Penalize high switch rates
    quality_scores.append(('Switch Rate', switch_score, f"{switch_rate:.1%}"))
    
    # 2. Few rapid switches is good
    rapid_switch_score = max(0, 1.0 - analysis['rapid_switches'] / 10)  # Penalize rapid switches
    quality_scores.append(('Rapid Switches', rapid_switch_score, f"{analysis['rapid_switches']} switches"))
    
    # 3. Few short segments is good  
    short_rate = analysis['short_segments'] / analysis['total_segments']
    short_score = max(0, 1.0 - short_rate * 3)  # Penalize short segments
    quality_scores.append(('Short Segments', short_score, f"{short_rate:.1%}"))
    
    # 4. High temporal clustering is good
    avg_clustering = sum(stats.get('temporal_clustering', 0) for stats in analysis['speaker_stats'].values()) / len(analysis['speaker_stats'])
    quality_scores.append(('Temporal Clustering', avg_clustering, f"{avg_clustering:.1%}"))
    
    for metric, score, detail in quality_scores:
        status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
        print(f"   {status} {metric}: {score:.1%} ({detail})")
    
    overall_quality = sum(score for _, score, _ in quality_scores) / len(quality_scores)
    status = "✅ EXCELLENT" if overall_quality >= 0.8 else "⚠️ GOOD" if overall_quality >= 0.6 else "❌ POOR"
    print(f"\n🏆 Overall Consistency: {status} ({overall_quality:.1%})")
    
    return overall_quality

async def test_speaker_consistency():
    """Test speaker consistency for the optimized results"""
    print("="*80)
    print("SPEAKER ATTRIBUTION CONSISTENCY ANALYSIS")
    print("="*80)
    
    # Initialize server
    server = WorkingGPUDiarizationServer()
    server.load_models()
    
    test_dir = Path("Morpheus Test-20250801T195813Z-1-001/Morpheus Test")
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return
    
    # Test the files we got perfect counts for
    test_cases = [
        {"name": "5min-5speakers", "audio_file": "5minutes_5speakers.mp3", "expected": 5},
        {"name": "10min-2speakers", "audio_file": "10minutes_2speakers.mp3", "expected": 2}
    ]
    
    results = []
    
    for test_case in test_cases:
        audio_path = test_dir / test_case["audio_file"]
        if not audio_path.exists():
            continue
            
        print(f"\n🔍 Analyzing: {test_case['name']}")
        
        # Get transcription with diarization
        result = await server.transcribe_with_diarization(str(audio_path), enable_diarization=True)
        
        # Analyze consistency
        analysis = analyze_speaker_consistency(result, test_case['name'])
        quality_score = print_consistency_report(analysis)
        
        results.append({
            'test': test_case['name'],
            'quality_score': quality_score,
            'analysis': analysis
        })
    
    # Summary
    print(f"\n" + "="*80)
    print("CONSISTENCY SUMMARY")
    print("="*80)
    
    for result in results:
        status = "✅" if result['quality_score'] >= 0.8 else "⚠️" if result['quality_score'] >= 0.6 else "❌"
        print(f"{status} {result['test']}: {result['quality_score']:.1%} consistency")
    
    avg_quality = sum(r['quality_score'] for r in results) / len(results)
    print(f"\n🎯 Average Consistency: {avg_quality:.1%}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_speaker_consistency()) 