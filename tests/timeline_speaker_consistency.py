#!/usr/bin/env python3
"""
Timeline Speaker Consistency Analysis
====================================
Analyzes whether speaker mappings remain consistent throughout the entire
transcription timeline between our results and Deepgram's golden samples.

The goal: If we say SPEAKER_A at time T and Deepgram says SPEAKER_X at time T,
then SPEAKER_A should ALWAYS map to SPEAKER_X throughout the entire transcription.
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

def extract_timeline_segments(result_data, source="our", segment_duration=5.0):
    """Extract speaker assignments for fixed time segments"""
    
    timeline_segments = []
    
    if source == "our":
        # Our result format
        segments = result_data.get('result', {}).get('segments', [])
        for segment in segments:
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            speaker = segment.get('speaker', 'UNKNOWN')
            text = segment.get('text', '').strip()
            
            timeline_segments.append({
                'start': start,
                'end': end,
                'speaker': speaker,
                'text': text,
                'source': 'our'
            })
    
    elif source == "deepgram":
        # Deepgram golden format - word level, group into segments
        words = result_data['results']['channels'][0]['alternatives'][0]['words']
        
        current_speaker = None
        current_text = []
        current_start = None
        
        for word in words:
            speaker = f"SPEAKER_{word['speaker']}"
            word_text = word['punctuated_word']
            word_start = word['start']
            word_end = word['end']
            
            if speaker != current_speaker:
                # Save previous segment
                if current_speaker and current_text:
                    text = ' '.join(current_text).strip()
                    timeline_segments.append({
                        'start': current_start,
                        'end': prev_word_end,
                        'speaker': current_speaker,
                        'text': text,
                        'source': 'deepgram'
                    })
                
                # Start new segment
                current_speaker = speaker
                current_text = [word_text]
                current_start = word_start
            else:
                current_text.append(word_text)
            
            prev_word_end = word_end
        
        # Save final segment
        if current_speaker and current_text:
            text = ' '.join(current_text).strip()
            timeline_segments.append({
                'start': current_start,
                'end': prev_word_end,
                'speaker': current_speaker,
                'text': text,
                'source': 'deepgram'
            })
    
    return sorted(timeline_segments, key=lambda x: x['start'])

def create_time_grid_mapping(our_segments, deepgram_segments, grid_size=2.0):
    """Create a time grid and map speakers at each time point"""
    
    # Find total duration
    all_segments = our_segments + deepgram_segments
    max_time = max(seg['end'] for seg in all_segments)
    
    # Create time grid
    time_points = []
    current_time = 0
    while current_time < max_time:
        time_points.append(current_time)
        current_time += grid_size
    
    mappings = []
    
    for time_point in time_points:
        # Find active speakers at this time point
        our_speaker = None
        deepgram_speaker = None
        our_text = ""
        deepgram_text = ""
        
        # Find our speaker at this time
        for segment in our_segments:
            if segment['start'] <= time_point < segment['end']:
                our_speaker = segment['speaker']
                our_text = segment['text'][:50] + "..." if len(segment['text']) > 50 else segment['text']
                break
        
        # Find Deepgram speaker at this time
        for segment in deepgram_segments:
            if segment['start'] <= time_point < segment['end']:
                deepgram_speaker = segment['speaker']
                deepgram_text = segment['text'][:50] + "..." if len(segment['text']) > 50 else segment['text']
                break
        
        # Record mapping if both speakers are active
        if our_speaker and deepgram_speaker:
            mappings.append({
                'time': time_point,
                'our_speaker': our_speaker,
                'deepgram_speaker': deepgram_speaker,
                'our_text': our_text,
                'deepgram_text': deepgram_text
            })
    
    return mappings

def analyze_mapping_consistency(mappings):
    """Analyze how consistent the speaker mappings are throughout the timeline"""
    
    if not mappings:
        return {'consistency': 0, 'mappings': {}, 'conflicts': [], 'stats': {}}
    
    # Build mapping matrix
    mapping_votes = defaultdict(lambda: defaultdict(int))
    
    for mapping in mappings:
        our_spk = mapping['our_speaker']
        dg_spk = mapping['deepgram_speaker']
        mapping_votes[our_spk][dg_spk] += 1
    
    # Find the most consistent mapping for each speaker
    consistent_mapping = {}
    conflicts = []
    
    for our_speaker, dg_votes in mapping_votes.items():
        # Find the most voted Deepgram speaker
        best_dg_speaker = max(dg_votes.items(), key=lambda x: x[1])
        total_votes = sum(dg_votes.values())
        consistency_ratio = best_dg_speaker[1] / total_votes
        
        consistent_mapping[our_speaker] = {
            'maps_to': best_dg_speaker[0],
            'consistency': consistency_ratio,
            'total_occurrences': total_votes,
            'confident_occurrences': best_dg_speaker[1]
        }
        
        # Record conflicts (when speaker mapping is inconsistent)
        if consistency_ratio < 1.0:
            conflict_details = []
            for dg_speaker, votes in dg_votes.items():
                if dg_speaker != best_dg_speaker[0]:
                    conflict_details.append({
                        'alternative_mapping': dg_speaker,
                        'occurrences': votes,
                        'percentage': votes / total_votes
                    })
            
            conflicts.append({
                'our_speaker': our_speaker,
                'primary_mapping': best_dg_speaker[0],
                'primary_consistency': consistency_ratio,
                'conflicts': conflict_details
            })
    
    # Calculate overall consistency
    total_consistent = sum(mapping['confident_occurrences'] for mapping in consistent_mapping.values())
    total_mappings = len(mappings)
    overall_consistency = total_consistent / total_mappings if total_mappings > 0 else 0
    
    # Statistics
    stats = {
        'total_time_points': total_mappings,
        'consistent_mappings': total_consistent,
        'consistency_percentage': overall_consistency,
        'number_of_conflicts': len(conflicts),
        'our_speakers': len(consistent_mapping),
        'deepgram_speakers': len(set(m['maps_to'] for m in consistent_mapping.values()))
    }
    
    return {
        'consistency': overall_consistency,
        'mappings': consistent_mapping,
        'conflicts': conflicts,
        'stats': stats
    }

def print_timeline_consistency_analysis(test_name, analysis):
    """Print detailed timeline consistency analysis"""
    
    print(f"\n⏰ TIMELINE SPEAKER CONSISTENCY: {test_name}")
    print("="*80)
    
    stats = analysis['stats']
    consistency = analysis['consistency']
    
    print(f"📊 Overview:")
    print(f"   Time points analyzed: {stats['total_time_points']}")
    print(f"   Our speakers: {stats['our_speakers']}")
    print(f"   Deepgram speakers: {stats['deepgram_speakers']}")
    print(f"   Overall consistency: {consistency:.1%}")
    
    print(f"\n🗺️ Speaker Mapping:")
    for our_speaker, mapping in analysis['mappings'].items():
        consistency_pct = mapping['consistency']
        status = "✅" if consistency_pct >= 0.9 else "⚠️" if consistency_pct >= 0.7 else "❌"
        
        print(f"   {status} {our_speaker} → {mapping['maps_to']} "
              f"({consistency_pct:.1%} consistent, {mapping['total_occurrences']} occurrences)")
    
    # Show conflicts if any
    if analysis['conflicts']:
        print(f"\n⚠️ Consistency Conflicts:")
        for conflict in analysis['conflicts']:
            print(f"   {conflict['our_speaker']} (primary: {conflict['primary_mapping']}, {conflict['primary_consistency']:.1%} consistent):")
            for alt in conflict['conflicts']:
                print(f"      Also maps to {alt['alternative_mapping']} {alt['percentage']:.1%} of the time ({alt['occurrences']} times)")
    
    # Overall assessment
    print(f"\n🏆 Timeline Consistency Assessment:")
    if consistency >= 0.9:
        status = "✅ EXCELLENT"
        message = "Speakers remain very consistently mapped throughout!"
    elif consistency >= 0.7:
        status = "⚠️ GOOD" 
        message = "Mostly consistent with some speaker switching"
    elif consistency >= 0.5:
        status = "⚠️ FAIR"
        message = "Moderate consistency - significant speaker confusion"
    else:
        status = "❌ POOR"
        message = "Poor consistency - major speaker attribution issues"
    
    print(f"   Status: {status}")
    print(f"   Assessment: {message}")
    
    return consistency

async def analyze_timeline_consistency():
    """Analyze timeline speaker consistency with Deepgram golden samples"""
    print("="*80)
    print("TIMELINE SPEAKER CONSISTENCY ANALYSIS")
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
            "golden_file": "5minutes_deepgram.json"
        },
        {
            "name": "10min-2speakers", 
            "audio_file": "10minutes_2speakers.mp3",
            "golden_file": "10minutes_deepgram.json"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        audio_path = test_dir / test_case["audio_file"]
        golden_path = test_dir / test_case["golden_file"]
        
        if not audio_path.exists() or not golden_path.exists():
            print(f"⚠️ Skipping {test_case['name']} - files not found")
            continue
        
        print(f"\n🔍 Processing: {test_case['name']}")
        
        # Get our transcription results
        our_result = await server.transcribe_with_diarization(str(audio_path), enable_diarization=True)
        
        # Load Deepgram golden results
        with open(golden_path, 'r') as f:
            deepgram_result = json.load(f)
        
        # Extract timeline segments
        our_segments = extract_timeline_segments(our_result, "our")
        deepgram_segments = extract_timeline_segments(deepgram_result, "deepgram")
        
        print(f"   Our segments: {len(our_segments)}")
        print(f"   Deepgram segments: {len(deepgram_segments)}")
        
        # Create time grid mapping
        mappings = create_time_grid_mapping(our_segments, deepgram_segments, grid_size=2.0)
        
        print(f"   Time grid points with both speakers: {len(mappings)}")
        
        # Analyze consistency
        analysis = analyze_mapping_consistency(mappings)
        
        # Print analysis
        consistency = print_timeline_consistency_analysis(test_case['name'], analysis)
        
        results.append({
            'test': test_case['name'],
            'consistency': consistency,
            'our_speakers': analysis['stats']['our_speakers'],
            'deepgram_speakers': analysis['stats']['deepgram_speakers'],
            'conflicts': len(analysis['conflicts'])
        })
    
    # Summary
    print(f"\n" + "="*80)
    print("TIMELINE CONSISTENCY SUMMARY")
    print("="*80)
    
    for result in results:
        status = "✅" if result['consistency'] >= 0.9 else "⚠️" if result['consistency'] >= 0.7 else "❌"
        speaker_match = "✅" if result['our_speakers'] == result['deepgram_speakers'] else "❌"
        print(f"{status} {result['test']}: {result['consistency']:.1%} consistent {speaker_match} ({result['our_speakers']}/{result['deepgram_speakers']} speakers, {result['conflicts']} conflicts)")
    
    if results:
        avg_consistency = sum(r['consistency'] for r in results) / len(results)
        print(f"\n🎯 Average Timeline Consistency: {avg_consistency:.1%}")
        
        if avg_consistency >= 0.9:
            print(f"\n🎉 EXCELLENT: Speakers remain consistently mapped throughout the transcription!")
            print(f"   This means when we say SPEAKER_A, it consistently corresponds to the same")
            print(f"   person that Deepgram calls SPEAKER_X throughout the entire audio.")
        elif avg_consistency >= 0.7:
            print(f"\n⚠️ GOOD: Mostly consistent speaker mapping with some confusion")
        else:
            print(f"\n❌ POOR: Significant speaker mapping inconsistencies")
        
        print(f"\n💡 This measures whether our speaker assignments remain consistent")
        print(f"   with Deepgram's assignments throughout the timeline, regardless of speaker IDs")
    
    return results

if __name__ == "__main__":
    asyncio.run(analyze_timeline_consistency()) 