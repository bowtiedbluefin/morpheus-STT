#!/usr/bin/env python3
"""
Text Attribution Accuracy Comparison
====================================
Compares the actual TEXT content attributed to each speaker between
our results and Deepgram's golden samples to measure attribution accuracy.
"""

import json
import asyncio
import logging
from pathlib import Path
from collections import defaultdict
import difflib
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from working_gpu_diarization_server import WorkingGPUDiarizationServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_speaker_text(result_data, source="our"):
    """Extract text content per speaker from transcription results"""
    
    speaker_texts = defaultdict(list)
    
    if source == "our":
        # Our result format
        segments = result_data.get('result', {}).get('segments', [])
        for segment in segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            text = segment.get('text', '').strip()
            if text:
                speaker_texts[speaker].append({
                    'text': text,
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'confidence': segment.get('speaker_confidence', 0)
                })
    
    elif source == "deepgram":
        # Deepgram golden format - word level
        words = result_data['results']['channels'][0]['alternatives'][0]['words']
        
        # Group consecutive words by speaker
        current_speaker = None
        current_text = []
        current_start = None
        current_end = None
        
        for word in words:
            speaker = f"SPEAKER_{word['speaker']}"
            word_text = word['punctuated_word']
            
            if speaker != current_speaker:
                # Save previous group
                if current_speaker and current_text:
                    text = ' '.join(current_text).strip()
                    if text:
                        speaker_texts[current_speaker].append({
                            'text': text,
                            'start': current_start,
                            'end': current_end,
                            'confidence': 0  # Deepgram doesn't have segment confidence
                        })
                
                # Start new group
                current_speaker = speaker
                current_text = [word_text]
                current_start = word['start']
                current_end = word['end']
            else:
                # Continue current group
                current_text.append(word_text)
                current_end = word['end']
        
        # Save final group
        if current_speaker and current_text:
            text = ' '.join(current_text).strip()
            if text:
                speaker_texts[current_speaker].append({
                    'text': text,
                    'start': current_start,
                    'end': current_end,
                    'confidence': 0
                })
    
    # Combine all text per speaker
    speaker_full_text = {}
    for speaker, segments in speaker_texts.items():
        full_text = ' '.join(segment['text'] for segment in segments)
        speaker_full_text[speaker] = {
            'text': full_text.strip(),
            'segments': segments,
            'word_count': len(full_text.split()),
            'total_duration': sum(seg['end'] - seg['start'] for seg in segments)
        }
    
    return speaker_full_text

def calculate_text_similarity(text1, text2):
    """Calculate text similarity between two strings"""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    
    # Normalize text
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    # Calculate similarity using difflib
    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    return similarity

def find_best_speaker_mapping(our_speakers, deepgram_speakers):
    """Find the best mapping between our speakers and Deepgram speakers based on text content"""
    
    mappings = {}
    similarity_matrix = {}
    
    # Calculate similarity between all speaker pairs
    for our_speaker, our_data in our_speakers.items():
        similarity_matrix[our_speaker] = {}
        for dg_speaker, dg_data in deepgram_speakers.items():
            similarity = calculate_text_similarity(our_data['text'], dg_data['text'])
            similarity_matrix[our_speaker][dg_speaker] = similarity
    
    # Find best mapping using greedy approach
    used_dg_speakers = set()
    
    # Sort our speakers by total duration (prioritize main speakers)
    our_speakers_sorted = sorted(our_speakers.items(), key=lambda x: x[1]['total_duration'], reverse=True)
    
    for our_speaker, our_data in our_speakers_sorted:
        best_match = None
        best_similarity = 0
        
        for dg_speaker in deepgram_speakers:
            if dg_speaker in used_dg_speakers:
                continue
            
            similarity = similarity_matrix[our_speaker][dg_speaker]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = dg_speaker
        
        if best_match and best_similarity > 0.3:  # Minimum threshold
            mappings[our_speaker] = {
                'deepgram_speaker': best_match,
                'similarity': best_similarity,
                'our_text': our_data['text'][:200] + "..." if len(our_data['text']) > 200 else our_data['text'],
                'dg_text': deepgram_speakers[best_match]['text'][:200] + "..." if len(deepgram_speakers[best_match]['text']) > 200 else deepgram_speakers[best_match]['text'],
                'our_word_count': our_data['word_count'],
                'dg_word_count': deepgram_speakers[best_match]['word_count']
            }
            used_dg_speakers.add(best_match)
    
    return mappings, similarity_matrix

def print_attribution_comparison(test_name, our_speakers, deepgram_speakers, mappings):
    """Print detailed attribution comparison"""
    
    print(f"\n📝 TEXT ATTRIBUTION ANALYSIS: {test_name}")
    print("="*80)
    
    print(f"📊 Speaker Count:")
    print(f"   Our Result: {len(our_speakers)} speakers")
    print(f"   Deepgram Golden: {len(deepgram_speakers)} speakers")
    
    print(f"\n🎯 Speaker Mapping & Text Accuracy:")
    
    total_similarity = 0
    mapped_count = 0
    
    for our_speaker, mapping in mappings.items():
        dg_speaker = mapping['deepgram_speaker']
        similarity = mapping['similarity']
        
        status = "✅" if similarity >= 0.8 else "⚠️" if similarity >= 0.6 else "❌"
        print(f"\n   {status} {our_speaker} → {dg_speaker} ({similarity:.1%} text match)")
        print(f"      Our text ({mapping['our_word_count']} words): \"{mapping['our_text']}\"")
        print(f"      Golden text ({mapping['dg_word_count']} words): \"{mapping['dg_text']}\"")
        
        total_similarity += similarity
        mapped_count += 1
    
    # Check for unmapped speakers
    unmapped_our = set(our_speakers.keys()) - set(mappings.keys())
    unmapped_dg = set(deepgram_speakers.keys()) - set(m['deepgram_speaker'] for m in mappings.values())
    
    if unmapped_our:
        print(f"\n   ❌ Unmapped Our Speakers: {list(unmapped_our)}")
        for speaker in unmapped_our:
            word_count = our_speakers[speaker]['word_count']
            sample_text = our_speakers[speaker]['text'][:100] + "..." if len(our_speakers[speaker]['text']) > 100 else our_speakers[speaker]['text']
            print(f"      {speaker} ({word_count} words): \"{sample_text}\"")
    
    if unmapped_dg:
        print(f"\n   ❌ Unmapped Deepgram Speakers: {list(unmapped_dg)}")
        for speaker in unmapped_dg:
            word_count = deepgram_speakers[speaker]['word_count']
            sample_text = deepgram_speakers[speaker]['text'][:100] + "..." if len(deepgram_speakers[speaker]['text']) > 100 else deepgram_speakers[speaker]['text']
            print(f"      {speaker} ({word_count} words): \"{sample_text}\"")
    
    # Calculate overall accuracy
    if mapped_count > 0:
        avg_similarity = total_similarity / mapped_count
        mapping_coverage = mapped_count / max(len(our_speakers), len(deepgram_speakers))
        overall_accuracy = avg_similarity * mapping_coverage
        
        print(f"\n🏆 Attribution Accuracy:")
        print(f"   Average Text Similarity: {avg_similarity:.1%}")
        print(f"   Speaker Mapping Coverage: {mapping_coverage:.1%}")
        print(f"   Overall Accuracy: {overall_accuracy:.1%}")
        
        status = "✅ EXCELLENT" if overall_accuracy >= 0.8 else "⚠️ GOOD" if overall_accuracy >= 0.6 else "❌ POOR"
        print(f"   Status: {status}")
        
        return overall_accuracy
    
    return 0.0

async def compare_text_attribution():
    """Compare text attribution accuracy with Deepgram golden samples"""
    print("="*80)
    print("TEXT ATTRIBUTION ACCURACY COMPARISON")
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
        
        # Extract speaker texts
        our_speakers = extract_speaker_text(our_result, "our")
        deepgram_speakers = extract_speaker_text(deepgram_result, "deepgram")
        
        # Find best speaker mapping
        mappings, similarity_matrix = find_best_speaker_mapping(our_speakers, deepgram_speakers)
        
        # Print comparison
        accuracy = print_attribution_comparison(test_case['name'], our_speakers, deepgram_speakers, mappings)
        
        results.append({
            'test': test_case['name'],
            'accuracy': accuracy,
            'our_speaker_count': len(our_speakers),
            'golden_speaker_count': len(deepgram_speakers),
            'mapped_speakers': len(mappings)
        })
    
    # Summary
    print(f"\n" + "="*80)
    print("TEXT ATTRIBUTION ACCURACY SUMMARY")
    print("="*80)
    
    for result in results:
        status = "✅" if result['accuracy'] >= 0.8 else "⚠️" if result['accuracy'] >= 0.6 else "❌"
        speaker_match = "✅" if result['our_speaker_count'] == result['golden_speaker_count'] else "❌"
        print(f"{status} {result['test']}: {result['accuracy']:.1%} accuracy {speaker_match} ({result['our_speaker_count']}/{result['golden_speaker_count']} speakers, {result['mapped_speakers']} mapped)")
    
    if results:
        avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
        print(f"\n🎯 Average Text Attribution Accuracy: {avg_accuracy:.1%}")
        
        if avg_accuracy >= 0.8:
            print(f"\n🎉 EXCELLENT: We're achieving high text attribution accuracy!")
        elif avg_accuracy >= 0.6:
            print(f"\n⚠️ GOOD: Decent attribution accuracy, but room for improvement")
        else:
            print(f"\n❌ POOR: Significant misattribution issues need addressing")
        
        print(f"\n💡 This measures if we're saying the same speaker said the same words as Deepgram")
    
    return results

if __name__ == "__main__":
    asyncio.run(compare_text_attribution()) 