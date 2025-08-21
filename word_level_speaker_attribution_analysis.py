#!/usr/bin/env python3
"""
Word-Level Speaker Attribution Analysis
======================================

This script performs detailed word-level speaker attribution comparison between:
1. Golden Reference (Deepgram)
2. WhisperX Configuration 1 (Original: 0.55/0.40/3.0s)
3. WhisperX Configuration 2 (Optimized: 0.5/0.3/1.0s)

Features:
- Intelligent speaker mapping between different systems
- Word-level alignment and attribution comparison
- Detailed accuracy metrics and mismatch analysis
- Speaker consistency analysis
- Visual reporting of attribution differences
"""

import os
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict, Counter
import difflib

# Configuration
SERVER_URL = "http://localhost:3337"
TEST_FILES_DIR = Path("/home/kyle/Desktop/whisper/Morpheus Test-20250801T195813Z-1-001/Morpheus Test")
ENV_FILE = Path("/home/kyle/Desktop/whisper/working_gpu.env")

# Test configurations
CONFIGS = {
    "config_1_original": {
        "PYANNOTE_CLUSTERING_THRESHOLD": "0.55",
        "PYANNOTE_SEGMENTATION_THRESHOLD": "0.40", 
        "MIN_SPEAKER_DURATION": "3.0",
        "description": "Original Configuration"
    },
    "config_2_optimized": {
        "PYANNOTE_CLUSTERING_THRESHOLD": "0.5",
        "PYANNOTE_SEGMENTATION_THRESHOLD": "0.3",
        "MIN_SPEAKER_DURATION": "1.0", 
        "description": "Optimized Configuration"
    }
}

# Test files mapping
TEST_FILES = {
    "5min": {
        "audio": "5minutes_5speakers.mp3",
        "reference": "5minutes_deepgram.json",
        "expected_speakers": 5,
        "duration_min": 5
    },
    "30min": {
        "audio": "30minutes_7speakers.mp3", 
        "reference": "30minutes_deepgram.json",
        "expected_speakers": 7,
        "duration_min": 30
    }
}

def log(message: str, level: str = "INFO"):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load and parse JSON data"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        log(f"Error loading JSON from {file_path}: {e}", "ERROR")
        return {}

def extract_word_segments_deepgram(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract word segments from Deepgram format"""
    words = []
    if 'results' in data and 'channels' in data['results']:
        channels = data['results']['channels']
        if channels and 'alternatives' in channels[0]:
            alternatives = channels[0]['alternatives']
            if alternatives and 'words' in alternatives[0]:
                for word in alternatives[0]['words']:
                    words.append({
                        'word': word.get('word', '').lower().strip('.,!?'),
                        'start': float(word.get('start', 0)),
                        'end': float(word.get('end', 0)),
                        'speaker': word.get('speaker', 'unknown'),
                        'confidence': word.get('confidence', 0.0)
                    })
    return words

def extract_word_segments_whisperx(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract word segments from WhisperX format"""
    words = []
    if 'result' in data and 'word_segments' in data['result']:
        for word in data['result']['word_segments']:
            words.append({
                'word': str(word.get('word', '')).lower().strip('.,!?'),
                'start': float(word.get('start', 0)),
                'end': float(word.get('end', 0)), 
                'speaker': word.get('speaker', 'unknown'),
                'confidence': word.get('score', 0.0)
            })
    return words

def align_word_sequences(ref_words: List[Dict], test_words: List[Dict], 
                        tolerance: float = 0.5) -> List[Tuple[Optional[Dict], Optional[Dict]]]:
    """Align word sequences based on timing and content"""
    aligned = []
    ref_idx = 0
    test_idx = 0
    
    while ref_idx < len(ref_words) or test_idx < len(test_words):
        ref_word = ref_words[ref_idx] if ref_idx < len(ref_words) else None
        test_word = test_words[test_idx] if test_idx < len(test_words) else None
        
        if ref_word is None:
            aligned.append((None, test_word))
            test_idx += 1
        elif test_word is None:
            aligned.append((ref_word, None))
            ref_idx += 1
        else:
            # Check if words match by content
            ref_text = ref_word['word']
            test_text = test_word['word']
            
            # Check timing proximity
            time_diff = abs(ref_word['start'] - test_word['start'])
            
            if ref_text == test_text and time_diff <= tolerance:
                # Perfect match
                aligned.append((ref_word, test_word))
                ref_idx += 1
                test_idx += 1
            elif time_diff <= tolerance:
                # Similar timing, different words
                aligned.append((ref_word, test_word))
                ref_idx += 1
                test_idx += 1
            elif ref_word['start'] < test_word['start']:
                # Reference word is earlier
                aligned.append((ref_word, None))
                ref_idx += 1
            else:
                # Test word is earlier
                aligned.append((None, test_word))
                test_idx += 1
    
    return aligned

def map_speakers_intelligent(ref_words: List[Dict], test_words: List[Dict]) -> Dict[str, str]:
    """Intelligently map speakers between reference and test using overlap analysis"""
    
    # Align sequences first
    aligned = align_word_sequences(ref_words, test_words)
    
    # Count speaker co-occurrences
    speaker_overlap = defaultdict(lambda: defaultdict(int))
    total_overlap = defaultdict(int)
    
    for ref_word, test_word in aligned:
        if ref_word and test_word and ref_word['word'] == test_word['word']:
            ref_speaker = ref_word['speaker']
            test_speaker = test_word['speaker']
            speaker_overlap[test_speaker][ref_speaker] += 1
            total_overlap[test_speaker] += 1
    
    # Create mapping based on highest overlap
    speaker_mapping = {}
    used_ref_speakers = set()
    
    # Sort test speakers by total overlap (most active first)
    sorted_test_speakers = sorted(total_overlap.keys(), 
                                key=lambda x: total_overlap[x], reverse=True)
    
    for test_speaker in sorted_test_speakers:
        if not speaker_overlap[test_speaker]:
            continue
            
        # Find best matching reference speaker
        best_ref_speaker = None
        best_score = 0
        
        for ref_speaker, count in speaker_overlap[test_speaker].items():
            if ref_speaker not in used_ref_speakers:
                score = count / total_overlap[test_speaker]  # Percentage overlap
                if score > best_score:
                    best_score = score
                    best_ref_speaker = ref_speaker
        
        if best_ref_speaker and best_score > 0.1:  # At least 10% overlap
            speaker_mapping[test_speaker] = best_ref_speaker
            used_ref_speakers.add(best_ref_speaker)
    
    return speaker_mapping

def analyze_speaker_attribution(ref_words: List[Dict], test_words: List[Dict], 
                              test_name: str) -> Dict[str, Any]:
    """Analyze word-level speaker attribution accuracy"""
    
    # Align word sequences
    aligned = align_word_sequences(ref_words, test_words)
    
    # Get speaker mapping
    speaker_mapping = map_speakers_intelligent(ref_words, test_words)
    
    # Analysis metrics
    total_words = 0
    correct_attributions = 0
    word_matches = 0
    mismatched_words = []
    speaker_confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    # Analyze each aligned pair
    for ref_word, test_word in aligned:
        if ref_word and test_word and ref_word['word'] == test_word['word']:
            total_words += 1
            word_matches += 1
            
            ref_speaker = ref_word['speaker']
            test_speaker = test_word['speaker']
            mapped_test_speaker = speaker_mapping.get(test_speaker, test_speaker)
            
            # Check if attribution is correct
            if str(ref_speaker) == str(mapped_test_speaker):
                correct_attributions += 1
            else:
                mismatched_words.append({
                    'word': ref_word['word'],
                    'start': ref_word['start'],
                    'reference_speaker': ref_speaker,
                    'test_speaker': test_speaker,
                    'mapped_test_speaker': mapped_test_speaker,
                    'confidence': test_word.get('confidence', 0.0)
                })
            
            # Update confusion matrix
            speaker_confusion_matrix[str(ref_speaker)][str(mapped_test_speaker)] += 1
        elif ref_word and not test_word:
            # Missing word in test
            total_words += 1
            mismatched_words.append({
                'word': ref_word['word'],
                'start': ref_word['start'],
                'reference_speaker': ref_word['speaker'],
                'test_speaker': 'MISSING',
                'mapped_test_speaker': 'MISSING',
                'confidence': 0.0
            })
        elif test_word and not ref_word:
            # Extra word in test
            mismatched_words.append({
                'word': test_word['word'],
                'start': test_word['start'],
                'reference_speaker': 'EXTRA',
                'test_speaker': test_word['speaker'],
                'mapped_test_speaker': test_word['speaker'],
                'confidence': test_word.get('confidence', 0.0)
            })
    
    # Calculate metrics
    attribution_accuracy = correct_attributions / total_words if total_words > 0 else 0.0
    word_alignment_accuracy = word_matches / len(aligned) if aligned else 0.0
    
    # Speaker statistics
    ref_speakers = set(w['speaker'] for w in ref_words)
    test_speakers = set(w['speaker'] for w in test_words)
    mapped_speakers = set(speaker_mapping.values())
    
    return {
        'test_name': test_name,
        'total_aligned_words': total_words,
        'word_matches': word_matches,
        'correct_attributions': correct_attributions,
        'attribution_accuracy': attribution_accuracy,
        'word_alignment_accuracy': word_alignment_accuracy,
        'speaker_mapping': speaker_mapping,
        'reference_speakers': sorted(list(ref_speakers)),
        'test_speakers': sorted(list(test_speakers)),
        'mapped_speakers': sorted(list(mapped_speakers)),
        'confusion_matrix': dict(speaker_confusion_matrix),
        'mismatched_words': mismatched_words[:20],  # First 20 mismatches
        'total_mismatches': len(mismatched_words)
    }

def update_env_config(config: Dict[str, str]) -> bool:
    """Update the working_gpu.env file with new configuration"""
    try:
        # Read current env file
        with open(ENV_FILE, 'r') as f:
            lines = f.readlines()
        
        # Update configuration values
        updated_lines = []
        for line in lines:
            updated = False
            for key, value in config.items():
                if line.startswith(f"{key}="):
                    updated_lines.append(f"{key}={value}\n")
                    updated = True
                    break
            if not updated:
                updated_lines.append(line)
        
        # Write back to file
        with open(ENV_FILE, 'w') as f:
            f.writelines(updated_lines)
        
        return True
    except Exception as e:
        log(f"Error updating env config: {e}", "ERROR")
        return False

def wait_for_server(timeout: int = 60) -> bool:
    """Wait for server to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{SERVER_URL}/health", timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False

def transcribe_file_with_config(file_path: Path, config_name: str) -> Optional[Dict[str, Any]]:
    """Transcribe file with specific configuration"""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'enable_diarization': 'true', 'response_format': 'json'}
            
            response = requests.post(
                f"{SERVER_URL}/v1/audio/transcriptions",
                files=files,
                data=data,
                timeout=300  # 5 minute timeout
            )
        
        if response.status_code == 200:
            result = response.json()
            log(f"Successfully transcribed {file_path.name} with {config_name}")
            return result
        else:
            log(f"Transcription failed for {config_name}: {response.status_code} - {response.text}", "ERROR")
            return None
            
    except Exception as e:
        log(f"Error during transcription with {config_name}: {e}", "ERROR")
        return None

def generate_detailed_report(results: Dict[str, Dict[str, Any]]) -> str:
    """Generate detailed word-level speaker attribution analysis report"""
    report = []
    report.append("=" * 100)
    report.append("DETAILED WORD-LEVEL SPEAKER ATTRIBUTION ANALYSIS")
    report.append("=" * 100)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary comparison
    report.append("ATTRIBUTION ACCURACY SUMMARY")
    report.append("-" * 50)
    
    for test_name, test_data in TEST_FILES.items():
        report.append(f"\n{test_name.upper()} TEST ({test_data['audio']})")
        report.append(f"Expected speakers: {test_data['expected_speakers']}")
        report.append("")
        
        # Headers
        report.append(f"{'Metric':<30} {'Golden Ref':<15} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
        report.append("-" * 90)
        
        if test_name in results:
            ref_data = results[test_name].get('reference', {})
            orig_data = results[test_name].get('original', {})
            opt_data = results[test_name].get('optimized', {})
            
            # Reference speakers
            ref_speakers = len(ref_data.get('reference_speakers', []))
            orig_speakers = len(orig_data.get('test_speakers', []))
            opt_speakers = len(opt_data.get('test_speakers', []))
            report.append(f"{'Speakers Detected':<30} {ref_speakers:<15} {orig_speakers:<15} {opt_speakers:<15} {opt_speakers - orig_speakers:+d}")
            
            # Attribution accuracy
            orig_acc = orig_data.get('attribution_accuracy', 0) * 100
            opt_acc = opt_data.get('attribution_accuracy', 0) * 100
            acc_diff = opt_acc - orig_acc
            report.append(f"{'Attribution Accuracy (%)':<30} {'100.0 (ref)':<15} {orig_acc:<15.1f} {opt_acc:<15.1f} {acc_diff:+.1f}")
            
            # Word alignment
            orig_align = orig_data.get('word_alignment_accuracy', 0) * 100
            opt_align = opt_data.get('word_alignment_accuracy', 0) * 100
            align_diff = opt_align - orig_align
            report.append(f"{'Word Alignment (%)':<30} {'100.0 (ref)':<15} {orig_align:<15.1f} {opt_align:<15.1f} {align_diff:+.1f}")
            
            # Total words analyzed
            orig_words = orig_data.get('total_aligned_words', 0)
            opt_words = opt_data.get('total_aligned_words', 0)
            report.append(f"{'Words Analyzed':<30} {'-':<15} {orig_words:<15} {opt_words:<15} {opt_words - orig_words:+d}")
            
            # Mismatches
            orig_mismatches = orig_data.get('total_mismatches', 0)
            opt_mismatches = opt_data.get('total_mismatches', 0)
            report.append(f"{'Total Mismatches':<30} {'-':<15} {orig_mismatches:<15} {opt_mismatches:<15} {opt_mismatches - orig_mismatches:+d}")
    
    # Detailed analysis for each test
    for test_name, test_results in results.items():
        report.append(f"\n\n{'=' * 80}")
        report.append(f"DETAILED ANALYSIS: {test_name.upper()}")
        report.append("=" * 80)
        
        for config_type, analysis in test_results.items():
            if config_type == 'reference':
                continue
                
            report.append(f"\n{config_type.upper()} CONFIGURATION ANALYSIS")
            report.append("-" * 40)
            
            # Speaker mapping
            speaker_mapping = analysis.get('speaker_mapping', {})
            if speaker_mapping:
                report.append("Speaker Mapping (Test -> Reference):")
                for test_spk, ref_spk in speaker_mapping.items():
                    report.append(f"  {test_spk} -> {ref_spk}")
            
            # Confusion matrix
            confusion = analysis.get('confusion_matrix', {})
            if confusion:
                report.append("\nSpeaker Confusion Matrix (Reference x Predicted):")
                all_speakers = set()
                for ref_spk in confusion:
                    all_speakers.add(ref_spk)
                    for pred_spk in confusion[ref_spk]:
                        all_speakers.add(pred_spk)
                
                sorted_speakers = sorted(all_speakers)
                
                # Header
                header = "Reference\\Predicted"
                for spk in sorted_speakers:
                    header += f"  {str(spk)[:8]:>8}"
                report.append(header)
                
                # Matrix rows
                for ref_spk in sorted_speakers:
                    row = f"{str(ref_spk)[:15]:<15}"
                    for pred_spk in sorted_speakers:
                        count = confusion.get(str(ref_spk), {}).get(str(pred_spk), 0)
                        row += f"  {count:>8}"
                    report.append(row)
            
            # Sample mismatches
            mismatches = analysis.get('mismatched_words', [])
            if mismatches:
                report.append(f"\nSample Mismatched Words (showing first {min(10, len(mismatches))}):")
                report.append(f"{'Word':<15} {'Time':<8} {'Reference':<12} {'Predicted':<12} {'Confidence':<10}")
                report.append("-" * 65)
                for mismatch in mismatches[:10]:
                    word = mismatch['word'][:14]
                    time_str = f"{mismatch['start']:.1f}s"
                    ref_spk = str(mismatch['reference_speaker'])[:11]
                    pred_spk = str(mismatch['mapped_test_speaker'])[:11]
                    conf = f"{mismatch['confidence']:.2f}"
                    report.append(f"{word:<15} {time_str:<8} {ref_spk:<12} {pred_spk:<12} {conf:<10}")
    
    # Overall conclusions
    report.append(f"\n\n{'=' * 80}")
    report.append("CONCLUSIONS AND RECOMMENDATIONS")
    report.append("=" * 80)
    
    # Calculate overall improvements
    total_orig_acc = 0
    total_opt_acc = 0
    total_tests = 0
    
    for test_name, test_results in results.items():
        orig_data = test_results.get('original', {})
        opt_data = test_results.get('optimized', {})
        
        if orig_data and opt_data:
            total_orig_acc += orig_data.get('attribution_accuracy', 0)
            total_opt_acc += opt_data.get('attribution_accuracy', 0)
            total_tests += 1
    
    if total_tests > 0:
        avg_orig_acc = (total_orig_acc / total_tests) * 100
        avg_opt_acc = (total_opt_acc / total_tests) * 100
        overall_improvement = avg_opt_acc - avg_orig_acc
        
        report.append(f"Overall Average Attribution Accuracy:")
        report.append(f"  Original Configuration: {avg_orig_acc:.1f}%")
        report.append(f"  Optimized Configuration: {avg_opt_acc:.1f}%")
        report.append(f"  Improvement: {overall_improvement:+.1f}%")
        report.append("")
        
        if overall_improvement > 0:
            report.append("✅ RECOMMENDATION: The optimized configuration shows improved speaker attribution accuracy.")
        elif abs(overall_improvement) < 1:
            report.append("⚖️  RECOMMENDATION: Both configurations show similar speaker attribution accuracy.")
        else:
            report.append("⚠️  RECOMMENDATION: The original configuration shows better speaker attribution accuracy.")
    
    return "\n".join(report)

def main():
    """Main execution function"""
    log("Starting Detailed Word-Level Speaker Attribution Analysis")
    
    # Verify test files exist
    for test_name, test_info in TEST_FILES.items():
        audio_file = TEST_FILES_DIR / test_info["audio"]
        reference_file = TEST_FILES_DIR / test_info["reference"]
        
        if not audio_file.exists():
            log(f"ERROR: Audio file not found: {audio_file}", "ERROR")
            return
        
        if not reference_file.exists():
            log(f"ERROR: Reference file not found: {reference_file}", "ERROR")
            return
        
        log(f"✓ Found test files for {test_name}")
    
    # Check server status
    if not wait_for_server():
        log("ERROR: Server is not responding", "ERROR")
        return
    
    all_results = {}
    
    # Process each test file
    for test_name, test_info in TEST_FILES.items():
        log(f"Processing {test_name} - {test_info['audio']}")
        
        audio_file = TEST_FILES_DIR / test_info["audio"]
        reference_file = TEST_FILES_DIR / test_info["reference"]
        
        # Load golden reference
        log(f"Loading golden reference: {test_info['reference']}")
        reference_data = load_json_file(reference_file)
        ref_words = extract_word_segments_deepgram(reference_data)
        log(f"Reference: {len(ref_words)} words, {len(set(w['speaker'] for w in ref_words))} speakers")
        
        test_results = {
            'reference': {
                'total_words': len(ref_words),
                'reference_speakers': sorted(list(set(w['speaker'] for w in ref_words)))
            }
        }
        
        # Test both configurations
        for config_name, config in CONFIGS.items():
            log(f"Testing {config['description']} ({config_name})")
            
            # Update configuration
            if not update_env_config(config):
                log(f"Failed to update config for {config_name}", "ERROR")
                continue
            
            # Wait a moment for config to take effect
            time.sleep(2)
            
            # Transcribe file
            result = transcribe_file_with_config(audio_file, config['description'])
            
            if result:
                # Extract words
                test_words = extract_word_segments_whisperx(result)
                log(f"{config['description']}: {len(test_words)} words, {len(set(w['speaker'] for w in test_words))} speakers")
                
                # Analyze attribution
                analysis = analyze_speaker_attribution(ref_words, test_words, 
                                                     f"{test_name}_{config_name}")
                
                config_type = "original" if "original" in config_name else "optimized"
                test_results[config_type] = analysis
                
                log(f"Attribution accuracy: {analysis['attribution_accuracy']:.3f}")
            else:
                log(f"Failed to transcribe with {config['description']}", "ERROR")
        
        all_results[test_name] = test_results
    
    # Generate comprehensive report
    report = generate_detailed_report(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as JSON
    results_file = Path(f"/home/kyle/Desktop/whisper/word_attribution_analysis_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save report as text
    report_file = Path(f"/home/kyle/Desktop/whisper/word_attribution_analysis_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    
    log(f"Detailed results saved to: {results_file}")
    log(f"Report saved to: {report_file}")
    
    print("\n" + "="*100)
    print("WORD-LEVEL SPEAKER ATTRIBUTION ANALYSIS COMPLETE")
    print("="*100)
    print(report)

if __name__ == "__main__":
    main() 