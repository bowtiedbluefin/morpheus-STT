#!/usr/bin/env python3
"""
Performance Sensitivity Analysis for WhisperX Diarization Server
================================================================

This script tests different parameter configurations for the working GPU diarization server:

Configuration 1 (Current):
- PYANNOTE_CLUSTERING_THRESHOLD=0.55
- PYANNOTE_SEGMENTATION_THRESHOLD=0.40
- MIN_SPEAKER_DURATION=3.0

Configuration 2 (Test):
- PYANNOTE_CLUSTERING_THRESHOLD=0.5
- PYANNOTE_SEGMENTATION_THRESHOLD=0.3
- MIN_SPEAKER_DURATION=1.0

Test files:
- 5minutes_5speakers.mp3 (5 expected speakers)
- 30minutes_7speakers.mp3 (7 expected speakers)

Golden references:
- 5minutes_deepgram.json
- 30minutes_deepgram.json
"""

import os
import json
import requests
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import difflib

# Configuration
SERVER_URL = "http://localhost:3337"
TEST_FILES_DIR = Path("/home/kyle/Desktop/whisper/Morpheus Test-20250801T195813Z-1-001/Morpheus Test")
ENV_FILE = Path("/home/kyle/Desktop/whisper/working_gpu.env")
SERVER_SCRIPT = Path("/home/kyle/Desktop/whisper/working_gpu_diarization_server.py")

# Test configurations
CONFIGS = {
    "config_1_current": {
        "PYANNOTE_CLUSTERING_THRESHOLD": "0.55",
        "PYANNOTE_SEGMENTATION_THRESHOLD": "0.40", 
        "MIN_SPEAKER_DURATION": "3.0",
        "description": "Current Configuration"
    },
    "config_2_test": {
        "PYANNOTE_CLUSTERING_THRESHOLD": "0.5",
        "PYANNOTE_SEGMENTATION_THRESHOLD": "0.3",
        "MIN_SPEAKER_DURATION": "1.0", 
        "description": "Test Configuration"
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

def load_reference_data(file_path: Path) -> Dict[str, Any]:
    """Load and parse reference JSON data"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        log(f"Error loading reference data from {file_path}: {e}", "ERROR")
        return {}

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
        
        log(f"Updated configuration: {config}")
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

def restart_server() -> bool:
    """Check if server is running, don't restart if working"""
    log("Checking server status...")
    
    # Check if server is already running and healthy
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            log("Server is already running and healthy")
            return True
    except:
        pass
    
    log("Server not responding, attempting restart with fixed GPU script...")
    
    # Kill existing server process
    try:
        subprocess.run(["pkill", "-f", "working_gpu_diarization_server.py"], check=False)
        time.sleep(3)
    except:
        pass
    
    # Start server using the fixed GPU script that handles cuDNN properly
    try:
        env = os.environ.copy()
        # Use environment variable for HuggingFace token - don't hardcode it!
        if 'HUGGINGFACE_TOKEN' not in env:
            print("WARNING: HUGGINGFACE_TOKEN not set in environment")
        
        # Use the fixed_gpu_start.sh script that handles cuDNN properly
        fixed_script = Path("/home/kyle/Desktop/whisper/fixed_gpu_start.sh")
        if fixed_script.exists():
            process = subprocess.Popen(
                [str(fixed_script)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=fixed_script.parent
            )
        else:
            # Fallback to direct python launch
            with open(ENV_FILE, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env[key] = value
            
            process = subprocess.Popen(
                ["python3", str(SERVER_SCRIPT)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=SERVER_SCRIPT.parent
            )
        
        # Wait for server to be ready  
        if wait_for_server(timeout=120):
            log("Server started successfully")
            return True
        else:
            log("Server failed to start within timeout", "ERROR")
            return False
    except Exception as e:
        log(f"Error starting server: {e}", "ERROR") 
        return False

def transcribe_file(file_path: Path) -> Tuple[Dict[str, Any], float]:
    """Transcribe audio file and return results with processing time"""
    start_time = time.time()
    
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
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            return response.json(), processing_time
        else:
            log(f"Transcription failed: {response.status_code} - {response.text}", "ERROR")
            return {}, processing_time
            
    except Exception as e:
        processing_time = time.time() - start_time
        log(f"Error during transcription: {e}", "ERROR")
        return {}, processing_time

def extract_speakers_from_result(result: Dict[str, Any]) -> List[str]:
    """Extract unique speakers from transcription result"""
    speakers = set()
    
    if 'result' in result and 'word_segments' in result['result']:
        for segment in result['result']['word_segments']:
            if 'speaker' in segment:
                speakers.add(segment['speaker'])
    
    return sorted(list(speakers))

def compare_word_level_attribution(result: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
    """Compare word-level speaker attribution between result and reference"""
    comparison = {
        "total_words_result": 0,
        "total_words_reference": 0,
        "matching_words": 0,
        "speaker_accuracy": 0.0,
        "mismatched_segments": []
    }
    
    # Extract word segments from result
    result_words = []
    if 'result' in result and 'word_segments' in result['result']:
        result_words = result['result']['word_segments']
    
    # Extract word segments from reference (Deepgram format)
    reference_words = []
    if 'results' in reference and 'channels' in reference['results']:
        channels = reference['results']['channels']
        if channels and 'alternatives' in channels[0]:
            alternatives = channels[0]['alternatives']
            if alternatives and 'words' in alternatives[0]:
                reference_words = alternatives[0]['words']
    
    comparison["total_words_result"] = len(result_words)
    comparison["total_words_reference"] = len(reference_words)
    
    # Simple word matching by content and approximate timing
    matches = 0
    for i, result_word in enumerate(result_words):
        if i < len(reference_words):
            ref_word = reference_words[i]
            
            # Check if words match (case insensitive, basic normalization)
            result_text = str(result_word.get('word', '')).lower().strip('.,!?')
            ref_text = str(ref_word.get('word', '')).lower().strip('.,!?')
            
            if result_text == ref_text:
                matches += 1
            else:
                comparison["mismatched_segments"].append({
                    "index": i,
                    "result_word": result_word.get('word', ''),
                    "reference_word": ref_word.get('word', ''),
                    "result_speaker": result_word.get('speaker', ''),
                    "reference_speaker": ref_word.get('speaker', 'N/A')
                })
    
    comparison["matching_words"] = matches
    if comparison["total_words_result"] > 0:
        comparison["speaker_accuracy"] = matches / comparison["total_words_result"]
    
    return comparison

def run_test_configuration(config_name: str, config: Dict[str, str]) -> Dict[str, Any]:
    """Run tests for a specific configuration"""
    log(f"Starting test for {config['description']} ({config_name})")
    
    # Update configuration
    if not update_env_config(config):
        return {"error": "Failed to update configuration"}
    
    # Restart server with new config
    if not restart_server():
        return {"error": "Failed to restart server"}
    
    test_results = {
        "config_name": config_name,
        "config": config,
        "tests": {}
    }
    
    # Run tests on each file
    for test_name, test_info in TEST_FILES.items():
        log(f"Testing {test_name} - {test_info['audio']}")
        
        audio_file = TEST_FILES_DIR / test_info["audio"]
        reference_file = TEST_FILES_DIR / test_info["reference"]
        
        if not audio_file.exists():
            log(f"Audio file not found: {audio_file}", "ERROR")
            continue
        
        if not reference_file.exists():
            log(f"Reference file not found: {reference_file}", "ERROR") 
            continue
        
        # Load reference data
        reference_data = load_reference_data(reference_file)
        
        # Transcribe file
        result, processing_time = transcribe_file(audio_file)
        
        if not result:
            test_results["tests"][test_name] = {
                "error": "Transcription failed",
                "processing_time": processing_time
            }
            continue
        
        # Analyze results
        detected_speakers = extract_speakers_from_result(result)
        word_comparison = compare_word_level_attribution(result, reference_data)
        
        test_results["tests"][test_name] = {
            "audio_file": str(audio_file),
            "reference_file": str(reference_file),
            "expected_speakers": test_info["expected_speakers"],
            "detected_speakers": len(detected_speakers),
            "speaker_list": detected_speakers,
            "processing_time": processing_time,
            "processing_speed": test_info["duration_min"] * 60 / processing_time if processing_time > 0 else 0,
            "word_level_comparison": word_comparison,
            "speaker_count_accuracy": abs(test_info["expected_speakers"] - len(detected_speakers)) == 0
        }
        
        log(f"Completed {test_name}: {len(detected_speakers)} speakers detected, {processing_time:.1f}s processing time")
    
    return test_results

def generate_comparison_report(results: Dict[str, Dict[str, Any]]) -> str:
    """Generate detailed comparison report"""
    report = []
    report.append("=" * 80)
    report.append("WHISPERX DIARIZATION SENSITIVITY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Configuration comparison
    report.append("CONFIGURATION COMPARISON")
    report.append("-" * 40)
    for config_name, result in results.items():
        if "error" not in result:
            report.append(f"{result['config']['description']}:")
            for param, value in result['config'].items():
                if param != 'description':
                    report.append(f"  {param}: {value}")
            report.append("")
    
    # Test results comparison
    report.append("TEST RESULTS COMPARISON")
    report.append("-" * 40)
    
    # Create comparison table
    test_names = list(TEST_FILES.keys())
    config_names = list(results.keys())
    
    for test_name in test_names:
        report.append(f"\n{test_name.upper()} TEST ({TEST_FILES[test_name]['audio']})")
        report.append("Expected speakers: " + str(TEST_FILES[test_name]['expected_speakers']))
        report.append("")
        
        # Headers
        report.append(f"{'Metric':<25} {'Config 1':<15} {'Config 2':<15} {'Difference':<15}")
        report.append("-" * 70)
        
        config1_data = results[config_names[0]]['tests'].get(test_name, {})
        config2_data = results[config_names[1]]['tests'].get(test_name, {})
        
        # Speaker count comparison
        speakers1 = config1_data.get('detected_speakers', 0)
        speakers2 = config2_data.get('detected_speakers', 0)
        diff_speakers = speakers2 - speakers1
        report.append(f"{'Detected Speakers':<25} {speakers1:<15} {speakers2:<15} {diff_speakers:+d}")
        
        # Processing time comparison  
        time1 = config1_data.get('processing_time', 0)
        time2 = config2_data.get('processing_time', 0)
        diff_time = time2 - time1
        report.append(f"{'Processing Time (s)':<25} {time1:<15.1f} {time2:<15.1f} {diff_time:+.1f}")
        
        # Processing speed comparison
        speed1 = config1_data.get('processing_speed', 0)
        speed2 = config2_data.get('processing_speed', 0)
        diff_speed = speed2 - speed1
        report.append(f"{'Processing Speed (x)':<25} {speed1:<15.1f} {speed2:<15.1f} {diff_speed:+.1f}")
        
        # Word accuracy comparison
        acc1 = config1_data.get('word_level_comparison', {}).get('speaker_accuracy', 0)
        acc2 = config2_data.get('word_level_comparison', {}).get('speaker_accuracy', 0)
        diff_acc = acc2 - acc1
        report.append(f"{'Word Accuracy':<25} {acc1:<15.3f} {acc2:<15.3f} {diff_acc:+.3f}")
        
        # Speaker count accuracy
        sc_acc1 = "✓" if config1_data.get('speaker_count_accuracy', False) else "✗"
        sc_acc2 = "✓" if config2_data.get('speaker_count_accuracy', False) else "✗"
        report.append(f"{'Speaker Count Accurate':<25} {sc_acc1:<15} {sc_acc2:<15} {'-':<15}")
        
        report.append("")
    
    # Summary and recommendations
    report.append("SUMMARY AND RECOMMENDATIONS")
    report.append("-" * 40)
    
    # Calculate overall metrics
    config1_total_time = sum([r['tests'][t].get('processing_time', 0) 
                             for r in results.values() for t in r['tests'] 
                             if 'config_1' in r.get('config_name', '')])
    config2_total_time = sum([r['tests'][t].get('processing_time', 0) 
                             for r in results.values() for t in r['tests'] 
                             if 'config_2' in r.get('config_name', '')])
    
    config1_accuracy = sum([r['tests'][t].get('word_level_comparison', {}).get('speaker_accuracy', 0) 
                           for r in results.values() for t in r['tests'] 
                           if 'config_1' in r.get('config_name', '') and 'word_level_comparison' in r['tests'][t]])
    config2_accuracy = sum([r['tests'][t].get('word_level_comparison', {}).get('speaker_accuracy', 0) 
                           for r in results.values() for t in r['tests'] 
                           if 'config_2' in r.get('config_name', '') and 'word_level_comparison' in r['tests'][t]])
    
    report.append(f"Configuration 1 total processing time: {config1_total_time:.1f}s")
    report.append(f"Configuration 2 total processing time: {config2_total_time:.1f}s")
    report.append(f"Processing time difference: {config2_total_time - config1_total_time:+.1f}s")
    report.append("")
    report.append(f"Configuration 1 average accuracy: {config1_accuracy/len(test_names):.3f}")
    report.append(f"Configuration 2 average accuracy: {config2_accuracy/len(test_names):.3f}")  
    report.append(f"Accuracy difference: {(config2_accuracy - config1_accuracy)/len(test_names):+.3f}")
    report.append("")
    
    # Recommendations
    if config2_total_time < config1_total_time and config2_accuracy >= config1_accuracy:
        report.append("✅ RECOMMENDATION: Use Configuration 2 (faster with equal/better accuracy)")
    elif config2_accuracy > config1_accuracy and config2_total_time <= config1_total_time * 1.1:
        report.append("✅ RECOMMENDATION: Use Configuration 2 (better accuracy, acceptable speed)")  
    elif config1_total_time < config2_total_time and config1_accuracy >= config2_accuracy:
        report.append("✅ RECOMMENDATION: Keep Configuration 1 (faster with equal/better accuracy)")
    else:
        report.append("⚠️  RECOMMENDATION: Mixed results - consider use case priorities")
    
    return "\n".join(report)

def main():
    """Main execution function"""
    log("Starting WhisperX Diarization Sensitivity Analysis")
    
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
    
    all_results = {}
    
    # Run tests for each configuration
    for config_name, config in CONFIGS.items():
        try:
            results = run_test_configuration(config_name, config)
            all_results[config_name] = results
            log(f"Completed testing {config['description']}")
        except Exception as e:
            log(f"Error testing {config_name}: {e}", "ERROR")
            all_results[config_name] = {"error": str(e)}
    
    # Generate report
    if all(not result.get('error') for result in all_results.values()):
        report = generate_comparison_report(all_results)
    else:
        report = "ERROR: One or more configurations failed to run properly.\n"
        for config_name, result in all_results.items():
            if result.get('error'):
                report += f"{config_name}: {result['error']}\n"
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as JSON
    results_file = Path(f"/home/kyle/Desktop/whisper/sensitivity_analysis_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save report as text
    report_file = Path(f"/home/kyle/Desktop/whisper/sensitivity_analysis_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    
    log(f"Results saved to: {results_file}")
    log(f"Report saved to: {report_file}")
    
    print("\n" + "="*80)
    print("PERFORMANCE SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    print(report)

if __name__ == "__main__":
    main() 