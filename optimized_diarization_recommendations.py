#!/usr/bin/env python3
"""
Comprehensive Diarization Parameter Optimization Recommendations
==============================================================

Based on the detailed word-level attribution analysis, this script provides
targeted recommendations and implements extended parameter testing to address:

1. Speaker over-detection (10 detected vs 7 expected)
2. Low attribution accuracy (68.8% average)
3. Parameter tuning ineffectiveness

Key Findings from Analysis:
- Current parameter changes (0.55→0.5 clustering, 0.40→0.3 segmentation, 3.0→1.0s duration) 
  produced ZERO improvement
- Need more dramatic parameter adjustments
- Focus on clustering threshold to reduce over-detection
- Address model compatibility issues
"""

import os
import json
import requests
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Configuration
SERVER_URL = "http://localhost:3337"
TEST_FILES_DIR = Path("/home/kyle/Desktop/whisper/Morpheus Test-20250801T195813Z-1-001/Morpheus Test")
ENV_FILE = Path("/home/kyle/Desktop/whisper/working_gpu.env")

# RECOMMENDED PARAMETER CONFIGURATIONS
# Based on analysis results and diarization research
PARAMETER_TESTS = {
    "current_baseline": {
        "PYANNOTE_CLUSTERING_THRESHOLD": "0.55",
        "PYANNOTE_SEGMENTATION_THRESHOLD": "0.40", 
        "MIN_SPEAKER_DURATION": "3.0",
        "description": "Current Baseline (Known Results)"
    },
    "aggressive_clustering": {
        "PYANNOTE_CLUSTERING_THRESHOLD": "0.75",  # MUCH higher to reduce over-detection
        "PYANNOTE_SEGMENTATION_THRESHOLD": "0.40",
        "MIN_SPEAKER_DURATION": "5.0",  # Longer minimum duration
        "description": "Aggressive Anti-Over-detection"
    },
    "conservative_clustering": {
        "PYANNOTE_CLUSTERING_THRESHOLD": "0.65",  # Moderately higher
        "PYANNOTE_SEGMENTATION_THRESHOLD": "0.45",
        "MIN_SPEAKER_DURATION": "4.0",
        "description": "Conservative Anti-Over-detection"
    },
    "high_precision": {
        "PYANNOTE_CLUSTERING_THRESHOLD": "0.80",  # Very high for precision
        "PYANNOTE_SEGMENTATION_THRESHOLD": "0.50",
        "MIN_SPEAKER_DURATION": "6.0",
        "description": "High Precision (May Under-detect)"
    },
    "balanced_improved": {
        "PYANNOTE_CLUSTERING_THRESHOLD": "0.70",  # Balanced approach
        "PYANNOTE_SEGMENTATION_THRESHOLD": "0.45",
        "MIN_SPEAKER_DURATION": "3.5",
        "description": "Balanced Improvement"
    },
    "fine_grained": {
        "PYANNOTE_CLUSTERING_THRESHOLD": "0.45",  # Lower for more sensitivity
        "PYANNOTE_SEGMENTATION_THRESHOLD": "0.30",
        "MIN_SPEAKER_DURATION": "2.0",
        "description": "Fine-Grained Detection (May Over-detect)"
    }
}

# ADDITIONAL PARAMETERS TO TEST
# These aren't currently exposed but should be considered
ADVANCED_PARAMETERS = {
    "segmentation_onset": "0.5",  # Voice activity detection sensitivity
    "segmentation_offset": "0.5",
    "clustering_method": "centroid",  # or "complete", "average"
    "embedding_batch_size": "32",
    "segmentation_batch_size": "32"
}

def log(message: str, level: str = "INFO"):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

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

def restart_server() -> bool:
    """Restart the diarization server to pick up new configuration"""
    try:
        log("Restarting server to apply new configuration...")
        
        # Kill existing server
        subprocess.run(["pkill", "-f", "working_gpu_diarization_server.py"], check=False)
        time.sleep(3)
        
        # Start server in background
        env = os.environ.copy()
        # Use environment variable for HuggingFace token - don't hardcode it!
        if "HUGGINGFACE_TOKEN" not in env:
            print("WARNING: HUGGINGFACE_TOKEN not set in environment")
        
        subprocess.Popen(
            ["bash", "/home/kyle/Desktop/whisper/fixed_gpu_start.sh"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for server to be ready
        return wait_for_server(90)
        
    except Exception as e:
        log(f"Error restarting server: {e}", "ERROR")
        return False

def quick_test_config(config: Dict[str, str], config_name: str) -> Dict[str, Any]:
    """Quick test of configuration using 5-minute file"""
    try:
        # Update configuration
        if not update_env_config(config):
            return {"error": "Failed to update config"}
        
        # Restart server (required for config changes)
        if not restart_server():
            return {"error": "Failed to restart server"}
        
        # Test with 5-minute file
        audio_file = TEST_FILES_DIR / "5minutes_5speakers.mp3"
        
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            data = {'enable_diarization': 'true', 'response_format': 'json'}
            
            response = requests.post(
                f"{SERVER_URL}/v1/audio/transcriptions",
                files=files,
                data=data,
                timeout=180
            )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract basic metrics
            speakers = set()
            word_count = 0
            
            if 'result' in result and 'word_segments' in result['result']:
                for word in result['result']['word_segments']:
                    speakers.add(word.get('speaker', 'unknown'))
                    word_count += 1
            
            return {
                "config_name": config_name,
                "speakers_detected": len(speakers),
                "word_count": word_count,
                "speakers_list": sorted(list(speakers)),
                "config": config,
                "success": True
            }
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
            
    except Exception as e:
        return {"error": str(e)}

def generate_recommendations_report(results: Dict[str, Dict[str, Any]]) -> str:
    """Generate comprehensive recommendations report"""
    report = []
    report.append("=" * 100)
    report.append("DIARIZATION PARAMETER OPTIMIZATION RECOMMENDATIONS")
    report.append("=" * 100)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("=" * 50)
    report.append("Based on detailed word-level attribution analysis:")
    report.append("• Current parameters (0.55→0.5 clustering) show ZERO improvement")
    report.append("• Major issue: Over-detection (10 speakers vs 7 expected)")
    report.append("• Attribution accuracy: 68.8% average (needs improvement)")
    report.append("• Solution: More aggressive parameter adjustments needed")
    report.append("")
    
    # Key Problems Identified
    report.append("KEY PROBLEMS IDENTIFIED")
    report.append("=" * 30)
    report.append("1. SPEAKER OVER-DETECTION:")
    report.append("   - 30min test: 10 detected vs 7 expected (43% over-detection)")
    report.append("   - 5min test: 5 detected vs 5 expected (correct but close)")
    report.append("")
    report.append("2. POOR WORD-LEVEL ATTRIBUTION:")
    report.append("   - 5min test: 65.0% accuracy (35% misattribution)")
    report.append("   - 30min test: 72.7% accuracy (27% misattribution)")
    report.append("")
    report.append("3. PARAMETER INEFFECTIVENESS:")
    report.append("   - Current tweaks too minor to create measurable change")
    report.append("   - Need more dramatic threshold adjustments")
    report.append("")
    
    # Specific Recommendations
    report.append("SPECIFIC RECOMMENDATIONS")
    report.append("=" * 30)
    report.append("")
    report.append("🎯 PRIMARY RECOMMENDATION: AGGRESSIVE CLUSTERING")
    report.append("-" * 50)
    report.append("PYANNOTE_CLUSTERING_THRESHOLD = 0.75  (was 0.55)")
    report.append("MIN_SPEAKER_DURATION = 5.0s            (was 3.0s)")
    report.append("PYANNOTE_SEGMENTATION_THRESHOLD = 0.40  (keep current)")
    report.append("")
    report.append("Expected Results:")
    report.append("• Reduce over-detection from 10→7 speakers")
    report.append("• Improve attribution by merging similar speakers")
    report.append("• May slightly reduce sensitivity but improve accuracy")
    report.append("")
    
    report.append("🔧 SECONDARY RECOMMENDATION: ADDRESS ROOT CAUSES")
    report.append("-" * 50)
    report.append("1. FIX MODEL COMPATIBILITY WARNINGS:")
    report.append("   - pyannote.audio 3.3.2 vs model trained on 0.0.1")
    report.append("   - PyTorch 2.6.0+cu126 vs model trained on 1.10.0+cu102")
    report.append("   - Consider downgrading pyannote.audio to 3.1.x")
    report.append("")
    report.append("2. ENABLE TF32 FOR BETTER ACCURACY:")
    report.append("   - Server disables TF32 but it may improve precision")
    report.append("   - Test with torch.backends.cuda.matmul.allow_tf32 = True")
    report.append("")
    report.append("3. EXPOSE ADDITIONAL PARAMETERS:")
    report.append("   - Segmentation onset/offset thresholds")
    report.append("   - Clustering method (centroid vs complete linkage)")
    report.append("   - Embedding model batch sizes")
    report.append("")
    
    # Test Results Summary
    if results:
        report.append("QUICK TEST RESULTS")
        report.append("=" * 20)
        
        # Sort by speaker count (ascending = better for over-detection)
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if v.get('success')],
            key=lambda x: x[1].get('speakers_detected', 999)
        )
        
        report.append(f"{'Configuration':<25} {'Speakers':<10} {'Expected':<10} {'Status':<15}")
        report.append("-" * 70)
        
        for config_name, result in sorted_results:
            speakers = result.get('speakers_detected', 'Error')
            expected = "5" if '5min' in str(result) else "5"  # Using 5min test
            status = "✅ GOOD" if speakers == 5 else f"❌ Over: +{speakers-5}" if speakers > 5 else f"⚠️ Under: {speakers-5}"
            report.append(f"{config_name:<25} {speakers:<10} {expected:<10} {status:<15}")
        
        # Best configuration
        if sorted_results:
            best_config, best_result = sorted_results[0]
            if best_result.get('speakers_detected', 99) == 5:
                report.append("")
                report.append(f"🏆 BEST CONFIGURATION: {best_config}")
                report.append("   - Exactly correct speaker count")
                report.append("   - Recommended for production testing")
    
    # Implementation Steps
    report.append("")
    report.append("IMPLEMENTATION STEPS")
    report.append("=" * 25)
    report.append("1. Update working_gpu.env with recommended parameters:")
    report.append("   PYANNOTE_CLUSTERING_THRESHOLD=0.75")
    report.append("   MIN_SPEAKER_DURATION=5.0")
    report.append("")
    report.append("2. Restart diarization server:")
    report.append("   pkill -f working_gpu_diarization_server.py")
    report.append("   HUGGINGFACE_TOKEN=hf_... ./fixed_gpu_start.sh")
    report.append("")
    report.append("3. Test with your audio files:")
    report.append("   python3 word_level_speaker_attribution_analysis.py")
    report.append("")
    report.append("4. If still over-detecting, try HIGH PRECISION config:")
    report.append("   PYANNOTE_CLUSTERING_THRESHOLD=0.80")
    report.append("   MIN_SPEAKER_DURATION=6.0")
    report.append("")
    
    # Advanced Optimizations
    report.append("ADVANCED OPTIMIZATIONS")
    report.append("=" * 25)
    report.append("If basic parameter adjustments don't resolve the issues:")
    report.append("")
    report.append("1. POST-PROCESSING SPEAKER MERGING:")
    report.append("   - Merge speakers with high voice similarity")
    report.append("   - Combine speakers with minimal speaking time")
    report.append("   - Use speaker embedding similarity thresholds")
    report.append("")
    report.append("2. MODEL UPGRADES:")
    report.append("   - Try pyannote/speaker-diarization-3.0 vs 3.1")
    report.append("   - Experiment with different embedding models")
    report.append("   - Consider fine-tuning on your specific audio domain")
    report.append("")
    report.append("3. PREPROCESSING OPTIMIZATIONS:")
    report.append("   - Audio normalization and noise reduction")
    report.append("   - Voice activity detection tuning") 
    report.append("   - Audio segmentation preprocessing")
    report.append("")
    
    # Monitoring and Validation
    report.append("MONITORING & VALIDATION")
    report.append("=" * 25)
    report.append("Track these metrics after implementing changes:")
    report.append("• Speaker Detection Accuracy: Target exactly expected count")
    report.append("• Word Attribution Accuracy: Target >80% (from current 68.8%)")
    report.append("• Processing Speed: Ensure no significant degradation")
    report.append("• False Positive Rate: Minimize extra speakers")
    report.append("• False Negative Rate: Avoid missing real speakers")
    report.append("")
    
    return "\n".join(report)

def main():
    """Generate comprehensive recommendations and quick test selected configs"""
    log("Starting Comprehensive Diarization Parameter Optimization Analysis")
    
    # Quick test a few key configurations
    log("Running quick tests on key parameter configurations...")
    
    test_configs = [
        ("current_baseline", PARAMETER_TESTS["current_baseline"]),
        ("aggressive_clustering", PARAMETER_TESTS["aggressive_clustering"]),
        ("conservative_clustering", PARAMETER_TESTS["conservative_clustering"]),
        ("high_precision", PARAMETER_TESTS["high_precision"])
    ]
    
    results = {}
    
    for config_name, config in test_configs:
        log(f"Testing {config['description']}...")
        result = quick_test_config(config, config_name)
        results[config_name] = result
        
        if result.get('success'):
            log(f"✅ {config_name}: {result['speakers_detected']} speakers detected")
        else:
            log(f"❌ {config_name}: {result.get('error', 'Unknown error')}", "ERROR")
    
    # Generate comprehensive report
    report = generate_recommendations_report(results)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(f"/home/kyle/Desktop/whisper/diarization_optimization_recommendations_{timestamp}.txt")
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Save test results
    results_file = Path(f"/home/kyle/Desktop/whisper/parameter_test_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    log(f"Recommendations saved to: {report_file}")
    log(f"Test results saved to: {results_file}")
    
    print("\n" + "="*100)
    print("DIARIZATION OPTIMIZATION RECOMMENDATIONS")
    print("="*100)
    print(report)

if __name__ == "__main__":
    main() 