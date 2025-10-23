#!/usr/bin/env python3
"""
Test concurrent transcription requests with the new fixes
"""

import requests
import json
import time
import subprocess
from datetime import datetime

# Configuration
SERVER_URL = "http://69.30.85.231:22138"
DOWNLOAD_URL = "https://kyle-test-bucket-1.s3.us-east-2.amazonaws.com/30minutes_7speakers.mp3?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEIL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMiJIMEYCIQDDqMmdMAmN8cAcHo3ZP56KAcYka8nfIpmrh6dDT0IP5wIhALmtNHZsZy8vYlKUqlF8wvwSuJjNYCweBzf8VR0N1bKEKrkDCDsQABoMNTg2Nzk0NDQxNDYxIgwa4zIP8NeNZgMYe4gqlgOpYi0uk1cKdjx5RP9FhGd1CYKVs9yqpL%2FxAXr1qihL%2F8cpe9kJxbuJtfqNdcxDsT8colCY8ckYqjIh9Xstev5jSTC37D8VM4OlQzfON47KwCY%2F412MzPPDbHR98kS17sT59SDL1PjIDvpkb7F%2BnMgbiXuw606t5fuE%2B6BBODuI%2FTo%2Bu78g54LxuP%2BhcBr66PdxLt%2FLTgj6QTPtCcBa4W15Eqe3kVx4Jd3lRxY3rUdQgZxtXcA0pLxsqx%2F%2BiE3dM36z50CbtRkzduq9CDt53pgFaVCwEINiQTXJSlN86zh%2Fuco8SFr6MxrW9f2txmAtj4zjRmTFeB0%2FpqKOoZth9xpG2%2F4eP%2BpejX9OemRnkVj%2Bb3R2Q4F9PLGHvKbgMMwGbLs4mPnYTDWvkL4ogaYaeRKSeZQ8f3%2FSA%2BOPMWo64lUq39luyVF4c2%2Fbc8A56Pmj80I07kfqQWNcttmIWYMr%2BQrG0HoWRNMzdXqouC1qFOAiFH%2FGlpYfaL1IHK%2FvqT4uxOd5yb5ADDTXyFdRy7pBgpz2rFdeL5d9MPaQ5scGOt0CGcAjTJyKofC0noW2QlylQfyZ98toI5SI%2BDGqePXDPNnNCV6ojxR%2BgcQsWZZeY%2BTqKKEuo7fA7MMXGv3I9Ff0OiPEMP7p%2BLQ%2FixGr0Mk%2Bh%2FUjfCKN8tPex98jiumofupO4lZ8YJzeUXs5avYG3Czy71DXJVA4uR4oUOFZRy%2BOmHggCZdYKcAiPRZa2gCuPFO%2F7j1PYyQIUr%2BDBMSLue0LAU1vwJLgZh8eIsNq1DeIOsNX2QU0oFyx8YG1s%2F9OT9%2B3Kf%2BnZCVXOiETctWE0M0a0RI5vWYTNm9A1cFWLPRXVzV94VDuxQFNYL7WT%2FiR9pfy3y%2FcDKJs6gyRHM6xk905YVKkC73o%2FBLZupQ9WE8MgZNZGcruDu7R69OoKr%2FJgLgCvoy%2BS%2B6xs9UAOC8Fv8OK9w%2FKJ0dVN8QuybjJojKFzp5d4kZYkA4f%2Fmp%2F1IP3dppQsY50N4hPOURlGS2F1g%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAYRH5MTL2YQL3Z2US%2F20251023%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20251023T014451Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=431b4eff7edd705167ded4262513b545ea07de5404d4fbba823dde6bba6944ac"

def generate_upload_url():
    """Generate a fresh upload presigned URL"""
    try:
        result = subprocess.run(
            ['python3', 'generate_upload_url.py'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract URL from output (last line before the separator)
        lines = result.stdout.strip().split('\n')
        for line in reversed(lines):
            if line.startswith('https://'):
                return line.strip()
        
        print("❌ Could not extract URL from output")
        return None
    except Exception as e:
        print(f"❌ Error generating upload URL: {e}")
        return None

def check_health():
    """Check server health and configuration"""
    print("\n" + "="*80)
    print("🏥 CHECKING SERVER HEALTH")
    print("="*80)
    
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=10)
        health = response.json()
        
        print(f"✅ Server Status: {health['status']}")
        print(f"🎮 GPU Available: {health['gpu_available']}")
        print(f"🖥️  Device: {health['device']}")
        
        if 'concurrency' in health:
            conc = health['concurrency']
            print(f"\n📊 CONCURRENCY CONFIG:")
            print(f"   Max Concurrent: {conc['max_concurrent_requests']}")
            print(f"   Active Jobs: {conc['active_jobs']}")
            print(f"   Processing: {conc['processing']}")
            print(f"   Queued: {conc['queued']}")
            print(f"   Available Slots: {conc['available_slots']}")
        
        if 'timeouts' in health:
            timeouts = health['timeouts']
            print(f"\n⏱️  TIMEOUT CONFIG:")
            print(f"   Upload Timeout: {timeouts['upload_timeout_seconds']}s ({timeouts['upload_timeout_minutes']} min)")
            print(f"   Note: {timeouts['note']}")
        
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def submit_job(job_number, upload_url):
    """Submit a transcription job"""
    print(f"\n{'='*80}")
    print(f"📤 SUBMITTING JOB {job_number}")
    print(f"{'='*80}")
    print(f"Upload URL: {upload_url[:80]}...")
    
    try:
        start_time = time.time()
        
        data = {
            's3_presigned_url': DOWNLOAD_URL,
            'upload_presigned_url': upload_url,
            'enable_diarization': 'true',
            'language': 'en'
        }
        
        response = requests.post(
            f"{SERVER_URL}/v1/audio/transcriptions",
            data=data,
            timeout=10
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id', 'unknown')
            status = result.get('status', 'unknown')
            
            print(f"✅ Job {job_number} submitted successfully!")
            print(f"   Job ID: {job_id}")
            print(f"   Status: {status}")
            print(f"   Response Time: {response_time:.3f}s")
            
            return {
                'job_number': job_number,
                'job_id': job_id,
                'status': status,
                'response_time': response_time,
                'upload_url': upload_url,
                'success': True
            }
        else:
            print(f"❌ Job {job_number} failed with status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return {
                'job_number': job_number,
                'success': False,
                'error': response.text
            }
            
    except Exception as e:
        print(f"❌ Job {job_number} submission failed: {e}")
        return {
            'job_number': job_number,
            'success': False,
            'error': str(e)
        }

def monitor_job(job_id, job_number):
    """Monitor a job's progress"""
    print(f"\n📊 Monitoring Job {job_number} ({job_id})...")
    
    try:
        response = requests.get(f"{SERVER_URL}/v1/jobs/{job_id}", timeout=5)
        if response.status_code == 200:
            job_info = response.json()
            status = job_info.get('status', 'unknown')
            progress = job_info.get('progress', 'unknown')
            elapsed = job_info.get('elapsed_time', 0)
            
            print(f"   Status: {status} | Progress: {progress} | Elapsed: {elapsed:.1f}s")
            return job_info
        else:
            print(f"   ⚠️  Could not fetch status: {response.status_code}")
            return None
    except Exception as e:
        print(f"   ⚠️  Error monitoring: {e}")
        return None

def check_all_jobs():
    """Check all active jobs"""
    print(f"\n{'='*80}")
    print(f"📋 CHECKING ALL ACTIVE JOBS")
    print(f"{'='*80}")
    
    try:
        response = requests.get(f"{SERVER_URL}/v1/jobs", timeout=5)
        if response.status_code == 200:
            jobs_data = response.json()
            total = jobs_data.get('total_jobs', 0)
            jobs = jobs_data.get('jobs', {})
            
            print(f"Total Active Jobs: {total}")
            
            if jobs:
                print("\nJob Details:")
                for job_id, job_info in jobs.items():
                    status = job_info.get('status', 'unknown')
                    progress = job_info.get('progress', 'N/A')
                    elapsed = job_info.get('elapsed_time', 0)
                    print(f"  {job_id[:8]}... | Status: {status:12} | Progress: {progress:25} | Elapsed: {elapsed:.1f}s")
            
            return jobs_data
        else:
            print(f"❌ Could not fetch jobs: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error checking jobs: {e}")
        return None

def main():
    print("="*80)
    print("🧪 CONCURRENT TRANSCRIPTION TEST - Version 5.2")
    print("="*80)
    print(f"Server: {SERVER_URL}")
    print(f"Audio: 30minutes_7speakers.mp3")
    print(f"Test: 2 concurrent requests")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check server health
    if not check_health():
        print("\n❌ Server health check failed. Aborting test.")
        return
    
    # Step 2: Generate upload URLs
    print(f"\n{'='*80}")
    print("🔗 GENERATING UPLOAD URLS")
    print(f"{'='*80}")
    
    upload_urls = []
    for i in range(2):
        print(f"\nGenerating URL {i+1}/2...")
        url = generate_upload_url()
        if url:
            print(f"✅ URL {i+1} generated")
            upload_urls.append(url)
            time.sleep(1)  # Brief delay between URL generations
        else:
            print(f"❌ Failed to generate URL {i+1}")
            return
    
    print(f"\n✅ All 2 upload URLs generated successfully!")
    
    # Step 3: Submit jobs concurrently
    print(f"\n{'='*80}")
    print("🚀 SUBMITTING 2 CONCURRENT JOBS")
    print(f"{'='*80}")
    
    jobs = []
    test_start = time.time()
    
    for i in range(2):
        job_result = submit_job(i+1, upload_urls[i])
        jobs.append(job_result)
        time.sleep(0.5)  # Small delay between submissions
    
    # Step 4: Check initial status
    time.sleep(2)
    check_all_jobs()
    
    # Step 5: Monitor jobs periodically
    print(f"\n{'='*80}")
    print("👀 MONITORING JOB PROGRESS")
    print(f"{'='*80}")
    print("Will check every 30 seconds for up to 10 minutes...")
    
    monitoring_duration = 600  # 10 minutes
    check_interval = 30  # 30 seconds
    checks = 0
    max_checks = monitoring_duration // check_interval
    
    while checks < max_checks:
        time.sleep(check_interval)
        checks += 1
        
        print(f"\n⏰ Check #{checks} ({checks * check_interval}s elapsed)")
        
        # Check all jobs
        jobs_data = check_all_jobs()
        
        if jobs_data:
            jobs_info = jobs_data.get('jobs', {})
            
            # Count statuses
            completed = sum(1 for j in jobs_info.values() if j.get('status') == 'completed')
            failed = sum(1 for j in jobs_info.values() if j.get('status') == 'failed')
            processing = sum(1 for j in jobs_info.values() if j.get('status') == 'processing')
            queued = sum(1 for j in jobs_info.values() if j.get('status') == 'queued')
            
            print(f"\n📊 Summary: Completed: {completed} | Failed: {failed} | Processing: {processing} | Queued: {queued}")
            
            # Check if all our jobs are done
            all_done = True
            for job in jobs:
                if job.get('success') and job.get('job_id'):
                    job_id = job['job_id']
                    if job_id in jobs_info:
                        status = jobs_info[job_id].get('status')
                        if status not in ['completed', 'failed']:
                            all_done = False
            
            if all_done:
                print("\n🎉 All jobs completed!")
                break
    
    # Final summary
    test_duration = time.time() - test_start
    
    print(f"\n{'='*80}")
    print("📊 FINAL TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Test Duration: {test_duration:.1f}s ({test_duration/60:.1f} minutes)")
    
    final_jobs = check_all_jobs()
    
    if final_jobs:
        jobs_info = final_jobs.get('jobs', {})
        
        print(f"\n✅ SUCCESS METRICS:")
        for i, job in enumerate(jobs, 1):
            if job.get('success'):
                job_id = job.get('job_id')
                if job_id in jobs_info:
                    info = jobs_info[job_id]
                    status = info.get('status')
                    elapsed = info.get('elapsed_time', 0)
                    
                    icon = "✅" if status == "completed" else "❌" if status == "failed" else "⏳"
                    print(f"   {icon} Job {i}: {status:12} | {elapsed:.1f}s")
                    
                    if status == "failed" and 'error' in info:
                        print(f"      Error: {info['error']}")
    
    print(f"\n{'='*80}")
    print("✅ TEST COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

