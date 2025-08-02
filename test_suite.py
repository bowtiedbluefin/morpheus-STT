import os
import requests
import json
import difflib
import re

# --- Configuration ---
SERVER_URL = "http://localhost:3333/v1/audio/transcriptions"
TEST_DATA_DIR = "/home/kyle/Desktop/whisper/Morpheus Test-20250801T195813Z-1-001/Morpheus Test"
GOLDEN_TRANSCRIPT_EXTENSION = "_deepgram.json"
AUDIO_EXTENSION = ".mp3"

def normalize_text_for_speech(text):
    """Normalize text for better speech transcription comparison."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", "", text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Handle contractions consistently
    text = text.replace("we're", "we are")
    text = text.replace("today,", "today")
    return text

def speech_similarity(text1, text2):
    """Calculate similarity appropriate for speech transcription."""
    # Normalize both texts
    norm1 = normalize_text_for_speech(text1)
    norm2 = normalize_text_for_speech(text2)
    
    # Word-level similarity (better for speech than character-level)
    words1 = norm1.split()
    words2 = norm2.split()
    
    # Use SequenceMatcher on normalized words
    word_similarity = difflib.SequenceMatcher(None, words1, words2).ratio()
    
    # Word overlap similarity
    words1_set = set(words1)
    words2_set = set(words2)
    intersection = words1_set.intersection(words2_set)
    union = words1_set.union(words2_set)
    overlap_similarity = len(intersection) / len(union) if union else 0
    
    # Combined score (weighted average favoring sequence)
    combined = (word_similarity * 0.7) + (overlap_similarity * 0.3)
    
    return combined

def find_test_files(directory):
    """Finds audio files and their corresponding golden transcripts."""
    print(f"Searching for test files in: '{directory}'")
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return []

    test_files = []
    print(f"Walking directory: '{directory}'...")
    for root, _, files in os.walk(directory):
        print(f"  In directory: '{root}'")
        for file in files:
            if file.endswith(AUDIO_EXTENSION):
                name, _ = os.path.splitext(file)
                # Correctly form the transcript filename by removing speaker info
                base_name = "_".join(name.split('_')[:-1])
                transcript_path = os.path.join(root, base_name + GOLDEN_TRANSCRIPT_EXTENSION)
                print(f"    Found audio file: '{file}'. Checking for transcript: '{os.path.basename(transcript_path)}'")
                if os.path.exists(transcript_path):
                    print(f"      Found matching transcript.")
                    test_files.append({
                        "audio_path": os.path.join(root, file),
                        "transcript_path": transcript_path
                    })
                else:
                    print(f"      Matching transcript NOT found.")
    print(f"Found {len(test_files)} total test files.")
    return test_files

def run_transcription_test(audio_path):
    """Sends a request to the transcription server and gets the result."""
    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f)}
            params = {
                "response_format": "json",
                "enable_diarization": "true",
            }
            response = requests.post(SERVER_URL, files=files, data=params)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API for {os.path.basename(audio_path)}: {e}")
        return None

def get_golden_transcript_from_json(json_path):
    """Extracts the full transcript from a Deepgram JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["results"]["channels"][0]["alternatives"][0]["transcript"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing golden transcript {os.path.basename(json_path)}: {e}")
        return ""

def compare_transcripts(golden_text, actual_text):
    """Compares two transcripts using speech-appropriate similarity."""
    return speech_similarity(golden_text, actual_text)

def main():
    """Main function to run the test suite."""
    print("--- Starting Transcription Test Suite ---")
    test_files = find_test_files(TEST_DATA_DIR)

    if not test_files:
        print(f"No test files found in '{TEST_DATA_DIR}'.")
        return

    passed_count = 0
    failed_tests = []

    for test in test_files:
        audio_file = os.path.basename(test["audio_path"])
        print(f"\nTesting: {audio_file}")

        # Get actual transcription from API
        api_result = run_transcription_test(test["audio_path"])
        if not api_result:
            failed_tests.append({"file": audio_file, "reason": "API request failed."})
            continue

        actual_transcript = api_result.get("text", "")

        # Get golden transcript from JSON
        golden_transcript = get_golden_transcript_from_json(test["transcript_path"])

        # Compare using better similarity measurement
        similarity = compare_transcripts(golden_transcript, actual_transcript)

        # Use 85% threshold for speech transcription (more realistic)
        if similarity > 0.85:  
            print(f"  [PASS] Similarity: {similarity:.2%}")
            passed_count += 1
        else:
            print(f"  [FAIL] Similarity: {similarity:.2%}")
            failed_tests.append({
                "file": audio_file,
                "similarity": similarity,
                "golden": golden_transcript,
                "actual": actual_transcript
            })

    print("\n--- Test Suite Summary ---")
    print(f"Total tests: {len(test_files)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print("\n--- Failed Tests Details ---")
        for failure in failed_tests:
            print(f"\nFile: {failure['file']}")
            if "reason" in failure:
                print(f"  Reason: {failure['reason']}")
            else:
                print(f"  Similarity: {failure['similarity']:.2%}")
                print(f"  Golden: '{failure['golden']}'")
                print(f"  Actual: '{failure['actual']}'")

if __name__ == "__main__":
    main() 