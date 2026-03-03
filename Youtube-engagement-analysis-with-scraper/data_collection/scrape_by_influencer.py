"""
Scrape transcripts for AI influencers - organized by influencer folders.
10 videos per influencer, 10 influencers = 100 total transcripts.
Uses Whisper tiny model for fast local transcription.
"""

import os
import sys
import json
import tempfile
import glob
import re

# Add ffmpeg to PATH
ffmpeg_path = r"C:\Users\isha and gaurav\AppData\Roaming\Python\Python314\site-packages\imageio_ffmpeg\binaries"
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

import yt_dlp
import whisper

# Configuration
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "ai_influencers")
VIDEOS_PER_INFLUENCER = 10  # 10 videos each = 100 total for 10 influencers

# AI Influencers list
AI_INFLUENCERS = [
    "Lu do Magalu",
    "Lil Miquela", 
    "Shudu Gram",
    "Noonoouri",
    "Aitana Lopez",
    "Imma",
    "Rozy",
    "Leya Love",
    "Kyra",
    "Milla Sofia"
]

# Load Whisper model (tiny for speed)
print("Loading Whisper model (tiny)...")
MODEL = whisper.load_model("tiny")


def sanitize_folder_name(name):
    """Convert influencer name to valid folder name."""
    # Remove special characters, keep letters, numbers, spaces
    sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
    return sanitized.strip()


def search_videos(query, max_results=15):
    """Search YouTube for videos matching query."""
    search_query = f'ytsearch{max_results}:{query}'
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'skip_download': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(search_query, download=False)
            if results and 'entries' in results:
                return results['entries']
    except Exception as e:
        print(f"    Search error: {e}")
    return []


def transcribe_video(video_id, video_url):
    """Download audio and transcribe with Whisper."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            'format': 'bestaudio/best',  # More flexible format selection
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
        }
        
        try:
            # Download audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(video_url, download=True)
            
            # Find downloaded file
            files = glob.glob(os.path.join(temp_dir, '*'))
            if not files:
                return None
            
            audio_file = files[0]
            print(f"    Transcribing with Whisper... ({os.path.basename(audio_file)})")
            
            # Transcribe
            result = MODEL.transcribe(audio_file, fp16=False)
            transcript = result.get('text', '').strip()
            
            if transcript:
                return transcript
                
        except Exception as e:
            print(f"    Transcription error: {e}")
    
    return None


def load_existing_transcripts(folder_path):
    """Load existing video IDs from a folder."""
    existing_ids = set()
    jsonl_path = os.path.join(folder_path, "transcripts.jsonl")
    
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    existing_ids.add(data.get('video_id'))
                except:
                    pass
    
    return existing_ids


def count_transcripts(folder_path):
    """Count transcripts in a folder."""
    jsonl_path = os.path.join(folder_path, "transcripts.jsonl")
    if not os.path.exists(jsonl_path):
        return 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def save_transcript(folder_path, data):
    """Append transcript to folder's jsonl file."""
    jsonl_path = os.path.join(folder_path, "transcripts.jsonl")
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def scrape_influencer(influencer_name, folder_path, target_count=10):
    """Scrape transcripts for a single influencer."""
    print(f"\n{'='*60}")
    print(f"INFLUENCER: {influencer_name}")
    print(f"{'='*60}")
    
    # Create folder
    os.makedirs(folder_path, exist_ok=True)
    
    # Load existing
    existing_ids = load_existing_transcripts(folder_path)
    current_count = len(existing_ids)
    
    if current_count >= target_count:
        print(f"  Already have {current_count} transcripts. Skipping.")
        return current_count
    
    print(f"  Current: {current_count}/{target_count}")
    
    # Search queries for this influencer
    queries = [
        f"{influencer_name} virtual influencer interview",
        f"{influencer_name} AI influencer explained",
        f"{influencer_name} virtual model",
        f"who is {influencer_name}",
        f"{influencer_name} documentary",
    ]
    
    for query in queries:
        if current_count >= target_count:
            break
        
        print(f"\n  Searching: {query}")
        videos = search_videos(query)
        
        if not videos:
            print(f"    No videos found")
            continue
        
        # Filter by duration (1-10 minutes)
        filtered = []
        for v in videos:
            if v and isinstance(v, dict):
                duration = v.get('duration', 0) or 0
                if 60 <= duration <= 600:
                    filtered.append(v)
        
        print(f"    Found {len(videos)} videos, {len(filtered)} in 1-10 min range")
        
        for video in filtered:
            if current_count >= target_count:
                break
            
            video_id = video.get('id')
            if not video_id or video_id in existing_ids:
                continue
            
            title = video.get('title', 'Unknown')[:50]
            duration = video.get('duration', 0)
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            print(f"\n  Processing: {title}... ({duration}s)")
            print(f"    Downloading audio...")
            
            transcript = transcribe_video(video_id, video_url)
            
            if transcript and len(transcript) > 100:
                data = {
                    'video_id': video_id,
                    'title': video.get('title', ''),
                    'influencer': influencer_name,
                    'query': query,
                    'duration': duration,
                    'transcript': transcript,
                    'transcript_length': len(transcript),
                    'method': 'whisper_tiny'
                }
                
                save_transcript(folder_path, data)
                existing_ids.add(video_id)
                current_count += 1
                
                print(f"    SUCCESS! Got {len(transcript)} chars")
                print(f"    Progress: {current_count}/{target_count}")
            else:
                print(f"    FAILED - no transcript")
    
    return current_count


def main():
    print("="*60)
    print("AI INFLUENCER TRANSCRIPT SCRAPER")
    print("="*60)
    print(f"\nTarget: {len(AI_INFLUENCERS)} influencers x {VIDEOS_PER_INFLUENCER} videos = {len(AI_INFLUENCERS) * VIDEOS_PER_INFLUENCER} total")
    print(f"Output: {BASE_OUTPUT_DIR}")
    print("\nUsing Whisper tiny model for fast local transcription.\n")
    
    # Create base directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    total_collected = 0
    
    for influencer in AI_INFLUENCERS:
        folder_name = sanitize_folder_name(influencer)
        folder_path = os.path.join(BASE_OUTPUT_DIR, folder_name)
        
        count = scrape_influencer(influencer, folder_path, VIDEOS_PER_INFLUENCER)
        total_collected += count
    
    print("\n" + "="*60)
    print("SCRAPING COMPLETE!")
    print("="*60)
    print(f"\nTotal transcripts collected: {total_collected}")
    print(f"\nFolders created in: {BASE_OUTPUT_DIR}")
    
    # Summary
    print("\nPer-influencer summary:")
    for influencer in AI_INFLUENCERS:
        folder_name = sanitize_folder_name(influencer)
        folder_path = os.path.join(BASE_OUTPUT_DIR, folder_name)
        count = count_transcripts(folder_path)
        print(f"  {influencer}: {count} transcripts")


if __name__ == "__main__":
    main()
