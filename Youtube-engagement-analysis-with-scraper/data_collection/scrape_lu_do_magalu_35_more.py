"""Scrape 35 more transcripts for Lu do Magalu using Whisper."""

import os
import json
import tempfile
import glob
import yt_dlp
import whisper
from datetime import datetime

# Set FFmpeg path
os.environ["PATH"] = r"C:\Users\isha and gaurav\AppData\Roaming\Python\Python314\site-packages\imageio_ffmpeg\binaries;" + os.environ.get("PATH", "")

OUTPUT_DIR = r"c:\Users\isha and gaurav\OneDrive\Desktop\yt-influe\Youtube-engagement-analysis-with-scraper\output\ai_influencers\Lu do Magalu"
INFLUENCER = "Lu do Magalu"
TARGET_NEW_TRANSCRIPTS = 35

os.makedirs(OUTPUT_DIR, exist_ok=True)

def count_existing_transcripts():
    """Count existing transcripts for Lu do Magalu."""
    filepath = os.path.join(OUTPUT_DIR, "lu-do-magalu-transcripts.jsonl")
    count = 0
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
    return count

def get_existing_video_ids():
    """Get video IDs already collected."""
    filepath = os.path.join(OUTPUT_DIR, "lu-do-magalu-transcripts.jsonl")
    video_ids = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'video_id' in data:
                        video_ids.add(data['video_id'])
                except:
                    pass
    return video_ids

def search_videos(query, max_results=50):
    """Search for videos about Lu do Magalu."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    videos = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_query = f"ytsearch{max_results}:{query}"
            results = ydl.extract_info(search_query, download=False)
            if results and 'entries' in results:
                for entry in results['entries']:
                    if entry and entry.get('id'):
                        videos.append({
                            'video_id': entry['id'],
                            'title': entry.get('title', ''),
                            'duration': entry.get('duration', 0),
                            'channel': entry.get('channel', entry.get('uploader', '')),
                            'upload_date': entry.get('upload_date', ''),
                        })
    except Exception as e:
        print(f"Search error: {e}")
    
    return videos

def transcribe_video(video_id, model):
    """Download and transcribe a video using Whisper."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"    Downloading audio...")
                ydl.extract_info(url, download=True)
            
            files = glob.glob(os.path.join(temp_dir, '*'))
            if not files:
                return None
            
            audio_file = files[0]
            print(f"    Transcribing...")
            result = model.transcribe(audio_file, fp16=False)
            return result.get('text', '')
        except Exception as e:
            print(f"    Error: {e}")
            return None

def save_transcript(data):
    """Append transcript to file."""
    filepath = os.path.join(OUTPUT_DIR, "lu-do-magalu-transcripts.jsonl")
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def main():
    print("\n" + "="*60)
    print("Lu do Magalu - Scraping 35 More Transcripts")
    print("="*60)
    
    current_count = count_existing_transcripts()
    print(f"\nCurrent transcripts: {current_count}")
    print(f"Target: {current_count + TARGET_NEW_TRANSCRIPTS}")
    
    print(f"\nLoading Whisper tiny model...")
    model = whisper.load_model("tiny")
    
    existing_ids = get_existing_video_ids()
    print(f"Will skip {len(existing_ids)} already-collected videos")
    
    print(f"\nSearching for Lu do Magalu videos...")
    videos = search_videos("Lu do Magalu", max_results=75)
    print(f"Found {len(videos)} videos")
    
    # Filter out already collected
    new_videos = [v for v in videos if v['video_id'] not in existing_ids]
    print(f"New videos to process: {len(new_videos)}")
    
    collected = 0
    for i, video in enumerate(new_videos, 1):
        if collected >= TARGET_NEW_TRANSCRIPTS:
            break
        
        video_id = video['video_id']
        title = video['title'][:60]
        
        print(f"\n[{collected + 1}/{TARGET_NEW_TRANSCRIPTS}] {title}...")
        
        transcript = transcribe_video(video_id, model)
        
        if transcript and len(transcript) > 50:
            data = {
                'video_id': video_id,
                'title': video['title'],
                'channel': video['channel'],
                'influencer': INFLUENCER,
                'transcript': transcript,
                'timestamp': datetime.now().isoformat(),
            }
            save_transcript(data)
            print(f"    ✓ Saved ({len(transcript)} chars)")
            collected += 1
        else:
            print(f"    ✗ No transcript or too short")
    
    final_count = count_existing_transcripts()
    print(f"\n" + "="*60)
    print(f"COMPLETE!")
    print(f"Total transcripts for Lu do Magalu: {final_count}")
    print(f"Added: {collected} new transcripts")
    print(f"{"="*60}\n")

if __name__ == "__main__":
    main()
