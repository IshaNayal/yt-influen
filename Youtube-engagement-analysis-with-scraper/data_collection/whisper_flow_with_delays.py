"""Whisper flow with rate limiting and retry logic to handle YouTube bot protection."""

import os
import json
import tempfile
import glob
import yt_dlp
import whisper
import time
import random

OUTPUT_DIR = r"c:\Users\isha and gaurav\OneDrive\Desktop\yt-influe\Youtube-engagement-analysis-with-scraper\output\ai_influencers"
AI_INFLUENCERS = ["Lu do Magalu"]  # Focus on Lu do Magalu first
VIDEOS_PER_INFLUENCER = 100

def get_existing_transcripts(influencer):
    folder = os.path.join(OUTPUT_DIR, influencer)
    filepath = os.path.join(folder, "transcripts.jsonl")
    video_ids = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            video_ids.add(data.get('video_id'))
                        except:
                            pass
        except:
            pass
    return video_ids

def search_videos(influencer, count=150, retry=3):
    """Search with retry and backoff for rate limit handling."""
    query = f"ytsearch{count}:{influencer}"
    
    for attempt in range(retry):
        videos = []
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'socket_timeout': 30,
            'retries': 5,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                results = ydl.extract_info(query, download=False)
                if results and 'entries' in results:
                    for entry in results['entries']:
                        if entry and entry.get('id'):
                            videos.append({
                                'id': entry['id'],
                                'title': entry.get('title', '')[:80]
                            })
            
            if videos:
                return videos
        except Exception as e:
            err = str(e)
            if "bot" in err.lower() or "sign in" in err.lower():
                wait = (2 ** attempt) * 10 + random.randint(0, 10)
                print(f"    Bot check - waiting {wait}s (attempt {attempt + 1}/{retry})")
                time.sleep(wait)
            else:
                return []
    
    return []

def download_and_transcribe(video_id, model):
    """Download audio and transcribe."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                'format': 'bestaudio',
                'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'socket_timeout': 30,
                'retries': 2,
                'nopostprocessors': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
            
            audio_file = None
            for f in glob.glob(os.path.join(temp_dir, '*')):
                if os.path.isfile(f):
                    audio_file = f
                    break
            
            if not audio_file:
                return None
            
            result = model.transcribe(audio_file, fp16=False, verbose=False)
            transcript = result.get('text', '').strip()
            return transcript if len(transcript) > 10 else None
    except Exception as e:
        return None

def save_transcript(influencer, video_id, title, transcript):
    """Save transcript to file."""
    folder = os.path.join(OUTPUT_DIR, influencer)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, "transcripts.jsonl")
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            'video_id': video_id,
            'title': title,
            'influencer': influencer,
            'transcript': transcript,
        }, ensure_ascii=False) + '\n')

def main():
    print("Loading Whisper model (base)...")
    model = whisper.load_model("base")
    print("✓ Ready!\n")
    
    for influencer in AI_INFLUENCERS:
        existing = get_existing_transcripts(influencer)
        count = len(existing)
        needed = VIDEOS_PER_INFLUENCER - count
        
        if needed <= 0:
            print(f"✓ {influencer}: COMPLETE ({count}/100)")
            continue
        
        print(f"\n[{influencer}] {count}/100 - Searching videos (with rate limits)...")
        videos = search_videos(influencer, count=needed * 2)
        
        if not videos:
            print(f"  ✗ Could not fetch videos - YouTube bot check")
            print(f"  Waiting 2 minutes before retry...")
            time.sleep(120)
            continue
        
        added = 0
        for idx, video in enumerate(videos):
            if added >= needed:
                break
            
            vid_id = video['id']
            if vid_id in existing:
                continue
            
            # Add delay between downloads to avoid rate limiting
            if idx > 0:
                delay = random.randint(5, 15)
                time.sleep(delay)
            
            num = count + added + 1
            title = video['title']
            print(f"  [{num:3d}] {title[:40]:40s} ", end="", flush=True)
            
            try:
                transcript = download_and_transcribe(vid_id, model)
                if transcript:
                    save_transcript(influencer, vid_id, video['title'], transcript)
                    print("✓")
                    added += 1
                else:
                    print("✗")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print("✗")
        
        print(f"  → Added {added}/{needed}")

if __name__ == "__main__":
    main()
