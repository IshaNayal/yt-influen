#!/usr/bin/env python3
"""
Import existing transcripts and continue scraping AI influencer videos.
Organizes transcripts into separate folders for each influencer.
"""

import os
import sys
import json
import re
import glob
import tempfile
import yt_dlp
import whisper

# Set FFmpeg path
os.environ['PATH'] = r"C:\Users\isha and gaurav\AppData\Roaming\Python\Python314\site-packages\imageio_ffmpeg\binaries;" + os.environ.get('PATH', '')

# Configuration
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

VIDEOS_PER_INFLUENCER = 10
BASE_OUTPUT_DIR = r"c:\Users\isha and gaurav\OneDrive\Desktop\yt-influe\Youtube-engagement-analysis-with-scraper\output\ai_influencers"
OLD_TRANSCRIPTS_FILE = r"c:\Users\isha and gaurav\OneDrive\Desktop\yt-influe\Youtube-engagement-analysis-with-scraper\output\ai_influencers\transcripts_whisper.jsonl"

def sanitize_folder_name(name):
    """Create safe folder name."""
    return re.sub(r'[<>:"/\\|?*]', '', name).strip()

def get_influencer_from_query(query):
    """Extract influencer name from query string."""
    query_lower = query.lower()
    for inf in AI_INFLUENCERS:
        if inf.lower().split()[0] in query_lower:
            return inf
    return None

def load_existing_transcripts(folder_path):
    """Load existing transcripts from a folder's transcripts.jsonl."""
    transcripts_file = os.path.join(folder_path, "transcripts.jsonl")
    existing = {}
    if os.path.exists(transcripts_file):
        with open(transcripts_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    existing[data['video_id']] = data
                except:
                    pass
    return existing

def save_transcript(folder_path, data):
    """Append transcript to folder's transcripts.jsonl."""
    os.makedirs(folder_path, exist_ok=True)
    transcripts_file = os.path.join(folder_path, "transcripts.jsonl")
    with open(transcripts_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def import_old_transcripts():
    """Import old transcripts into influencer folders."""
    if not os.path.exists(OLD_TRANSCRIPTS_FILE):
        print("No old transcripts file found.")
        return {}
    
    imported_counts = {inf: 0 for inf in AI_INFLUENCERS}
    
    print("\n" + "="*60)
    print("IMPORTING PREVIOUS TRANSCRIPTS")
    print("="*60)
    
    with open(OLD_TRANSCRIPTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                query = data.get('query', '')
                influencer = get_influencer_from_query(query)
                
                if not influencer:
                    # Try to match by title
                    title = data.get('title', '').lower()
                    for inf in AI_INFLUENCERS:
                        if inf.lower().split()[0] in title:
                            influencer = inf
                            break
                
                if influencer:
                    folder_name = sanitize_folder_name(influencer)
                    folder_path = os.path.join(BASE_OUTPUT_DIR, folder_name)
                    
                    # Check if already imported
                    existing = load_existing_transcripts(folder_path)
                    if data['video_id'] not in existing:
                        # Update query to match current influencer
                        data['influencer'] = influencer
                        save_transcript(folder_path, data)
                        imported_counts[influencer] += 1
                        print(f"  Imported to {influencer}: {data.get('title', '')[:50]}...")
            except Exception as e:
                continue
    
    print("\nImport Summary:")
    for inf, count in imported_counts.items():
        if count > 0:
            print(f"  {inf}: {count} transcripts imported")
    
    return imported_counts

def search_videos(query, max_results=20):
    """Search YouTube for videos."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            if results and 'entries' in results:
                return results['entries']
    except Exception as e:
        print(f"    Search error: {e}")
    return []

def get_video_info(video_id):
    """Get video duration and metadata."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            return info
    except:
        return None

def transcribe_video(video_id, model):
    """Download and transcribe a video."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128',
            }],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                duration = info.get('duration', 0)
                title = info.get('title', '')
            
            audio_files = glob.glob(os.path.join(temp_dir, '*'))
            if not audio_files:
                return None, None, None
            
            audio_file = audio_files[0]
            print(f"    Transcribing with Whisper... ({os.path.basename(audio_file)})")
            
            result = model.transcribe(audio_file, fp16=False)
            transcript = result.get('text', '').strip()
            
            return transcript, duration, title
            
        except Exception as e:
            print(f"    Error: {e}")
            return None, None, None

def scrape_influencer(influencer, model, existing_video_ids):
    """Scrape videos for one influencer."""
    folder_name = sanitize_folder_name(influencer)
    folder_path = os.path.join(BASE_OUTPUT_DIR, folder_name)
    
    # Load existing transcripts for this influencer
    existing = load_existing_transcripts(folder_path)
    current_count = len(existing)
    all_video_ids = set(existing.keys()) | existing_video_ids
    
    print(f"\n{'='*60}")
    print(f"INFLUENCER: {influencer}")
    print(f"{'='*60}")
    print(f"  Already have: {current_count}/{VIDEOS_PER_INFLUENCER}")
    
    if current_count >= VIDEOS_PER_INFLUENCER:
        print(f"  Already complete!")
        return current_count
    
    # Search queries to try
    search_queries = [
        f"{influencer} virtual influencer interview",
        f"{influencer} AI influencer",
        f"{influencer} interview",
        f"{influencer} virtual model",
    ]
    
    for query in search_queries:
        if current_count >= VIDEOS_PER_INFLUENCER:
            break
            
        print(f"\n  Searching: {query}")
        videos = search_videos(query)
        
        # Filter to 1-10 minute videos
        suitable_videos = []
        for v in videos:
            if v and v.get('id') and v.get('id') not in all_video_ids:
                info = get_video_info(v['id'])
                if info:
                    duration = info.get('duration', 0)
                    if 60 <= duration <= 600:  # 1-10 minutes
                        suitable_videos.append({
                            'id': v['id'],
                            'title': info.get('title', ''),
                            'duration': duration
                        })
        
        print(f"    Found {len(videos)} videos, {len(suitable_videos)} in 1-10 min range")
        
        for video in suitable_videos:
            if current_count >= VIDEOS_PER_INFLUENCER:
                break
            
            video_id = video['id']
            title = video['title']
            duration = video['duration']
            
            print(f"\n  Processing: {title[:50]}... ({duration}s)")
            print(f"    Downloading audio...")
            
            transcript, _, _ = transcribe_video(video_id, model)
            
            if transcript and len(transcript) > 100:
                data = {
                    'video_id': video_id,
                    'title': title,
                    'influencer': influencer,
                    'query': query,
                    'duration': duration,
                    'transcript': transcript,
                    'transcript_length': len(transcript),
                    'method': 'whisper'
                }
                
                save_transcript(folder_path, data)
                all_video_ids.add(video_id)
                current_count += 1
                
                print(f"    SUCCESS! Got {len(transcript)} chars")
                print(f"    Progress: {current_count}/{VIDEOS_PER_INFLUENCER}")
            else:
                print(f"    FAILED - no valid transcript")
    
    return current_count

def main():
    print("="*60)
    print("AI INFLUENCER TRANSCRIPT SCRAPER")
    print("(With Previous Transcript Import)")
    print("="*60)
    
    # Step 1: Import old transcripts
    import_old_transcripts()
    
    # Collect all existing video IDs to avoid duplicates
    all_existing_ids = set()
    if os.path.exists(OLD_TRANSCRIPTS_FILE):
        with open(OLD_TRANSCRIPTS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    all_existing_ids.add(data.get('video_id'))
                except:
                    pass
    
    # Step 2: Load Whisper model
    print("\n" + "="*60)
    print("LOADING WHISPER MODEL (tiny)")
    print("="*60)
    model = whisper.load_model("tiny")
    print("Model loaded!")
    
    # Step 3: Scrape each influencer
    total_transcripts = 0
    
    for influencer in AI_INFLUENCERS:
        count = scrape_influencer(influencer, model, all_existing_ids)
        total_transcripts += count
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    for influencer in AI_INFLUENCERS:
        folder_name = sanitize_folder_name(influencer)
        folder_path = os.path.join(BASE_OUTPUT_DIR, folder_name)
        existing = load_existing_transcripts(folder_path)
        print(f"  {influencer}: {len(existing)}/{VIDEOS_PER_INFLUENCER}")
    
    print(f"\nTotal transcripts: {total_transcripts}")
    print("Done!")

if __name__ == "__main__":
    main()
