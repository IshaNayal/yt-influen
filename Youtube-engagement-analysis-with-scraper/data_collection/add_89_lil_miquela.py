"""Add 89 more Lil Miquela transcripts using Whisper with yt-dlp."""

import os
import json
import tempfile
import glob
import yt_dlp
import whisper
import time

# Set FFmpeg path
os.environ["PATH"] = r"C:\Users\isha and gaurav\AppData\Roaming\Python\Python314\site-packages\imageio_ffmpeg\binaries;" + os.environ.get("PATH", "")

OUTPUT_DIR = r"c:\Users\isha and gaurav\OneDrive\Desktop\yt-influe\Youtube-engagement-analysis-with-scraper\output\ai_influencers\Lil Miquela"
INFLUENCER = "Lil Miquela"
STARTING_COUNT = 15
TARGET_NEW_TRANSCRIPTS = 89

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_existing_video_ids():
    """Get video IDs already collected."""
    filepath = os.path.join(OUTPUT_DIR, "lil-miquela-transcripts.jsonl")
    video_ids = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        parts = line.split(' ', 1)
                        if len(parts) > 1 and parts[0].isdigit():
                            json_str = parts[1]
                        else:
                            json_str = line
                        data = json.loads(json_str)
                        if 'video_id' in data:
                            video_ids.add(data['video_id'])
                    except:
                        pass
    return video_ids


def search_videos(query, max_results=80):
    """Search for videos using yt-dlp."""
    videos = []
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
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
                ydl.extract_info(url, download=True)
            
            files = glob.glob(os.path.join(temp_dir, '*'))
            if not files:
                return None
            
            audio_file = files[0]
            result = model.transcribe(audio_file, fp16=False)
            return result.get('text', '')
        except Exception as e:
            return None


def save_transcript(count, data):
    """Append transcript to file with count prefix."""
    filepath = os.path.join(OUTPUT_DIR, "lil-miquela-transcripts.jsonl")
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"{count} {json.dumps(data, ensure_ascii=False)}\n")


def main():
    print("\n" + "="*70)
    print(f"Lil Miquela - Adding {TARGET_NEW_TRANSCRIPTS} More Transcripts (Whisper)")
    print("="*70)
    
    print(f"\nCurrent count: {STARTING_COUNT}")
    print(f"Target final count: {STARTING_COUNT + TARGET_NEW_TRANSCRIPTS}")
    
    print(f"\nLoading Whisper tiny model...")
    model = whisper.load_model("tiny")
    
    existing_ids = get_existing_video_ids()
    print(f"Will skip {len(existing_ids)} already-collected videos\n")
    
    # Search with multiple queries
    all_videos = []
    seen_ids = set()
    
    search_queries = [
        "Lil Miquela",
        "Lil Miquela AI",
        "Lil Miquela influencer",
        "Miquela virtual model",
        "Lil Miquela interview",
    ]
    
    for query in search_queries:
        print(f"Searching: '{query}'...")
        videos = search_videos(query, max_results=60)
        print(f"  Found {len(videos)} videos")
        
        for v in videos:
            if v['video_id'] not in seen_ids:
                all_videos.append(v)
                seen_ids.add(v['video_id'])
        
        time.sleep(1)
    
    print(f"\nTotal unique videos: {len(all_videos)}")
    
    # Filter out already collected
    new_videos = [v for v in all_videos if v['video_id'] not in existing_ids]
    print(f"New videos to process: {len(new_videos)}\n")
    
    collected = 0
    for i, video in enumerate(new_videos, 1):
        if collected >= TARGET_NEW_TRANSCRIPTS:
            break
        
        video_id = video['video_id']
        title = video['title'][:60]
        count_num = STARTING_COUNT + collected + 1
        
        print(f"[{collected + 1}/{TARGET_NEW_TRANSCRIPTS}] #{count_num} {title}...", end=" ", flush=True)
        
        transcript = transcribe_video(video_id, model)
        
        if transcript and len(transcript) > 50:
            data = {
                'video_id': video_id,
                'title': video['title'],
                'channel': video['channel'],
                'influencer': INFLUENCER,
                'transcript': transcript,
            }
            save_transcript(count_num, data)
            print(f"✓ ({len(transcript)} chars)")
            collected += 1
        else:
            print("✗ No transcript")
        
        time.sleep(0.5)
    
    print(f"\n" + "="*70)
    print(f"COMPLETE!")
    print(f"Added: {collected} new transcripts")
    print(f"Total transcripts: {STARTING_COUNT + collected}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
