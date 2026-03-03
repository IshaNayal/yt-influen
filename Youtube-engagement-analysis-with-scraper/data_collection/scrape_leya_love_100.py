"""Scrape 100 Leya Love video transcripts using Whisper."""

import os
import json
import tempfile
import glob
import yt_dlp
import whisper
import time

os.environ["PATH"] = r"C:\Users\isha and gaurav\AppData\Roaming\Python\Python314\site-packages\imageio_ffmpeg\binaries;" + os.environ.get("PATH", "")

OUTPUT_DIR = r"c:\Users\isha and gaurav\OneDrive\Desktop\yt-influe\Youtube-engagement-analysis-with-scraper\output\ai_influencers\Leya Love"
INFLUENCER = "Leya Love"
TARGET = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_highest_count():
    """Get the highest count number in the file."""
    filepath = os.path.join(OUTPUT_DIR, "leya-love-transcripts.jsonl")
    max_count = 0
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        parts = line.split(' ', 1)
                        if len(parts) > 1 and parts[0].isdigit():
                            count = int(parts[0])
                            if count > max_count:
                                max_count = count
                    except:
                        pass
    return max_count


def get_existing_video_ids():
    """Get video IDs already collected."""
    filepath = os.path.join(OUTPUT_DIR, "leya-love-transcripts.jsonl")
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


def search_videos(query, max_results=50):
    """Search for videos using yt-dlp."""
    videos = []
    ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': True}
    
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
                            'channel': entry.get('channel', entry.get('uploader', '')),
                        })
    except:
        pass
    
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
        except:
            return None


def save_transcript(count, data):
    """Append transcript to file."""
    filepath = os.path.join(OUTPUT_DIR, "leya-love-transcripts.jsonl")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"{count} {json.dumps(data, ensure_ascii=False)}\n")


def main():
    highest = get_highest_count()
    existing_ids = get_existing_video_ids()
    
    print("\n" + "="*70)
    print(f"Leya Love - Scrape {TARGET} Transcripts")
    print("="*70)
    print(f"\nCurrent count: {highest}")
    print(f"Target: {TARGET}")
    print(f"Existing IDs: {len(existing_ids)}\n")
    
    print(f"Loading Whisper tiny model...")
    model = whisper.load_model("tiny")
    
    # Search for videos using multiple queries
    all_videos_dict = {}
    queries = [
        "Leya Love",
        "Leya Love AI",
        "Leya Love influencer",
        "Leya Love interviews",
    ]
    
    print(f"\nSearching for videos...")
    for q in queries:
        print(f"  {q}...", end=" ", flush=True)
        videos = search_videos(q, max_results=50)
        print(f"Found {len(videos)}")
        
        for v in videos:
            if v['video_id'] not in all_videos_dict:
                all_videos_dict[v['video_id']] = v
        
        time.sleep(0.3)
    
    new_videos = [v for v in all_videos_dict.values() if v['video_id'] not in existing_ids]
    print(f"\nTotal unique videos: {len(all_videos_dict)}")
    print(f"New videos to process: {len(new_videos)}\n")
    
    added = 0
    failed = 0
    
    for idx, video in enumerate(new_videos, 1):
        if highest + added >= TARGET:
            break
        
        count_num = highest + added + 1
        title = video['title'][:50]
        
        print(f"[{idx}] #{count_num} {title}...", end=" ", flush=True)
        
        transcript = transcribe_video(video['video_id'], model)
        
        if transcript and len(transcript) > 50:
            data = {
                'video_id': video['video_id'],
                'title': video['title'],
                'channel': video['channel'],
                'influencer': INFLUENCER,
                'transcript': transcript,
            }
            save_transcript(count_num, data)
            print(f"✓ ({len(transcript)} chars)")
            added += 1
        else:
            print("✗ Failed")
            failed += 1
        
        time.sleep(0.5)
    
    final_count = get_highest_count()
    print(f"\n" + "="*70)
    print(f"Added: {added} new transcripts")
    print(f"Failed: {failed}")
    print(f"Total transcripts: {final_count}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
