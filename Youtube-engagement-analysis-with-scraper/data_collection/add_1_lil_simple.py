"""Add 1 more Lil Miquela transcript - simple robust version."""

import os
import json
import tempfile
import glob
import yt_dlp
import whisper
import time

os.environ["PATH"] = r"C:\Users\isha and gaurav\AppData\Roaming\Python\Python314\site-packages\imageio_ffmpeg\binaries;" + os.environ.get("PATH", "")

OUTPUT_DIR = r"c:\Users\isha and gaurav\OneDrive\Desktop\yt-influe\Youtube-engagement-analysis-with-scraper\output\ai_influencers\Lil Miquela"
INFLUENCER = "Lil Miquela"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_highest_count():
    """Get the highest count number in the file."""
    filepath = os.path.join(OUTPUT_DIR, "lil-miquela-transcripts.jsonl")
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


def search_videos(query, max_results=30):
    """Search for videos using yt-dlp with better error handling."""
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
    except Exception as e:
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
        except Exception as e:
            return None


def save_transcript(count, data):
    """Append transcript to file."""
    filepath = os.path.join(OUTPUT_DIR, "lil-miquela-transcripts.jsonl")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"{count} {json.dumps(data, ensure_ascii=False)}\n")


def main():
    highest = get_highest_count()
    print(f"\nCurrent highest count: {highest}")
    print(f"Loading Whisper tiny model...")
    model = whisper.load_model("tiny")
    
    existing_ids = get_existing_video_ids()
    print(f"Checking {len(existing_ids)} existing videos\n")
    
    # Search for new videos with multiple queries
    all_videos_dict = {}
    queries = ["Lil Miquela", "Lil Miquela AI", "Miquela influencer", "Lil Miquela interviews"]
    
    for q in queries:
        print(f"Searching: {q}...", end=" ", flush=True)
        videos = search_videos(q, max_results=40)
        print(f"Found: {len(videos)}")
        for v in videos:
            if v['video_id'] not in all_videos_dict:
                all_videos_dict[v['video_id']] = v
        time.sleep(0.3)
    
    print(f"\nTotal unique videos found: {len(all_videos_dict)}")
    
    new_videos = [v for v in all_videos_dict.values() if v['video_id'] not in existing_ids]
    print(f"New videos: {len(new_videos)}\n")
    
    if new_videos:
        for idx, video in enumerate(new_videos, 1):
            count_num = highest + idx
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
                break  # Stop after 1
            else:
                print("✗ No transcript")
    
    final_count = get_highest_count()
    print(f"\nFinal count: {final_count}")


if __name__ == "__main__":
    main()
