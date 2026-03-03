"""Scrape 100 transcripts per influencer using faster-whisper large-v2 with rate limit handling."""

import os
import json
import tempfile
import glob
import yt_dlp
from faster_whisper import WhisperModel
import time
import random

OUTPUT_DIR = r"c:\Users\isha and gaurav\OneDrive\Desktop\yt-influe\Youtube-engagement-analysis-with-scraper\output\ai_influencers"
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
VIDEOS_PER_INFLUENCER = 100

def get_existing_transcripts(influencer):
    folder = os.path.join(OUTPUT_DIR, influencer)
    filepath = os.path.join(folder, "transcripts.jsonl")
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

def search_videos(influencer, max_results=120):
    queries = [
        f"ytsearch{max_results//3}:{influencer}",
        f"ytsearch{max_results//3}:{influencer} AI",
        f"ytsearch{max_results//3}:{influencer} virtual",
    ]
    all_videos = {}
    for query in queries:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'socket_timeout': 30,
            'retries': 3,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                results = ydl.extract_info(query, download=False)
                if results and 'entries' in results:
                    for entry in results['entries']:
                        if entry and entry.get('id'):
                            vid_id = entry['id']
                            if vid_id not in all_videos:
                                all_videos[vid_id] = {
                                    'video_id': vid_id,
                                    'title': entry.get('title', '')[:100],
                                    'duration': entry.get('duration', 0),
                                }
            # Delay between search queries to avoid rate limit
            time.sleep(random.uniform(2, 5))
        except Exception as e:
            err_msg = str(e)[:60]
            if "Sign in" not in err_msg and "bot" not in err_msg.lower():
                print(f"  Search error: {err_msg}")
            time.sleep(random.uniform(5, 10))
    return list(all_videos.values())

def transcribe_video(video_id, model):
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
            'retries': 2,
            'nopostprocessors': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
            
            files = glob.glob(os.path.join(temp_dir, '*'))
            if not files:
                return None
            
            audio_file = files[0]
            segments, info = model.transcribe(audio_file, beam_size=1)
            transcript = " ".join([seg.text for seg in segments])
            return transcript if transcript.strip() else None
            
        except Exception as e:
            return None
        finally:
            # Delay after download to avoid rate limit
            time.sleep(random.uniform(1, 3))

def save_transcript(influencer, data):
    folder = os.path.join(OUTPUT_DIR, influencer)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, "transcripts.jsonl")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def main():
    print("Loading faster-whisper large-v2 model...")
    model = WhisperModel("large-v2", device="cpu", compute_type="int8")
    print("Model loaded!\n")
    
    for influencer in AI_INFLUENCERS:
        existing_ids = get_existing_transcripts(influencer)
        current_count = len(existing_ids)
        needed = VIDEOS_PER_INFLUENCER - current_count
        
        print(f"\n{'='*60}")
        print(f"[{influencer}] Have {current_count}/{VIDEOS_PER_INFLUENCER}")
        print(f"{'='*60}")
        
        if needed <= 0:
            print(f"  Already complete!")
            continue
        
        print(f"  Searching for {needed} videos...")
        videos = search_videos(influencer, max_results=min(120, needed * 2))
        
        processed = 0
        for video in videos:
            if processed >= needed:
                break
            
            vid_id = video['video_id']
            if vid_id in existing_ids:
                continue
            
            num = current_count + processed + 1
            title = video['title'][:45]
            print(f"  [{num:3d}/100] {title:45s} ", end="", flush=True)
            
            try:
                transcript = transcribe_video(vid_id, model)
                if transcript and len(transcript) > 20:
                    save_transcript(influencer, {
                        'video_id': vid_id,
                        'title': video['title'],
                        'influencer': influencer,
                        'transcript': transcript,
                        'length': len(transcript),
                    })
                    print("✓")
                    processed += 1
                else:
                    print("✗")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print("✗")
        
        if processed > 0:
            print(f"  → Added {processed} new transcripts\n")

if __name__ == "__main__":
    main()
