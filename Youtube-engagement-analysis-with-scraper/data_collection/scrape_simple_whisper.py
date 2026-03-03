"""Scrape and transcribe YouTube videos using OpenAI Whisper (base model) - no FFmpeg post-processing."""

import os
import json
import tempfile
import glob
import yt_dlp
import whisper
import time

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
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'video_id' in data:
                                video_ids.add(data['video_id'])
                        except:
                            pass
        except:
            pass
    return video_ids

def search_videos(influencer, max_results=100):
    query = f"ytsearch{max_results}:{influencer}"
    all_videos = {}
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'socket_timeout': 30,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(query, download=False)
            if results and 'entries' in results:
                for entry in results['entries']:
                    if entry and entry.get('id'):
                        vid_id = entry['id']
                        all_videos[vid_id] = {
                            'video_id': vid_id,
                            'title': entry.get('title', '')[:100],
                            'duration': entry.get('duration', 0),
                        }
    except Exception as e:
        print(f"  Search error: {str(e)[:80]}")
    return list(all_videos.values())

def download_audio(video_id):
    """Download audio without requiring FFmpeg post-processing."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            'format': 'bestaudio',
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'nopostprocessors': True,
            'retries': 2,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
            
            # Find the downloaded file
            for file in glob.glob(os.path.join(temp_dir, '*')):
                if os.path.isfile(file) and not file.endswith('.json'):
                    return file
        except Exception as e:
            if "not available" not in str(e).lower():
                print(f"      Download error: {str(e)[:60]}")
    return None

def transcribe_audio(audio_file, model):
    """Transcribe audio file to text."""
    try:
        result = model.transcribe(audio_file, fp16=False, verbose=False)
        transcript = result.get('text', '').strip()
        return transcript if transcript else None
    except Exception as e:
        return None

def save_transcript(influencer, data):
    folder = os.path.join(OUTPUT_DIR, influencer)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, "transcripts.jsonl")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def main():
    print("Loading Whisper base model...")
    model = whisper.load_model("base")
    print("Model ready. Starting transcription...\n")
    
    for influencer in AI_INFLUENCERS:
        existing_ids = get_existing_transcripts(influencer)
        current_count = len(existing_ids)
        needed = VIDEOS_PER_INFLUENCER - current_count
        
        if needed <= 0:
            continue
        
        print(f"[{influencer}] {current_count}/100 - Searching videos...")
        videos = search_videos(influencer, max_results=needed * 3)
        
        processed = 0
        for video in videos:
            if processed >= needed:
                break
            
            vid_id = video['video_id']
            if vid_id in existing_ids:
                continue
            
            num = current_count + processed + 1
            title_short = video['title'][:40]
            print(f"  [{num:3d}/100] {title_short:40s} ", end="", flush=True)
            
            try:
                audio_file = download_audio(vid_id)
                if audio_file:
                    transcript = transcribe_audio(audio_file, model)
                    if transcript and len(transcript) > 10:
                        save_transcript(influencer, {
                            'video_id': vid_id,
                            'title': video['title'],
                            'influencer': influencer,
                            'transcript': transcript,
                            'length': len(transcript),
                        })
                        print("✓ DONE")
                        processed += 1
                    else:
                        print("✗ no speech")
                else:
                    print("✗ download failed")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"✗ error")
        
        if processed > 0:
            print(f"  → Added {processed} transcripts\n")

if __name__ == "__main__":
    main()
