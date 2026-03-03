"""Simple Whisper flow for scraping AI influencer transcripts - focused on speed."""

import os
import json
import tempfile
import glob
import yt_dlp
import whisper

OUTPUT_DIR = r"c:\Users\isha and gaurav\OneDrive\Desktop\yt-influe\Youtube-engagement-analysis-with-scraper\output\ai_influencers"
AI_INFLUENCERS = ["Lu do Magalu", "Lil Miquela", "Shudu Gram", "Noonoouri", "Aitana Lopez", "Imma", "Rozy", "Leya Love", "Kyra", "Milla Sofia"]
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

def search_videos(influencer, count=150):
    """Search for videos using yt-dlp."""
    query = f"ytsearch{count}:{influencer}"
    videos = []
    ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': False}
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
    except Exception as e:
        print(f"  Search error: {str(e)[:60]}")
    return videos

def download_and_transcribe(video_id, model):
    """Download audio and transcribe in one go."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                'format': 'bestaudio',
                'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'socket_timeout': 30,
                'nopostprocessors': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
            
            # Find audio file
            audio_file = None
            for f in glob.glob(os.path.join(temp_dir, '*')):
                if os.path.isfile(f):
                    audio_file = f
                    break
            
            if not audio_file:
                return None
            
            # Transcribe
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
        
        print(f"\n[{influencer}] {count}/100 - Searching videos...")
        videos = search_videos(influencer, count=needed * 2)
        
        added = 0
        for video in videos:
            if added >= needed:
                break
            
            vid_id = video['id']
            if vid_id in existing:
                continue
            
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
            except:
                print("✗")
        
        print(f"  → Added {added}/{needed}")

if __name__ == "__main__":
    main()
