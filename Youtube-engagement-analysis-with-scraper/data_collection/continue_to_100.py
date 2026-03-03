"""Continue scraping until we have 100 total transcripts using Whisper tiny model."""

import os
import json
import tempfile
import glob
import yt_dlp
import whisper

# Set FFmpeg path
os.environ["PATH"] = r"C:\Users\isha and gaurav\AppData\Roaming\Python\Python314\site-packages\imageio_ffmpeg\binaries;" + os.environ.get("PATH", "")

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

TARGET_TOTAL = 100
VIDEOS_PER_INFLUENCER = 10

def count_existing_transcripts():
    """Count all existing transcripts across all influencer folders."""
    total = 0
    counts = {}
    for influencer in AI_INFLUENCERS:
        folder = os.path.join(OUTPUT_DIR, influencer)
        filepath = os.path.join(folder, "transcripts.jsonl")
        count = 0
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)
        counts[influencer] = count
        total += count
    return total, counts

def get_existing_video_ids(influencer):
    """Get video IDs already collected for an influencer."""
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

def search_videos(influencer, max_results=15):
    """Search for videos about an AI influencer."""
    query = f"ytsearch{max_results}:{influencer} AI influencer"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    videos = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(query, download=False)
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
            print(f"Transcription error: {e}")
            return None

def save_transcript(influencer, data):
    """Append transcript to influencer's file."""
    folder = os.path.join(OUTPUT_DIR, influencer)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, "transcripts.jsonl")
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def main():
    print("Loading Whisper tiny model...")
    model = whisper.load_model("tiny")
    
    total, counts = count_existing_transcripts()
    print(f"\nCurrent progress: {total}/{TARGET_TOTAL} transcripts")
    for inf, cnt in counts.items():
        if cnt > 0:
            print(f"  {inf}: {cnt}")
    
    if total >= TARGET_TOTAL:
        print(f"\nAlready have {total} transcripts!")
        return
    
    print(f"\nNeed {TARGET_TOTAL - total} more transcripts\n")
    
    for influencer in AI_INFLUENCERS:
        if total >= TARGET_TOTAL:
            break
        
        current_count = counts.get(influencer, 0)
        needed = VIDEOS_PER_INFLUENCER - current_count
        
        if needed <= 0:
            print(f"[{influencer}] Already has {current_count} transcripts, skipping")
            continue
        
        print(f"\n{'='*50}")
        print(f"[{influencer}] Have {current_count}, need {needed} more")
        print(f"{'='*50}")
        
        existing_ids = get_existing_video_ids(influencer)
        videos = search_videos(influencer, max_results=needed + 5)
        
        # Filter out already collected videos
        videos = [v for v in videos if v['video_id'] not in existing_ids]
        
        collected = 0
        for video in videos:
            if collected >= needed or total >= TARGET_TOTAL:
                break
            
            video_id = video['video_id']
            title = video['title'][:50]
            
            print(f"  [{current_count + collected + 1}/{VIDEOS_PER_INFLUENCER}] {title}...")
            
            transcript = transcribe_video(video_id, model)
            
            if transcript and len(transcript) > 50:
                data = {
                    'video_id': video_id,
                    'title': video['title'],
                    'channel': video['channel'],
                    'influencer': influencer,
                    'transcript': transcript,
                }
                save_transcript(influencer, data)
                print(f"    -> Got {len(transcript)} chars")
                collected += 1
                total += 1
            else:
                print(f"    -> No transcript")
        
        counts[influencer] = current_count + collected
    
    print(f"\n{'='*50}")
    print(f"DONE! Total transcripts: {total}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
