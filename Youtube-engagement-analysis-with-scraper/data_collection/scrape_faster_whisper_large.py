"""Scrape 100 transcripts per influencer using faster-whisper large-v2 model for best speed and accuracy."""

import os
import json
import tempfile
import glob
import yt_dlp
from faster_whisper import WhisperModel

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
        f"ytsearch{max_results//3}:{influencer} AI influencer",
        f"ytsearch{max_results//3}:{influencer} interview virtual",
    ]
    all_videos = {}
    for query in queries:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
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
                                    'title': entry.get('title', ''),
                                    'duration': entry.get('duration', 0),
                                    'channel': entry.get('channel', entry.get('uploader', '')),
                                }
        except Exception as e:
            print(f"  Search error: {e}")
    return list(all_videos.values())

def transcribe_video(video_id, model):
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
            segments, info = model.transcribe(audio_file, beam_size=1)
            transcript = " ".join([seg.text for seg in segments])
            return transcript
        except Exception as e:
            print(f"    Error: {e}")
            return None

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
        print(f"  Searching for videos...")
        videos = search_videos(influencer, max_results=needed + 30)
        videos = [v for v in videos if v['video_id'] not in existing_ids]
        print(f"  Found {len(videos)} new videos to process")
        collected = 0
        for i, video in enumerate(videos):
            if collected >= needed:
                break
            video_id = video['video_id']
            title = video['title'][:45]
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
                existing_ids.add(video_id)
                print(f"    -> {len(transcript)} chars")
                collected += 1
            else:
                print(f"    -> skipped (no transcript)")
        print(f"  Done: {current_count + collected}/{VIDEOS_PER_INFLUENCER}")
    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
