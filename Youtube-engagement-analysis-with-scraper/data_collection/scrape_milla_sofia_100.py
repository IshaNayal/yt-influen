"""Scrape top 100 Milla Sofia AI influencer transcripts."""
import os
import json
import tempfile
import glob
import yt_dlp
import whisper
import time

os.environ["PATH"] = r"C:\Users\isha and gaurav\AppData\Roaming\Python\Python314\site-packages\imageio_ffmpeg\binaries;" + os.environ.get("PATH", "")

OUTPUT_DIR = r"c:\Users\isha and gaurav\OneDrive\Desktop\yt-influe\Youtube-engagement-analysis-with-scraper\output\ai_influencers\Milla Sofia"
INFLUENCER = "Milla Sofia"
TARGET = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_highest_count():
    filepath = os.path.join(OUTPUT_DIR, "milla-sofia-transcripts.jsonl")
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
    filepath = os.path.join(OUTPUT_DIR, "milla-sofia-transcripts.jsonl")
    video_ids = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        parts = line.split(' ', 1)
                        if len(parts) > 1 and parts[0].isdigit():
                            data = json.loads(parts[1])
                            if 'video_id' in data:
                                video_ids.add(data['video_id'])
                    except:
                        pass
    return video_ids

def search_videos(query, max_results=40):
    videos = []
    ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_query = f"ytsearch{max_results}:{query}"
            results = ydl.extract_info(search_query, download=False)
            if results and 'entries' in results:
                for entry in results['entries']:
                    if entry and entry.get('id'):
                        videos.append({'video_id': entry['id'], 'title': entry.get('title', ''), 'channel': entry.get('channel', entry.get('uploader', ''))})
    except:
        pass
    return videos

def is_valid_transcript(transcript):
    if not transcript or len(transcript) < 100:
        return False
    lines = transcript.split('\n')
    if len(lines) > 0:
        word_freq = {}
        for line in lines:
            for word in line.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        for word, count in word_freq.items():
            if len(word) > 2 and count > len(lines) * 0.3:
                return False
    unique_words = len(set(transcript.lower().split()))
    total_words = len(transcript.split())
    if total_words > 0 and unique_words / total_words < 0.15:
        return False
    return True

def transcribe_video(video_id, model):
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {'format': 'bestaudio/best', 'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'), 'quiet': True, 'no_warnings': True, 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}]}
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
    filepath = os.path.join(OUTPUT_DIR, "milla-sofia-transcripts.jsonl")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"{count} {json.dumps(data, ensure_ascii=False)}\n")

def main():
    highest = get_highest_count()
    print(f"\n{'='*70}\nMilla Sofia - Virtual Influencer - Top {TARGET} Transcripts\n{'='*70}\nCurrent: {highest}, Target: {TARGET}\nNeed: {max(0, TARGET - highest)} more\n")
    model = whisper.load_model("tiny")
    existing_ids = get_existing_video_ids()
    print(f"Already have: {len(existing_ids)} videos\n")
    all_videos_dict = {}
    queries = ["Milla Sofia virtual influencer interview", "Milla Sofia digital avatar fashion", "Milla Sofia lifestyle vlog", "Milla Sofia content creator", "Milla Sofia fashion tips", "Milla Sofia makeup tutorial", "Milla Sofia Instagram content", "Milla Sofia social media", "Milla Sofia brand partnership", "Milla Sofia trending", "Milla Sofia photoshoot", "Milla Sofia behind the scenes", "Milla Sofia virtual model", "Milla Sofia collaboration"]
    for q in queries:
        print(f"Searching: {q}...", end=" ", flush=True)
        videos = search_videos(q, max_results=40)
        print(f"Found {len(videos)}")
        for v in videos:
            if v['video_id'] not in all_videos_dict:
                all_videos_dict[v['video_id']] = v
        time.sleep(0.3)
    print(f"\nTotal unique videos: {len(all_videos_dict)}")
    new_videos = [v for v in all_videos_dict.values() if v['video_id'] not in existing_ids]
    print(f"New videos available: {len(new_videos)}\n")
    added, failed = 0, 0
    for idx, video in enumerate(new_videos, 1):
        if highest + added >= TARGET:
            break
        count_num = highest + added + 1
        title = video['title'][:60]
        print(f"[{idx}] #{count_num} {title}...", end=" ", flush=True)
        transcript = transcribe_video(video['video_id'], model)
        if transcript and is_valid_transcript(transcript):
            data = {'video_id': video['video_id'], 'title': video['title'], 'channel': video['channel'], 'influencer': INFLUENCER, 'transcript': transcript}
            save_transcript(count_num, data)
            print(f"✓ ({len(transcript)} chars)")
            added += 1
        else:
            print("✗")
            failed += 1
    final_count = get_highest_count()
    print(f"\n{'='*70}\nAdded: {added} transcripts\nFailed: {failed}\nTotal transcripts: {final_count}\n{'='*70}\n")

if __name__ == "__main__":
    main()
