"""Scrape top 100 Rozy AI influencer transcripts."""

import os
import json
import tempfile
import glob
import yt_dlp
import whisper
import time

os.environ["PATH"] = r"C:\Users\isha and gaurav\AppData\Roaming\Python\Python314\site-packages\imageio_ffmpeg\binaries;" + os.environ.get("PATH", "")

OUTPUT_DIR = r"c:\Users\isha and gaurav\OneDrive\Desktop\yt-influe\Youtube-engagement-analysis-with-scraper\output\ai_influencers\Rozy"
INFLUENCER = "Rozy"
TARGET = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_highest_count():
    """Get the highest count number in the file."""
    filepath = os.path.join(OUTPUT_DIR, "rozy-transcripts.jsonl")
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
    filepath = os.path.join(OUTPUT_DIR, "rozy-transcripts.jsonl")
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


def is_valid_transcript(transcript):
    """Check if transcript is actual spoken content, not music/lyrics."""
    if not transcript or len(transcript) < 100:
        return False
    
    # Check for excessive repetition (sign of music/DJ set)
    lines = transcript.split('\n')
    if len(lines) > 0:
        word_freq = {}
        for line in lines:
            for word in line.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # If any single word appears too many times, likely music
        for word, count in word_freq.items():
            if len(word) > 2 and count > len(lines) * 0.3:  # 30% of lines
                return False
    
    # Check for meaningful content (should have some variation)
    unique_words = len(set(transcript.lower().split()))
    total_words = len(transcript.split())
    
    # Less than 15% unique words = probably repetitive music
    if total_words > 0 and unique_words / total_words < 0.15:
        return False
    
    return True


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
    filepath = os.path.join(OUTPUT_DIR, "rozy-transcripts.jsonl")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"{count} {json.dumps(data, ensure_ascii=False)}\n")


def main():
    highest = get_highest_count()
    print("\n" + "="*70)
    print(f"Rozy AI - Scraping Top {TARGET} Transcripts")
    print("="*70)
    print(f"\nCurrent: {highest}, Target: {TARGET}")
    print(f"Need: {max(0, TARGET - highest)} more\n")
    
    print(f"Loading Whisper tiny model...")
    model = whisper.load_model("tiny")
    
    existing_ids = get_existing_video_ids()
    print(f"Already have: {len(existing_ids)} videos\n")
    
    # Search for Rozy - South Korean Virtual Influencer, First Virtual Influencer from Korea, Fashion, Lifestyle, Glamour, Shopping
    all_videos_dict = {}
    queries = [
        "Rozy first virtual influencer Korea",
        "Rozy Sidus Studio X interview",
        "Rozy fashion week",
        "Rozy luxury lifestyle",
        "Rozy shopping haul",
        "Rozy makeup tutorial",
        "Rozy travel vlog",
        "Rozy interview virtual",
        "Rozy brand collaboration",
        "Rozy facial expressions GamSeong",
        "Rozy glamorous poses",
        "Rozy fashion show",
        "Rozy social media influencer",
        "Rozy digital avatar fashion",
        "Rozy lifestyle content",
    ]
    
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
    
    # Process videos
    added = 0
    failed = 0
    
    for idx, video in enumerate(new_videos, 1):
        if highest + added >= TARGET:
            break
        
        count_num = highest + added + 1
        title = video['title'][:60]
        print(f"[{idx}] #{count_num} {title}...", end=" ", flush=True)
        
        transcript = transcribe_video(video['video_id'], model)
        
        if transcript and is_valid_transcript(transcript):
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
            print("✗")
            failed += 1
    
    final_count = get_highest_count()
    print(f"\n" + "="*70)
    print(f"Added: {added} transcripts")
    print(f"Failed: {failed}")
    print(f"Total transcripts: {final_count}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
