#!/usr/bin/env python3
"""
Scrape transcripts for videos about AI influencers using Whisper.
Downloads audio and transcribes locally - no YouTube subtitle API needed.
"""

import os
import sys
import json
import time
import random
import tempfile
import glob
import yt_dlp

# Add FFmpeg to PATH
ffmpeg_path = os.path.join(
    os.path.expanduser("~"),
    "AppData", "Roaming", "Python", "Python314", "site-packages",
    "imageio_ffmpeg", "binaries"
)
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

import whisper

# AI influencers to search for - using more specific queries
AI_INFLUENCERS = [
    "Lu do Magalu virtual influencer",
    "Lil Miquela",
    "Shudu Gram CGI model",
    "Noonoouri virtual influencer",
    "Aitana Lopez AI influencer",
    "Imma virtual model Japan",
    "Rozy virtual influencer Korea",
    "Leya Love virtual influencer",
    "Kyra virtual influencer India",
    "Milla Sofia AI influencer Finland"
]

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "ai_influencers")

# Load Whisper model (tiny is fastest, base is balanced, large is most accurate)
print("Loading Whisper model (tiny)...")
MODEL = whisper.load_model("tiny")  # Options: tiny, base, small, medium, large


def search_videos(query, max_results=10):
    """Search YouTube for videos matching query."""
    search_query = f'ytsearch{max_results}:{query}'
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(search_query, download=False)
            
            videos = []
            if results and 'entries' in results:
                for entry in results.get('entries', []):
                    if entry:
                        videos.append({
                            'video_id': entry.get('id'),
                            'title': entry.get('title'),
                            'duration': entry.get('duration', 0),
                        })
            return videos
    except Exception as e:
        print(f"  Search error: {e}")
        return []


def transcribe_video(video_id, max_duration=600):
    """Download audio and transcribe with Whisper."""
    
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download audio without conversion (no FFmpeg needed)
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio',
                'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }
            
            print(f"    Downloading audio...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                duration = info.get('duration', 0)
                
                # Skip very long videos
                if duration and duration > max_duration:
                    print(f"    Skipping - too long ({duration}s)")
                    return None
            
            # Find the audio file (check multiple extensions)
            audio_file = None
            for ext in ['m4a', 'webm', 'opus', 'mp4', 'mp3', 'wav', 'ogg']:
                potential_file = os.path.join(temp_dir, f'audio.{ext}')
                if os.path.exists(potential_file):
                    audio_file = potential_file
                    break
            
            if not audio_file:
                # List what's in the directory
                files = glob.glob(os.path.join(temp_dir, '*'))
                if files:
                    audio_file = files[0]  # Use whatever was downloaded
                else:
                    print(f"    Audio file not found")
                    return None
            
            print(f"    Transcribing with Whisper... ({os.path.basename(audio_file)})")
            result = MODEL.transcribe(audio_file, fp16=False)
            transcript = result.get('text', '')
            
            return transcript.strip() if transcript else None
            
    except Exception as e:
        error_str = str(e)
        if '429' in error_str:
            print(f"    Rate limited on video download")
        else:
            print(f"    Error: {e}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_file = os.path.join(OUTPUT_DIR, "transcripts_whisper.jsonl")
    
    # Track collected videos
    collected = []
    seen_ids = set()
    target = 100
    
    # Load existing if any
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    collected.append(entry)
                    seen_ids.add(entry.get('video_id'))
        print(f"Loaded {len(collected)} existing transcripts")
    
    print(f"\nTarget: {target} transcripts")
    print(f"Current: {len(collected)}")
    print("\nUsing Whisper for local transcription (bypasses subtitle API).\n")
    
    # Generate search queries
    queries = []
    for influencer in AI_INFLUENCERS:
        queries.extend([
            influencer,
            f"{influencer} interview",
            f"{influencer} explained",
        ])
    
    # Add general AI influencer queries
    queries.extend([
        "AI virtual influencer explained",
        "CGI influencer documentary",
        "virtual Instagram models",
        "AI generated influencer",
        "virtual influencers future marketing",
        "Lil Miquela interview real",
        "fake Instagram influencers CGI",
    ])
    
    random.shuffle(queries)
    
    for query in queries:
        if len(collected) >= target:
            break
            
        print(f"\nSearching: {query}")
        time.sleep(random.uniform(2, 5))
        
        videos = search_videos(query, max_results=12)
        print(f"  Found {len(videos)} videos")
        
        # Filter to reasonable duration (1-10 minutes)
        videos = [v for v in videos if v.get('duration', 0) >= 60 and v.get('duration', 0) <= 600]
        print(f"  After duration filter: {len(videos)} videos (1-10 min)")
        
        for video in videos:
            if len(collected) >= target:
                break
                
            video_id = video.get('video_id')
            if not video_id or video_id in seen_ids:
                continue
            
            seen_ids.add(video_id)
            title = video.get('title', '')
            duration = video.get('duration', 0)
            print(f"\n  Processing: {title[:50]}... ({duration}s)")
            
            # Small delay
            time.sleep(random.uniform(1, 3))
            
            transcript = transcribe_video(video_id)
            
            if transcript and len(transcript) > 100:
                entry = {
                    'video_id': video_id,
                    'title': title,
                    'query': query,
                    'duration': duration,
                    'transcript': transcript,
                    'transcript_length': len(transcript),
                    'method': 'whisper'
                }
                collected.append(entry)
                
                # Save immediately
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                print(f"    SUCCESS! Got {len(transcript)} chars")
                print(f"    Progress: {len(collected)}/{target}")
            else:
                print(f"    No transcript generated")
    
    print(f"\n\nDone! Collected {len(collected)} transcripts")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
