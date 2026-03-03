#!/usr/bin/env python3
"""
Scrape transcripts for videos about AI influencers using yt-dlp
with proper rate limiting and delays to avoid 429 errors.
"""

import os
import json
import time
import random
import glob
import re
import tempfile
import yt_dlp

# AI influencers to search for
AI_INFLUENCERS = [
    "Lu do Magalu", "Lil Miquela", "Shudu", "Noonoouri", 
    "Aitana Lopez", "Imma virtual model", "Rozy virtual influencer", 
    "Leya Love AI", "Kyra virtual influencer", "Milla Sofia"
]

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "ai_influencers")

def search_videos(query, max_results=10):
    """Search YouTube for videos matching query."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'default_search': f'ytsearch{max_results}',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(query, download=False)
            
            videos = []
            if results and 'entries' in results:
                for entry in results.get('entries', []):
                    if entry:
                        videos.append({
                            'video_id': entry.get('id'),
                            'title': entry.get('title'),
                            'url': entry.get('url', f"https://www.youtube.com/watch?v={entry.get('id')}")
                        })
            return videos
    except Exception as e:
        print(f"  Search error: {e}")
        return []


def get_subtitles_with_retry(video_id, max_retries=3, base_delay=30):
    """Download subtitles for a video with retry and backoff."""
    
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    for attempt in range(max_retries):
        try:
            # Create temp directory for subtitle files
            with tempfile.TemporaryDirectory() as temp_dir:
                sub_file = os.path.join(temp_dir, 'subs')
                
                ydl_opts = {
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en', 'en-US', 'en-GB'],
                    'subtitlesformat': 'vtt',
                    'skip_download': True,
                    'outtmpl': sub_file,
                    'quiet': True,
                    'no_warnings': True,
                    'ignoreerrors': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                
                # Find any VTT files created
                vtt_files = glob.glob(os.path.join(temp_dir, '*.vtt'))
                
                for vtt_file in vtt_files:
                    with open(vtt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse VTT to text
                    lines = content.split('\n')
                    text_lines = []
                    for line in lines:
                        # Skip VTT headers, timestamps, and empty lines
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith('WEBVTT'):
                            continue
                        if '-->' in line:
                            continue
                        if re.match(r'^\d+$', line):
                            continue
                        if line.startswith('Kind:') or line.startswith('Language:'):
                            continue
                        # Remove HTML tags
                        line = re.sub(r'<[^>]+>', '', line)
                        if line:
                            text_lines.append(line)
                    
                    if text_lines:
                        transcript = ' '.join(text_lines)
                        return transcript
                
                # No subtitles found
                return None
                
        except Exception as e:
            error_str = str(e)
            if '429' in error_str:
                delay = base_delay * (2 ** attempt) + random.uniform(5, 15)
                print(f"    Rate limited, waiting {delay:.0f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"    Error: {e}")
                return None
    
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_file = os.path.join(OUTPUT_DIR, "transcripts_with_delays.jsonl")
    
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
    print("\nNote: Using delays to avoid rate limiting.")
    print("This will take some time but is more reliable.\n")
    
    # Generate search queries
    queries = []
    for influencer in AI_INFLUENCERS:
        queries.extend([
            f"{influencer}",
            f"{influencer} AI influencer",
            f"{influencer} virtual influencer",
            f"{influencer} interview",
            f"{influencer} documentary",
        ])
    
    random.shuffle(queries)
    
    for query in queries:
        if len(collected) >= target:
            break
            
        print(f"\nSearching: {query}")
        time.sleep(random.uniform(2, 5))  # Delay before search
        
        videos = search_videos(query, max_results=15)
        print(f"  Found {len(videos)} videos")
        
        for video in videos:
            if len(collected) >= target:
                break
                
            video_id = video.get('video_id')
            if not video_id or video_id in seen_ids:
                continue
            
            seen_ids.add(video_id)
            title = video.get('title', '')
            print(f"\n  Processing: {title[:60]}...")
            
            # Wait before each subtitle request
            wait_time = random.uniform(10, 20)
            print(f"    Waiting {wait_time:.0f}s before request...")
            time.sleep(wait_time)
            
            transcript = get_subtitles_with_retry(video_id)
            
            if transcript and len(transcript) > 100:
                entry = {
                    'video_id': video_id,
                    'title': title,
                    'query': query,
                    'transcript': transcript,
                    'transcript_length': len(transcript)
                }
                collected.append(entry)
                
                # Save immediately
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                print(f"    SUCCESS! Got {len(transcript)} chars")
                print(f"    Progress: {len(collected)}/{target}")
            else:
                print(f"    No transcript available")
    
    print(f"\n\nDone! Collected {len(collected)} transcripts")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
