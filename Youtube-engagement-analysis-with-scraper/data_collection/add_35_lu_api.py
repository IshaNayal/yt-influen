"""Add 35 more Lu do Magalu transcripts using YouTube API captions."""

import os
import json
import time
from typing import Optional, Set, List
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

# Load environment variables
load_dotenv()

OUTPUT_DIR = r"c:\Users\isha and gaurav\OneDrive\Desktop\yt-influe\Youtube-engagement-analysis-with-scraper\output\ai_influencers\Lu do Magalu"
INFLUENCER = "Lu do Magalu"
STARTING_COUNT = 73
TARGET_NEW_TRANSCRIPTS = 35

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_youtube_client(api_key: str):
    """Initialize YouTube Data API v3 client."""
    return build('youtube', 'v3', developerKey=api_key)


def search_videos(youtube, query: str, max_results: int = 50) -> List[dict]:
    """Search for videos using YouTube search API."""
    videos = []
    try:
        request = youtube.search().list(
            part='snippet',
            q=query,
            type='video',
            maxResults=min(max_results, 50),
            order='relevance',
            safeSearch='none'
        )
        response = request.execute()
        
        for item in response.get('items', []):
            if item['id'].get('videoId'):
                videos.append({
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'channel': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt']
                })
    except HttpError as e:
        print(f"Search error: {e}")
    
    return videos


def get_transcript(video_id: str) -> Optional[str]:
    """Get transcript from YouTube video."""
    try:
        # Get available transcripts for the video
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get transcript in preferred languages
        transcript = None
        for lang in ['en', 'pt', 'pt-BR']:
            try:
                transcript = transcript_list.find_transcript([lang])
                break
            except:
                continue
        
        # Fallback to first available transcript
        if not transcript:
            transcript = transcript_list.find_transcript(transcript_list.available_transcripts[0].language)
        
        transcript_data = transcript.fetch()
        transcript_text = ' '.join([item['text'] for item in transcript_data])
        return transcript_text if len(transcript_text) > 50 else None
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception as e:
        return None


def get_existing_video_ids() -> Set[str]:
    """Get video IDs already collected."""
    filepath = os.path.join(OUTPUT_DIR, "lu-do-magalu-transcripts.jsonl")
    video_ids = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        # Skip count number at start if present
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


def save_transcript(count: int, data: dict):
    """Append transcript to file with count prefix."""
    filepath = os.path.join(OUTPUT_DIR, "lu-do-magalu-transcripts.jsonl")
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"{count} {json.dumps(data, ensure_ascii=False)}\n")


def main():
    print("\n" + "="*70)
    print(f"Lu do Magalu - Adding {TARGET_NEW_TRANSCRIPTS} More Transcripts (YouTube API)")
    print("="*70)
    
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print("Error: YOUTUBE_API_KEY not found in .env file")
        return
    
    print(f"\nCurrent count: {STARTING_COUNT}")
    print(f"Target final count: {STARTING_COUNT + TARGET_NEW_TRANSCRIPTS}")
    
    youtube = get_youtube_client(api_key)
    existing_ids = get_existing_video_ids()
    print(f"Will skip {len(existing_ids)} already-collected videos\n")
    
    # Search for videos with different queries to maximize results
    search_queries = [
        "Lu do Magalu",
        "Lu Magalu influencer",
        "Magalu virtual influencer",
        "Lu do Magalu interview",
        "Magalu AI",
        "Magalu influencer digital"
    ]
    
    all_videos = []
    seen_ids = set()
    
    for query in search_queries:
        print(f"Searching: '{query}'...")
        videos = search_videos(youtube, query, max_results=50)
        
        for v in videos:
            if v['video_id'] not in seen_ids:
                all_videos.append(v)
                seen_ids.add(v['video_id'])
        
        time.sleep(0.5)  # Rate limiting
    
    print(f"\nFound {len(all_videos)} unique videos")
    
    # Filter out already collected
    new_videos = [v for v in all_videos if v['video_id'] not in existing_ids]
    print(f"New videos to process: {len(new_videos)}\n")
    
    collected = 0
    for i, video in enumerate(new_videos, 1):
        if collected >= TARGET_NEW_TRANSCRIPTS:
            break
        
        video_id = video['video_id']
        title = video['title'][:60]
        count_num = STARTING_COUNT + collected + 1
        
        print(f"[{collected + 1}/{TARGET_NEW_TRANSCRIPTS}] #{count_num} {title}...", end=" ")
        
        transcript = get_transcript(video_id)
        
        if transcript:
            data = {
                'video_id': video_id,
                'title': video['title'],
                'channel': video['channel'],
                'influencer': INFLUENCER,
                'transcript': transcript,
            }
            save_transcript(count_num, data)
            print(f"✓ ({len(transcript)} chars)")
            collected += 1
        else:
            print("✗ No transcript")
        
        time.sleep(0.3)  # Rate limiting
    
    print(f"\n" + "="*70)
    print(f"COMPLETE!")
    print(f"Added: {collected} new transcripts")
    print(f"Total transcripts: {STARTING_COUNT + collected}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
