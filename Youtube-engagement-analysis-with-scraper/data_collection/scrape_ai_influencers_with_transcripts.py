"""
Script to scrape transcripts of top 100 RECENT videos WITH TRANSCRIPTS from AI influencers.
Only collects videos that have available transcripts.
"""

import os
import sys
import json
import time
from typing import List, Dict, Optional, Set
from dotenv import load_dotenv
from tqdm import tqdm
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


# AI Influencers to scrape with specific search terms
AI_INFLUENCERS = [
    "Lu do Magalu",           # Brazilian virtual influencer
    "Lil Miquela",            # Most famous AI influencer
    "Shudu Gram",             # Digital supermodel
    "Noonoouri",              # Virtual fashion influencer
    "Aitana Lopez AI",        # Spanish AI model
    "Imma virtual model",     # Japanese virtual model
    "Rozy virtual",           # Korean virtual influencer
    "Leya Love AI",           # AI influencer
    "Kyra virtual influencer", # Indian virtual influencer
    "Milla Sofia AI"          # Finnish AI influencer
]

# Target: 100 videos with transcripts per influencer
TARGET_TRANSCRIPTS_PER_INFLUENCER = 100


def get_youtube_client(api_key: str):
    """Initialize YouTube Data API v3 client."""
    return build('youtube', 'v3', developerKey=api_key)


def search_channel_by_name(youtube, channel_name: str) -> Optional[Dict]:
    """Search for a channel by name and return channel info."""
    try:
        request = youtube.search().list(
            part='snippet',
            q=channel_name,
            type='channel',
            maxResults=5
        )
        response = request.execute()
        
        if response.get('items'):
            print(f"\nSearching for '{channel_name}':")
            for i, item in enumerate(response['items'][:3], 1):
                channel_id = item['snippet']['channelId']
                channel_title = item['snippet']['title']
                print(f"  {i}. {channel_title} (ID: {channel_id})")
            
            first_result = response['items'][0]
            return {
                'search_term': channel_name,
                'channel_id': first_result['snippet']['channelId'],
                'channel_title': first_result['snippet']['title']
            }
        return None
            
    except HttpError as e:
        print(f"Error searching for '{channel_name}': {e}")
        return None


def get_uploads_playlist_id(youtube, channel_id: str) -> Optional[str]:
    """Get the uploads playlist ID for a channel."""
    try:
        request = youtube.channels().list(
            part='contentDetails',
            id=channel_id
        )
        response = request.execute()
        
        if response.get('items'):
            return response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        return None
    except HttpError:
        return None


def get_recent_video_ids(youtube, playlist_id: str, max_videos: int = 500) -> List[str]:
    """Get recent video IDs from a uploads playlist (most recent first)."""
    video_ids = []
    next_page_token = None
    
    try:
        while len(video_ids) < max_videos:
            request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response.get('items', []):
                video_ids.append(item['contentDetails']['videoId'])
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
        return video_ids[:max_videos]
        
    except HttpError as e:
        print(f"Error getting videos: {e}")
        return video_ids


def check_transcript_available(video_id: str) -> Optional[Dict]:
    """
    Check if transcript is available and fetch it.
    Returns transcript data if available, None otherwise.
    """
    try:
        api = YouTubeTranscriptApi()
        
        # Try to list available transcripts
        transcript_list = api.list(video_id)
        
        # Get any available transcript
        for transcript in transcript_list:
            try:
                segments = transcript.fetch()
                
                # Format segments
                formatted_segments = []
                for seg in segments:
                    formatted_segments.append({
                        'start': seg.start if hasattr(seg, 'start') else seg.get('start', 0),
                        'duration': seg.duration if hasattr(seg, 'duration') else seg.get('duration', 0),
                        'text': seg.text if hasattr(seg, 'text') else seg.get('text', '')
                    })
                
                return {
                    'video_id': video_id,
                    'transcript_source': 'youtube_captions',
                    'language': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'segments': formatted_segments
                }
            except:
                continue
        
        return None
        
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception:
        return None


def get_video_metadata(youtube, video_id: str) -> Optional[Dict]:
    """Get metadata for a single video."""
    try:
        request = youtube.videos().list(
            part='snippet,contentDetails,statistics',
            id=video_id
        )
        response = request.execute()
        
        if response.get('items'):
            item = response['items'][0]
            snippet = item['snippet']
            statistics = item.get('statistics', {})
            
            return {
                'video_id': video_id,
                'title': snippet['title'],
                'channel_id': snippet['channelId'],
                'channel_title': snippet['channelTitle'],
                'published_at': snippet['publishedAt'],
                'duration': item['contentDetails']['duration'],
                'view_count': int(statistics.get('viewCount', 0)),
                'like_count': int(statistics.get('likeCount', 0)),
                'comment_count': int(statistics.get('commentCount', 0))
            }
        return None
    except HttpError:
        return None


def load_processed_videos(filepath: str) -> Set[str]:
    """Load already processed video IDs from JSONL file."""
    video_ids = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    video_ids.add(data['video_id'])
                except:
                    pass
    return video_ids


def append_jsonl(filepath: str, data: dict):
    """Append a record to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def main():
    """Main entry point."""
    
    # Get API key
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print("Error: YOUTUBE_API_KEY not found in environment")
        return
    
    # Initialize YouTube client
    youtube = get_youtube_client(api_key)
    
    # Output paths
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'ai_influencers')
    os.makedirs(output_dir, exist_ok=True)
    
    channels_file = os.path.join(output_dir, 'channels.jsonl')
    videos_file = os.path.join(output_dir, 'videos_with_transcripts.jsonl')
    transcripts_file = os.path.join(output_dir, 'transcripts_only.jsonl')
    
    # Load existing processed data
    processed_transcripts = load_processed_videos(transcripts_file)
    
    print("=" * 70)
    print("AI Influencers - Top 100 Recent Videos WITH Transcripts")
    print("=" * 70)
    
    # Step 1: Find channel IDs
    print("\n--- STEP 1: Finding Channel IDs ---\n")
    
    channels_found = []
    for influencer in AI_INFLUENCERS:
        channel_info = search_channel_by_name(youtube, influencer)
        if channel_info:
            channels_found.append(channel_info)
            append_jsonl(channels_file, channel_info)
        time.sleep(0.3)
    
    print(f"\nFound {len(channels_found)} channels")
    
    if not channels_found:
        print("No channels found. Exiting.")
        return
    
    # Step 2: For each channel, find recent videos WITH transcripts
    print("\n--- STEP 2: Finding Recent Videos WITH Transcripts ---\n")
    
    total_success = 0
    total_failed = 0
    
    for channel in channels_found:
        print(f"\n{'='*60}")
        print(f"Processing: {channel['channel_title']} ({channel['search_term']})")
        print(f"{'='*60}")
        
        # Get uploads playlist
        uploads_playlist = get_uploads_playlist_id(youtube, channel['channel_id'])
        if not uploads_playlist:
            print("  Could not find uploads playlist")
            continue
        
        # Get recent video IDs (up to 500 to find 100 with transcripts)
        print("  Fetching recent videos...")
        video_ids = get_recent_video_ids(youtube, uploads_playlist, max_videos=500)
        print(f"  Found {len(video_ids)} recent videos")
        
        # Filter to videos not already processed
        video_ids = [vid for vid in video_ids if vid not in processed_transcripts]
        
        # Find videos with transcripts
        transcripts_found = 0
        videos_checked = 0
        
        pbar = tqdm(video_ids, desc=f"  Checking transcripts", leave=True)
        
        for video_id in pbar:
            videos_checked += 1
            
            # Check if transcript is available
            transcript = check_transcript_available(video_id)
            
            if transcript:
                # Get video metadata
                metadata = get_video_metadata(youtube, video_id)
                
                if metadata:
                    # Add influencer info
                    metadata['influencer_name'] = channel['search_term']
                    transcript['influencer_name'] = channel['search_term']
                    transcript['channel_title'] = channel['channel_title']
                    transcript['video_title'] = metadata['title']
                    
                    # Save both
                    append_jsonl(videos_file, metadata)
                    append_jsonl(transcripts_file, transcript)
                    
                    transcripts_found += 1
                    total_success += 1
                    processed_transcripts.add(video_id)
                    
                    pbar.set_description(f"  Found {transcripts_found}/{TARGET_TRANSCRIPTS_PER_INFLUENCER}")
                    
                    # Stop if we have enough
                    if transcripts_found >= TARGET_TRANSCRIPTS_PER_INFLUENCER:
                        break
            else:
                total_failed += 1
            
            # Rate limiting
            time.sleep(0.2)
        
        print(f"  ✓ Found {transcripts_found} videos with transcripts (checked {videos_checked})")
    
    # Summary
    print("\n" + "=" * 70)
    print("SCRAPING COMPLETE")
    print("=" * 70)
    print(f"\nChannels processed: {len(channels_found)}")
    print(f"Videos with transcripts: {total_success}")
    print(f"Videos without transcripts: {total_failed}")
    print(f"\nOutput files saved to: {output_dir}")
    print("  - channels.jsonl")
    print("  - videos_with_transcripts.jsonl")
    print("  - transcripts_only.jsonl")


if __name__ == '__main__':
    main()
