"""
Script to scrape transcripts of top 100 RECENT videos WITH TRANSCRIPTS from AI influencers.
Uses pre-discovered channel IDs to avoid YouTube Data API search quota.
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
try:
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable
    )
except ImportError:
    TranscriptsDisabled = Exception
    NoTranscriptFound = Exception
    VideoUnavailable = Exception

# Load environment variables
load_dotenv()


# Pre-discovered AI Influencer channels (from previous runs)
AI_INFLUENCER_CHANNELS = [
    {
        'influencer_name': 'Lu do Magalu',
        'channel_id': 'UCeaQ72LrN6K3f9a8JkFV98w',
        'channel_title': 'Canal da Lu - Magalu'
    },
    {
        'influencer_name': 'Lil Miquela',
        'channel_id': 'UCWeHb_SrtJbrT8VD-_QQpRA',
        'channel_title': 'Miquela'
    },
    {
        'influencer_name': 'Noonoouri',
        'channel_id': 'UCmIj9ZSb2QAurwfD3g2G2eQ',
        'channel_title': 'noonoouri'
    },
    {
        'influencer_name': 'Aitana Lopez',
        'channel_id': 'UCx_Z4tFuH4llsgU7N197cUQ',
        'channel_title': 'Aitana López'
    },
    {
        'influencer_name': 'Imma',
        'channel_id': 'UCaAPSltuomXWut-skKB2WHg',
        'channel_title': 'imma channel'
    },
    {
        'influencer_name': 'Rozy',
        'channel_id': 'UC5nWy6Cu9vFVQnP8xJPdQqQ',
        'channel_title': 'ROZY'
    },
    {
        'influencer_name': 'Leya Love',
        'channel_id': 'UCg4STDrdEXxz1ZjCVBxxT7w',
        'channel_title': 'Leya Love'
    },
    {
        'influencer_name': 'Kyra',
        'channel_id': 'UCHRF1JtCzLNQwq9r_RCCddw',
        'channel_title': 'Kyra'
    },
    {
        'influencer_name': 'Milla Sofia',
        'channel_id': 'UCgH514RZsaqiRQFc5DkZyKg',
        'channel_title': 'Milla Sofia - Topic'
    },
]

# Target: 100 videos with transcripts per influencer
TARGET_TRANSCRIPTS_PER_INFLUENCER = 100


def get_youtube_client(api_key: str):
    """Initialize YouTube Data API v3 client."""
    return build('youtube', 'v3', developerKey=api_key)


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
    except HttpError as e:
        print(f"  API Error: {e}")
        return None


def get_recent_video_ids(youtube, playlist_id: str, max_videos: int = 500) -> List[str]:
    """Get recent video IDs from uploads playlist (most recent first)."""
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
        print(f"  API Error getting videos: {e}")
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
    transcripts_file = os.path.join(output_dir, 'transcripts.jsonl')
    
    # Load existing processed data
    processed_transcripts = load_processed_videos(transcripts_file)
    
    print("=" * 70)
    print("AI Influencers - Top 100 Recent Videos WITH Transcripts")
    print("=" * 70)
    print(f"\nUsing {len(AI_INFLUENCER_CHANNELS)} pre-configured channels")
    print(f"Already processed: {len(processed_transcripts)} videos")
    
    # Save channel info
    for channel in AI_INFLUENCER_CHANNELS:
        append_jsonl(channels_file, channel)
    
    # Process each channel
    print("\n--- Finding Recent Videos WITH Transcripts ---\n")
    
    total_success = 0
    total_checked = 0
    
    for channel in AI_INFLUENCER_CHANNELS:
        print(f"\n{'='*60}")
        print(f"Processing: {channel['channel_title']} ({channel['influencer_name']})")
        print(f"{'='*60}")
        
        # Get uploads playlist
        uploads_playlist = get_uploads_playlist_id(youtube, channel['channel_id'])
        if not uploads_playlist:
            print("  Could not find uploads playlist (API quota may be exceeded)")
            continue
        
        # Get recent video IDs
        print("  Fetching recent videos...")
        video_ids = get_recent_video_ids(youtube, uploads_playlist, max_videos=500)
        print(f"  Found {len(video_ids)} recent videos")
        
        if not video_ids:
            print("  No videos found or API quota exceeded")
            continue
        
        # Filter to videos not already processed
        video_ids = [vid for vid in video_ids if vid not in processed_transcripts]
        print(f"  {len(video_ids)} videos need processing")
        
        # Find videos with transcripts
        transcripts_found = 0
        
        pbar = tqdm(video_ids, desc="  Checking for transcripts", leave=True)
        
        for video_id in pbar:
            total_checked += 1
            
            # Check if transcript is available (this doesn't use YouTube API quota)
            transcript = check_transcript_available(video_id)
            
            if transcript:
                # Get video metadata (uses API quota but minimal)
                metadata = get_video_metadata(youtube, video_id)
                
                if metadata:
                    # Add influencer info
                    metadata['influencer_name'] = channel['influencer_name']
                    transcript['influencer_name'] = channel['influencer_name']
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
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        print(f"  ✓ Found {transcripts_found} videos with transcripts")
    
    # Summary
    print("\n" + "=" * 70)
    print("SCRAPING COMPLETE")
    print("=" * 70)
    print(f"\nChannels processed: {len(AI_INFLUENCER_CHANNELS)}")
    print(f"Videos checked: {total_checked}")
    print(f"Videos with transcripts saved: {total_success}")
    print(f"\nOutput files saved to: {output_dir}")
    print("  - channels.jsonl")
    print("  - videos_with_transcripts.jsonl")
    print("  - transcripts.jsonl")


if __name__ == '__main__':
    main()
