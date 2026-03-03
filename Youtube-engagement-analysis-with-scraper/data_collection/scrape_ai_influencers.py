"""
Script to scrape transcripts of top 100 videos from AI influencers.
AI influencers include: Lu of Magalu, Lil Miquela, Shudu, Noonoouri, 
Aitana Lopez, Imma, Rozy, Leya Love, Kyra, Milla Sofia
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

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transcripts import get_transcript

# Load environment variables
load_dotenv()


# AI Influencers to scrape with specific search terms
AI_INFLUENCERS = [
    "Lu do Magalu",           # Brazilian virtual influencer (Portuguese)
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


def get_youtube_client(api_key: str):
    """Initialize YouTube Data API v3 client."""
    return build('youtube', 'v3', developerKey=api_key)


def search_channel_by_name(youtube, channel_name: str) -> Optional[Dict]:
    """
    Search for a channel by name and return channel info.
    
    Args:
        youtube: YouTube API client
        channel_name: Channel name to search
        
    Returns:
        Dictionary with channel info or None
    """
    try:
        # Search for channel
        request = youtube.search().list(
            part='snippet',
            q=channel_name,
            type='channel',
            maxResults=5
        )
        response = request.execute()
        
        if response.get('items'):
            print(f"\nSearching for '{channel_name}':")
            for i, item in enumerate(response['items'], 1):
                channel_id = item['snippet']['channelId']
                channel_title = item['snippet']['title']
                print(f"  {i}. {channel_title} (ID: {channel_id})")
            
            # Return the first (most relevant) result
            first_result = response['items'][0]
            return {
                'search_term': channel_name,
                'channel_id': first_result['snippet']['channelId'],
                'channel_title': first_result['snippet']['title']
            }
        else:
            print(f"No channel found for '{channel_name}'")
            return None
            
    except HttpError as e:
        print(f"Error searching for '{channel_name}': {e}")
        return None


def get_channel_videos_by_popularity(youtube, channel_id: str, max_results: int = 100) -> List[Dict]:
    """
    Get top videos from a channel sorted by view count.
    
    Args:
        youtube: YouTube API client
        channel_id: YouTube channel ID
        max_results: Maximum number of videos to retrieve
        
    Returns:
        List of video dictionaries
    """
    videos = []
    
    try:
        # Search for videos from this channel, ordered by view count
        next_page_token = None
        
        while len(videos) < max_results:
            request = youtube.search().list(
                part='snippet',
                channelId=channel_id,
                type='video',
                order='viewCount',  # Sort by popularity
                maxResults=min(50, max_results - len(videos)),
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response.get('items', []):
                videos.append({
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'published_at': item['snippet']['publishedAt'],
                    'channel_id': channel_id
                })
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
        return videos[:max_results]
        
    except HttpError as e:
        print(f"Error getting videos for channel {channel_id}: {e}")
        return videos


def get_video_metadata_batch(youtube, video_ids: List[str]) -> List[Dict]:
    """
    Get metadata for multiple videos in batches.
    
    Args:
        youtube: YouTube API client
        video_ids: List of video IDs
        
    Returns:
        List of video metadata dictionaries
    """
    metadata_list = []
    
    # Process in batches of 50
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        
        try:
            request = youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=','.join(batch_ids)
            )
            response = request.execute()
            
            for item in response.get('items', []):
                snippet = item['snippet']
                statistics = item.get('statistics', {})
                
                metadata_list.append({
                    'video_id': item['id'],
                    'title': snippet['title'],
                    'channel_id': snippet['channelId'],
                    'channel_title': snippet['channelTitle'],
                    'published_at': snippet['publishedAt'],
                    'duration': item['contentDetails']['duration'],
                    'view_count': int(statistics.get('viewCount', 0)),
                    'like_count': int(statistics.get('likeCount', 0)),
                    'comment_count': int(statistics.get('commentCount', 0))
                })
                
        except HttpError as e:
            print(f"Error fetching metadata batch: {e}")
            
    return metadata_list


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


def append_jsonl_batch(filepath: str, data_list: List[dict]):
    """Append multiple records to a JSONL file."""
    if not data_list:
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'a', encoding='utf-8') as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def main():
    """Main entry point."""
    
    # Get API key
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print("Error: YOUTUBE_API_KEY not found in environment")
        print("Please set YOUTUBE_API_KEY in your .env file")
        return
    
    # Initialize YouTube client
    youtube = get_youtube_client(api_key)
    
    # Output paths
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'ai_influencers')
    os.makedirs(output_dir, exist_ok=True)
    
    channels_file = os.path.join(output_dir, 'channels.jsonl')
    videos_file = os.path.join(output_dir, 'videos.jsonl')
    transcripts_file = os.path.join(output_dir, 'transcripts.jsonl')
    
    # Load existing processed data for idempotency
    processed_transcripts = load_processed_videos(transcripts_file)
    
    print("=" * 60)
    print("AI Influencers Transcript Scraper")
    print("=" * 60)
    
    # Step 1: Find channel IDs for each AI influencer
    print("\n--- STEP 1: Finding Channel IDs ---\n")
    
    channels_found = []
    for influencer in AI_INFLUENCERS:
        channel_info = search_channel_by_name(youtube, influencer)
        if channel_info:
            channels_found.append(channel_info)
            append_jsonl(channels_file, channel_info)
        time.sleep(0.5)  # Rate limiting
    
    print(f"\nFound {len(channels_found)} channels out of {len(AI_INFLUENCERS)} influencers")
    
    if not channels_found:
        print("No channels found. Exiting.")
        return
    
    # Step 2: Get top 100 videos for each channel
    print("\n--- STEP 2: Getting Top 100 Videos Per Channel ---\n")
    
    all_videos = []
    for channel in channels_found:
        print(f"\nFetching videos for: {channel['channel_title']}")
        videos = get_channel_videos_by_popularity(youtube, channel['channel_id'], max_results=100)
        
        # Add influencer name to each video
        for video in videos:
            video['influencer_name'] = channel['search_term']
            video['channel_title'] = channel['channel_title']
        
        all_videos.extend(videos)
        print(f"  Found {len(videos)} videos")
        time.sleep(0.5)  # Rate limiting
    
    print(f"\nTotal videos found: {len(all_videos)}")
    
    # Get detailed metadata for all videos
    print("\nFetching detailed metadata...")
    video_ids = [v['video_id'] for v in all_videos]
    metadata_list = get_video_metadata_batch(youtube, video_ids)
    
    # Create a lookup for metadata
    metadata_lookup = {m['video_id']: m for m in metadata_list}
    
    # Merge influencer names with metadata
    for video in all_videos:
        if video['video_id'] in metadata_lookup:
            metadata = metadata_lookup[video['video_id']]
            metadata['influencer_name'] = video['influencer_name']
            append_jsonl(videos_file, metadata)
    
    print(f"Saved metadata for {len(metadata_list)} videos")
    
    # Step 3: Extract transcripts
    print("\n--- STEP 3: Extracting Transcripts ---\n")
    
    videos_to_process = [v for v in all_videos if v['video_id'] not in processed_transcripts]
    print(f"Videos needing transcripts: {len(videos_to_process)}")
    
    success_count = 0
    failed_count = 0
    
    for video in tqdm(videos_to_process, desc="Extracting transcripts"):
        video_id = video['video_id']
        
        transcript = get_transcript(video_id)
        
        if transcript:
            transcript['influencer_name'] = video.get('influencer_name', '')
            transcript['channel_title'] = video.get('channel_title', '')
            transcript['video_title'] = video.get('title', '')
            append_jsonl(transcripts_file, transcript)
            success_count += 1
        else:
            # Record failed attempt
            append_jsonl(transcripts_file, {
                'video_id': video_id,
                'influencer_name': video.get('influencer_name', ''),
                'channel_title': video.get('channel_title', ''),
                'video_title': video.get('title', ''),
                'transcript_source': 'failed',
                'segments': []
            })
            failed_count += 1
        
        # Rate limiting to avoid 429 errors
        time.sleep(0.3)
    
    # Summary
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"\nChannels found: {len(channels_found)}")
    print(f"Total videos: {len(all_videos)}")
    print(f"Transcripts extracted: {success_count}")
    print(f"Transcripts failed: {failed_count}")
    print(f"\nOutput files saved to: {output_dir}")
    print("  - channels.jsonl")
    print("  - videos.jsonl") 
    print("  - transcripts.jsonl")


if __name__ == '__main__':
    main()
