"""
Add channel_id to videos.jsonl by mapping videos to their source channels.
"""

import json
import os
from dotenv import load_dotenv
from youtube_api import get_youtube_client, get_uploads_playlist, get_recent_videos

load_dotenv()

# Channel definitions from main.py
CHANNELS = {
    'UCyLqyEa45kWaSZlpvJvKhHA': 'Hautemess Tom',
    'UCc95c_6uMb1VyFEJjgydHRA': 'Tashira Halyard',
    'UCD9VnTKTGliNFiPTBfQUBYw': 'Heylulaa',
    'UColKM5Unut13hF9_e41RGkw': 'alexonabudget',
    'UCMjoPHi64Ofikn-udtEra4w': 'stylecrusader',
    'UCu0V4K1jf8cISkIzpi77p9Q': 'dermangelo',
    'UCquUgphHkwCF_d0qBLrfAdA': 'olenabeley',
}


def build_video_to_channel_map(api_key, years=2):
    """
    Build mapping of video_id -> channel_id by fetching videos from each channel.
    
    Returns:
        Dictionary mapping video_id to channel_id
    """
    print("Building video_id -> channel_id mapping...")
    print("=" * 60)
    
    youtube = get_youtube_client(api_key)
    video_to_channel = {}
    
    for channel_id, channel_name in CHANNELS.items():
        print(f"\nFetching videos from {channel_name}...")
        
        # Get uploads playlist
        uploads_playlist = get_uploads_playlist(youtube, channel_id)
        if not uploads_playlist:
            print(f"  Could not get uploads playlist")
            continue
        
        # Get recent videos
        video_ids = get_recent_videos(youtube, uploads_playlist, years)
        print(f"  Found {len(video_ids)} videos")
        
        # Map each video to this channel
        for video_id in video_ids:
            video_to_channel[video_id] = channel_id
    
    print(f"\n{'=' * 60}")
    print(f"Total videos mapped: {len(video_to_channel)}")
    return video_to_channel


def add_channel_ids_to_videos(videos_path, output_path, video_to_channel):
    """
    Add channel_id field to each video in videos.jsonl.
    
    Args:
        videos_path: Input videos.jsonl path
        output_path: Output path for updated videos.jsonl
        video_to_channel: Mapping of video_id -> channel_id
    """
    print(f"\nAdding channel_id to videos...")
    print("=" * 60)
    
    videos_with_channel = []
    videos_without_channel = []
    
    with open(videos_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            video = json.loads(line)
            video_id = video['video_id']
            
            # Add channel_id if we have it
            if video_id in video_to_channel:
                video['channel_id'] = video_to_channel[video_id]
                videos_with_channel.append(video)
            else:
                videos_without_channel.append(video)
    
    print(f"Videos with channel_id: {len(videos_with_channel)}")
    print(f"Videos without channel_id: {len(videos_without_channel)}")
    
    # Write updated videos
    with open(output_path, 'w', encoding='utf-8') as f:
        for video in videos_with_channel:
            f.write(json.dumps(video, ensure_ascii=False) + '\n')
    
    print(f"Saved updated videos to {output_path}")
    
    # Show distribution
    print("\nVideos per channel:")
    channel_counts = {}
    for video in videos_with_channel:
        ch_id = video['channel_id']
        channel_counts[ch_id] = channel_counts.get(ch_id, 0) + 1
    
    for channel_id, count in sorted(channel_counts.items(), key=lambda x: -x[1]):
        channel_name = CHANNELS.get(channel_id, channel_id)
        print(f"  {channel_name}: {count} videos")


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("ADDING CHANNEL_ID TO VIDEOS.JSONL")
    print("=" * 60 + "\n")
    
    # Get API key
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print("Error: YOUTUBE_API_KEY not found in .env")
        return
    
    # Build mapping
    video_to_channel = build_video_to_channel_map(api_key, years=2)
    
    # Add channel_ids to videos
    add_channel_ids_to_videos(
        'output/videos.jsonl',
        'output/videos_with_channels.jsonl',
        video_to_channel
    )
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Backup original: mv output/videos.jsonl output/videos_backup.jsonl")
    print("2. Use new file: mv output/videos_with_channels.jsonl output/videos.jsonl")
    print("3. Rerun analysis: python language_engagement_study.py")
    print()


if __name__ == '__main__':
    main()
