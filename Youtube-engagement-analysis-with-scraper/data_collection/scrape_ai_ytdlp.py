"""
Script to scrape transcripts from AI influencers using yt-dlp (no API quota needed).
Gets recent videos from channels and extracts transcripts.
"""

import os
import sys
import json
import time
from typing import List, Dict, Optional, Set
from tqdm import tqdm
import yt_dlp
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


# AI Influencer channels with their YouTube channel URLs
AI_INFLUENCER_CHANNELS = [
    {
        'influencer_name': 'Lu do Magalu',
        'channel_url': 'https://www.youtube.com/@MagazineLuiza',
        'channel_id': 'UCeaQ72LrN6K3f9a8JkFV98w'
    },
    {
        'influencer_name': 'Lil Miquela',
        'channel_url': 'https://www.youtube.com/@maboroshimiquela',
        'channel_id': 'UCWeHb_SrtJbrT8VD-_QQpRA'
    },
    {
        'influencer_name': 'Noonoouri',
        'channel_url': 'https://www.youtube.com/@noonoouri',
        'channel_id': 'UCmIj9ZSb2QAurwfD3g2G2eQ'
    },
    {
        'influencer_name': 'Aitana Lopez',
        'channel_url': 'https://www.youtube.com/@aitanalopez_oficial',
        'channel_id': 'UCx_Z4tFuH4llsgU7N197cUQ'
    },
    {
        'influencer_name': 'Imma',
        'channel_url': 'https://www.youtube.com/@immachannel',
        'channel_id': 'UCaAPSltuomXWut-skKB2WHg'
    },
    {
        'influencer_name': 'Rozy',
        'channel_url': 'https://www.youtube.com/@ROZY_official',
        'channel_id': 'UC5nWy6Cu9vFVQnP8xJPdQqQ'
    },
    {
        'influencer_name': 'Leya Love',
        'channel_url': 'https://www.youtube.com/@LeyaLove',
        'channel_id': 'UCg4STDrdEXxz1ZjCVBxxT7w'
    },
    {
        'influencer_name': 'Kyra',
        'channel_url': 'https://www.youtube.com/@KyraOnIG',
        'channel_id': 'UCHRF1JtCzLNQwq9r_RCCddw'
    },
    {
        'influencer_name': 'Milla Sofia',
        'channel_url': 'https://www.youtube.com/@MillaSofia',
        'channel_id': 'UCgH514RZsaqiRQFc5DkZyKg'
    },
]

TARGET_TRANSCRIPTS_PER_INFLUENCER = 100


def get_channel_videos_ytdlp(channel_url: str, max_videos: int = 200) -> List[Dict]:
    """
    Get recent videos from a channel using yt-dlp Python library (no API quota needed).
    Returns list of video info dicts.
    """
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'playlistend': max_videos,
        }
        
        videos = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f'{channel_url}/videos', download=False)
            
            if result and 'entries' in result:
                for entry in result['entries']:
                    if entry:
                        videos.append({
                            'video_id': entry.get('id', ''),
                            'title': entry.get('title', ''),
                            'url': entry.get('url', '')
                        })
        
        return videos
        
    except Exception as e:
        print(f"  Error: {e}")
        return []


def get_transcript(video_id: str) -> Optional[Dict]:
    """
    Fetch transcript for a video.
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
    
    # Output paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'output', 'ai_influencers')
    os.makedirs(output_dir, exist_ok=True)
    
    channels_file = os.path.join(output_dir, 'channels.jsonl')
    videos_file = os.path.join(output_dir, 'videos_with_transcripts.jsonl')
    transcripts_file = os.path.join(output_dir, 'transcripts.jsonl')
    
    # Load existing processed data
    processed_transcripts = load_processed_videos(transcripts_file)
    
    print("=" * 70)
    print("AI Influencers - Top 100 Recent Videos WITH Transcripts")
    print("Using yt-dlp (No API Quota Required)")
    print("=" * 70)
    print(f"\nProcessing {len(AI_INFLUENCER_CHANNELS)} channels")
    print(f"Already processed: {len(processed_transcripts)} videos")
    
    total_success = 0
    total_checked = 0
    
    for channel in AI_INFLUENCER_CHANNELS:
        print(f"\n{'='*60}")
        print(f"Processing: {channel['influencer_name']}")
        print(f"Channel: {channel['channel_url']}")
        print(f"{'='*60}")
        
        # Save channel info
        append_jsonl(channels_file, channel)
        
        # Get recent videos using yt-dlp
        print("  Fetching recent videos...")
        videos = get_channel_videos_ytdlp(channel['channel_url'], max_videos=300)
        print(f"  Found {len(videos)} videos")
        
        if not videos:
            print("  No videos found, trying alternative URL...")
            # Try with channel ID
            alt_url = f"https://www.youtube.com/channel/{channel['channel_id']}"
            videos = get_channel_videos_ytdlp(alt_url, max_videos=300)
            print(f"  Found {len(videos)} videos from alt URL")
        
        if not videos:
            continue
        
        # Filter to videos not already processed
        videos = [v for v in videos if v['video_id'] not in processed_transcripts and v['video_id']]
        print(f"  {len(videos)} videos need processing")
        
        # Find videos with transcripts
        transcripts_found = 0
        
        pbar = tqdm(videos, desc="  Checking transcripts", leave=True)
        
        for video in pbar:
            video_id = video['video_id']
            if not video_id:
                continue
                
            total_checked += 1
            
            # Check if transcript is available
            transcript = get_transcript(video_id)
            
            if transcript:
                # Add metadata
                transcript['influencer_name'] = channel['influencer_name']
                transcript['channel_url'] = channel['channel_url']
                transcript['video_title'] = video.get('title', '')
                
                # Save video info
                video_data = {
                    'video_id': video_id,
                    'title': video.get('title', ''),
                    'influencer_name': channel['influencer_name'],
                    'channel_url': channel['channel_url']
                }
                
                append_jsonl(videos_file, video_data)
                append_jsonl(transcripts_file, transcript)
                
                transcripts_found += 1
                total_success += 1
                processed_transcripts.add(video_id)
                
                pbar.set_description(f"  Found {transcripts_found}/{TARGET_TRANSCRIPTS_PER_INFLUENCER}")
                
                # Stop if we have enough
                if transcripts_found >= TARGET_TRANSCRIPTS_PER_INFLUENCER:
                    break
            
            # Small delay
            time.sleep(0.1)
        
        print(f"  ✓ Found {transcripts_found} videos with transcripts")
    
    # Summary
    print("\n" + "=" * 70)
    print("SCRAPING COMPLETE")
    print("=" * 70)
    print(f"\nChannels processed: {len(AI_INFLUENCER_CHANNELS)}")
    print(f"Videos checked: {total_checked}")
    print(f"Videos with transcripts: {total_success}")
    print(f"\nOutput saved to: {output_dir}")
    print("  - channels.jsonl")
    print("  - videos_with_transcripts.jsonl")
    print("  - transcripts.jsonl")


if __name__ == '__main__':
    main()
