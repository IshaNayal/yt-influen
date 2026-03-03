"""
Search for videos ABOUT AI influencers (from news/tech/review channels)
These videos have spoken content with transcripts.
"""

import os
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


# Search queries for AI influencers
AI_INFLUENCER_SEARCHES = [
    "Lu do Magalu virtual influencer",
    "Lil Miquela interview",
    "Lil Miquela AI influencer",
    "Shudu virtual model",
    "Shudu Gram digital model",
    "Noonoouri virtual influencer",
    "Aitana Lopez AI model",
    "Imma virtual model Japan",
    "Rozy virtual influencer Korea",
    "Leya Love AI influencer",
    "Kyra virtual influencer India",
    "Milla Sofia AI influencer",
    "AI influencer documentary",
    "virtual influencer explained",
    "CGI influencer marketing",
]

TARGET_VIDEOS_PER_SEARCH = 20


def search_youtube_videos(query: str, max_results: int = 30) -> List[Dict]:
    """
    Search YouTube for videos matching a query using yt-dlp.
    """
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'default_search': 'ytsearch',
        }
        
        videos = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f'ytsearch{max_results}:{query}', download=False)
            
            if result and 'entries' in result:
                for entry in result['entries']:
                    if entry:
                        videos.append({
                            'video_id': entry.get('id', ''),
                            'title': entry.get('title', ''),
                            'channel': entry.get('channel', entry.get('uploader', '')),
                            'url': entry.get('url', ''),
                            'search_query': query
                        })
        
        return videos
        
    except Exception as e:
        print(f"  Search error: {e}")
        return []


def get_transcript(video_id: str) -> Optional[Dict]:
    """
    Fetch transcript for a video.
    """
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        for transcript in transcript_list:
            try:
                segments = transcript.fetch()
                
                formatted_segments = []
                for seg in segments:
                    formatted_segments.append({
                        'start': seg.start if hasattr(seg, 'start') else seg.get('start', 0),
                        'duration': seg.duration if hasattr(seg, 'duration') else seg.get('duration', 0),
                        'text': seg.text if hasattr(seg, 'text') else seg.get('text', '')
                    })
                
                # Get full text
                full_text = ' '.join([s['text'] for s in formatted_segments])
                
                return {
                    'video_id': video_id,
                    'transcript_source': 'youtube_captions',
                    'language': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'full_text': full_text,
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
    """Load already processed video IDs."""
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
    
    videos_file = os.path.join(output_dir, 'videos_about_ai_influencers.jsonl')
    transcripts_file = os.path.join(output_dir, 'transcripts.jsonl')
    
    # Load existing processed data
    processed_videos = load_processed_videos(transcripts_file)
    
    print("=" * 70)
    print("Searching for Videos ABOUT AI Influencers (with transcripts)")
    print("=" * 70)
    print(f"\nRunning {len(AI_INFLUENCER_SEARCHES)} search queries")
    print(f"Already processed: {len(processed_videos)} videos")
    
    total_success = 0
    total_checked = 0
    all_videos = []
    
    # Step 1: Search for videos
    print("\n--- STEP 1: Searching for videos ---\n")
    
    for query in AI_INFLUENCER_SEARCHES:
        print(f"Searching: '{query}'...")
        videos = search_youtube_videos(query, max_results=30)
        
        # Filter out already processed
        new_videos = [v for v in videos if v['video_id'] not in processed_videos and v['video_id']]
        all_videos.extend(new_videos)
        print(f"  Found {len(videos)} videos, {len(new_videos)} new")
        
        time.sleep(0.5)
    
    # Remove duplicates
    seen_ids = set()
    unique_videos = []
    for v in all_videos:
        if v['video_id'] not in seen_ids:
            seen_ids.add(v['video_id'])
            unique_videos.append(v)
    
    print(f"\nTotal unique videos to check: {len(unique_videos)}")
    
    # Step 2: Get transcripts
    print("\n--- STEP 2: Extracting Transcripts ---\n")
    
    pbar = tqdm(unique_videos, desc="Extracting transcripts")
    
    for video in pbar:
        video_id = video['video_id']
        total_checked += 1
        
        transcript = get_transcript(video_id)
        
        if transcript:
            # Add video metadata
            transcript['title'] = video.get('title', '')
            transcript['channel'] = video.get('channel', '')
            transcript['search_query'] = video.get('search_query', '')
            
            # Save video info
            append_jsonl(videos_file, video)
            append_jsonl(transcripts_file, transcript)
            
            total_success += 1
            processed_videos.add(video_id)
            
            pbar.set_description(f"Found {total_success} transcripts")
        
        time.sleep(0.1)
    
    # Summary
    print("\n" + "=" * 70)
    print("SCRAPING COMPLETE")
    print("=" * 70)
    print(f"\nVideos checked: {total_checked}")
    print(f"Videos with transcripts: {total_success}")
    print(f"\nOutput saved to: {output_dir}")
    print("  - videos_about_ai_influencers.jsonl")
    print("  - transcripts.jsonl")


if __name__ == '__main__':
    main()
