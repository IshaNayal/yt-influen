"""
Scrape transcripts from videos about AI influencers using yt-dlp subtitles.
"""

import os
import json
import time
import tempfile
from typing import List, Dict, Optional, Set
from tqdm import tqdm
import yt_dlp


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

TARGET_VIDEOS = 100  # Total videos with transcripts


def search_youtube_videos(query: str, max_results: int = 20) -> List[Dict]:
    """
    Search YouTube for videos using yt-dlp.
    """
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        videos = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f'ytsearch{max_results}:{query}', download=False)
            
            if result and 'entries' in result:
                for entry in result['entries']:
                    if entry and entry.get('id'):
                        videos.append({
                            'video_id': entry.get('id', ''),
                            'title': entry.get('title', ''),
                            'channel': entry.get('channel', entry.get('uploader', '')),
                            'search_query': query
                        })
        
        return videos
        
    except Exception as e:
        print(f"  Search error: {e}")
        return []


def get_subtitles_ytdlp(video_id: str) -> Optional[Dict]:
    """
    Get subtitles for a video using yt-dlp.
    Returns transcript data or None.
    """
    try:
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-GB'],
            'subtitlesformat': 'json3',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            if not info:
                return None
            
            # Try to get English subtitles
            subtitles = info.get('subtitles', {})
            auto_captions = info.get('automatic_captions', {})
            
            # Find English subtitles
            sub_data = None
            sub_lang = None
            is_auto = False
            
            # First try manual subtitles
            for lang in ['en', 'en-US', 'en-GB']:
                if lang in subtitles:
                    sub_data = subtitles[lang]
                    sub_lang = lang
                    break
            
            # Fall back to auto-generated
            if not sub_data:
                for lang in ['en', 'en-US', 'en-GB']:
                    if lang in auto_captions:
                        sub_data = auto_captions[lang]
                        sub_lang = lang
                        is_auto = True
                        break
            
            if not sub_data:
                return None
            
            # Get the subtitle URL (prefer json3 format)
            sub_url = None
            for fmt in sub_data:
                if fmt.get('ext') == 'json3':
                    sub_url = fmt.get('url')
                    break
            
            if not sub_url:
                # Try vtt format
                for fmt in sub_data:
                    if fmt.get('ext') == 'vtt':
                        sub_url = fmt.get('url')
                        break
            
            if not sub_url:
                return None
            
            # Download and parse subtitles
            import urllib.request
            import ssl
            
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            with urllib.request.urlopen(sub_url, context=ctx) as response:
                content = response.read().decode('utf-8')
            
            # Parse based on format
            segments = []
            full_text = ""
            
            if sub_url.endswith('json3') or '&fmt=json3' in sub_url:
                # JSON3 format
                data = json.loads(content)
                events = data.get('events', [])
                
                for event in events:
                    if 'segs' in event:
                        text = ''.join([s.get('utf8', '') for s in event['segs']])
                        if text.strip():
                            segments.append({
                                'start': event.get('tStartMs', 0) / 1000,
                                'duration': event.get('dDurationMs', 0) / 1000,
                                'text': text.strip()
                            })
                            full_text += " " + text.strip()
            else:
                # VTT format - simple parsing
                lines = content.split('\n')
                current_text = []
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('WEBVTT') and not '-->' in line and not line.isdigit():
                        current_text.append(line)
                
                full_text = ' '.join(current_text)
                segments = [{'text': full_text}]
            
            if not full_text.strip():
                return None
            
            return {
                'video_id': video_id,
                'transcript_source': 'youtube_captions',
                'language': sub_lang,
                'is_auto_generated': is_auto,
                'full_text': full_text.strip(),
                'segments': segments,
                'title': info.get('title', ''),
                'channel': info.get('channel', info.get('uploader', '')),
                'duration': info.get('duration', 0),
                'view_count': info.get('view_count', 0),
            }
            
    except Exception as e:
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
    
    transcripts_file = os.path.join(output_dir, 'transcripts.jsonl')
    
    # Load existing
    processed_videos = load_processed_videos(transcripts_file)
    
    print("=" * 70)
    print("AI Influencer Videos - Transcript Scraper (yt-dlp)")
    print("=" * 70)
    print(f"\nSearching with {len(AI_INFLUENCER_SEARCHES)} queries")
    print(f"Already have: {len(processed_videos)} transcripts")
    print(f"Target: {TARGET_VIDEOS} total transcripts")
    
    total_success = len(processed_videos)
    
    # Search and collect
    print("\n--- Searching and extracting transcripts ---\n")
    
    for query in AI_INFLUENCER_SEARCHES:
        if total_success >= TARGET_VIDEOS:
            print(f"\nReached target of {TARGET_VIDEOS} transcripts!")
            break
            
        print(f"\nSearching: '{query}'")
        videos = search_youtube_videos(query, max_results=15)
        
        # Filter already processed
        new_videos = [v for v in videos if v['video_id'] not in processed_videos]
        print(f"  Found {len(new_videos)} new videos")
        
        for video in tqdm(new_videos, desc="  Extracting", leave=False):
            if total_success >= TARGET_VIDEOS:
                break
                
            video_id = video['video_id']
            
            result = get_subtitles_ytdlp(video_id)
            
            if result:
                result['search_query'] = query
                append_jsonl(transcripts_file, result)
                processed_videos.add(video_id)
                total_success += 1
                print(f"    ✓ {result['title'][:50]}... ({result['language']})")
            
            time.sleep(0.5)  # Rate limiting
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nTotal transcripts: {total_success}")
    print(f"Output: {transcripts_file}")


if __name__ == '__main__':
    main()
