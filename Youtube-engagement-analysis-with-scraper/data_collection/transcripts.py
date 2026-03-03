"""
Transcript extraction module.
Fetches captions using youtube-transcript-api only.
Supports multiple languages for AI influencers from different countries.
"""

import time
from typing import Dict, List, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig
try:
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
        TooManyRequests,
        RequestBlocked,
        IpBlocked
    )
except ImportError:
    # Some errors may not exist in older versions
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable
    )
    TooManyRequests = Exception
    RequestBlocked = Exception
    IpBlocked = Exception


# Languages to try for AI influencers from different countries
LANGUAGES_TO_TRY = [
    'en', 'en-US', 'en-GB',   # English
    'pt', 'pt-BR',            # Portuguese (Lu of Magalu)
    'es', 'es-ES',            # Spanish (Aitana Lopez)
    'de',                     # German (Noonoouri)
    'ja',                     # Japanese (Imma)
    'ko',                     # Korean (Rozy)
    'fi',                     # Finnish (Milla Sofia)
    'hi', 'en-IN',            # Hindi/Indian English (Kyra)
]


def get_transcript(video_id: str, proxy_config: Optional[GenericProxyConfig] = None) -> Optional[Dict]:
    """
    Attempt to fetch transcript using YouTube captions in multiple languages.
    
    Args:
        video_id: YouTube video ID
        proxy_config: Optional proxy configuration to bypass IP blocks
        
    Returns:
        Dictionary with transcript data and source, or None if unavailable
    """
    try:
        # Create API client with optional proxy
        if proxy_config:
            api = YouTubeTranscriptApi(proxy_config=proxy_config)
        else:
            api = YouTubeTranscriptApi()
        
        # Try to fetch transcript in any available language
        segments = None
        detected_language = None
        
        # First try with preferred languages list
        try:
            segments = api.fetch(video_id, languages=LANGUAGES_TO_TRY)
            detected_language = 'auto'
        except NoTranscriptFound:
            # Try to get any available transcript
            try:
                transcript_list = api.list(video_id)
                # Get first available transcript
                for transcript in transcript_list:
                    try:
                        segments = transcript.fetch()
                        detected_language = transcript.language_code
                        break
                    except:
                        continue
            except:
                pass
        
        if segments is None:
            return None
        
        # Format segments - API v1.2+ returns objects with attributes
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
            'language': detected_language,
            'segments': formatted_segments
        }
    
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except (TooManyRequests, RequestBlocked, IpBlocked) as e:
        # Rate limited or IP blocked - wait briefly and return None
        time.sleep(5)
        return None
    except Exception as e:
        if '429' in str(e) or 'Too Many Requests' in str(e) or 'blocked' in str(e).lower():
            time.sleep(5)
        return None



