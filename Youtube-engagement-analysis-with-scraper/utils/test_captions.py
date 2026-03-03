"""Quick test to check if transcript API is working."""

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, IpBlocked, RequestBlocked

# Test with a known popular video that definitely has captions
# Using a TED Talk video which always has captions
test_videos = [
    'jNQXAC9IVRw',  # "Me at the zoo" - first YouTube video
    'dQw4w9WgXcQ',  # Rick Astley - Never Gonna Give You Up (definitely has captions)
    '9bZkp7q19f0',  # Gangnam Style (definitely has captions)
]

print("Testing transcript API with known videos that have captions...\n")

for video_id in test_videos:
    print(f"Testing video: {video_id}")
    print(f"  URL: https://www.youtube.com/watch?v={video_id}")
    
    try:
        api = YouTubeTranscriptApi()
        segments = api.fetch(video_id, languages=['en'])
        
        # Get first few segments
        count = 0
        for seg in segments[:3]:
            text = seg.text if hasattr(seg, 'text') else seg.get('text', '')
            print(f"    ✓ Got segment: {text[:50]}...")
            count += 1
        
        print(f"  ✓ SUCCESS - Got {len(segments)} segments total\n")
        break  # If one works, we're good
        
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"  ✗ No transcript available: {e}\n")
    except (IpBlocked, RequestBlocked) as e:
        print(f"  ✗ IP BLOCKED: {e}")
        print("  → Warp proxy is not working or IP is blocked by YouTube\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

print("\nTest complete!")
