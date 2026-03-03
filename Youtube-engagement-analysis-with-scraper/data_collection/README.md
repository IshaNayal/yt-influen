# Data Collection Scripts

This directory contains scripts for scraping YouTube channels and collecting video metadata, transcripts, and comments.

## Scripts

### main.py

**Main pipeline orchestration script**

- Coordinates the entire data collection process
- Loops through configured channel IDs
- Calls YouTube API, transcript, and comment scrapers
- Supports proxy configuration for IP blocking issues
- Optimized to use caption API only (no Whisper fallback)

**Usage:**

```bash
python data_collection/main.py
```

### youtube_api.py

**YouTube Data API v3 wrappers**

- `get_youtube_client()`: Creates authenticated API client
- `get_recent_videos()`: Fetches videos from channel's uploads playlist
- `get_video_metadata_batch()`: Retrieves metadata for up to 50 videos
- `get_all_comments()`: Scrapes all comments with pagination

**Features:**

- Batch processing for efficiency
- Proper error handling for API quotas
- Filters videos by publication date (last 2 years)

### transcripts.py

**Caption extraction via youtube-transcript-api**

- `get_transcript(video_id, proxy_config)`: Fetches captions for a video
- Handles manually created and auto-generated captions
- Supports proxy configuration to bypass IP blocks
- Returns combined transcript text

**Processing time:** ~3 seconds per video

### comments.py

**Comment scraping with pagination**

- `scrape_comments(youtube, video_id)`: Fetches all top-level comments
- Handles disabled comments gracefully
- Uses pagination with maxResults=100
- Saves individual comment text, author, timestamp

## Output Format

All data is saved to JSONL files in the `output/` directory:

**videos.jsonl** - One line per video:

```json
{
  "video_id": "dQw4w9WgXcQ",
  "title": "Video Title",
  "published_at": "2023-01-15T10:30:00Z",
  "duration_iso": "PT4M30S",
  "view_count": 1000000,
  "like_count": 50000,
  "comment_count": 1500,
  "channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw"
}
```

**transcripts.jsonl** - One line per transcript:

```json
{
  "video_id": "dQw4w9WgXcQ",
  "transcript": "Combined transcript text...",
  "source": "captions"
}
```

**comments.jsonl** - One line per comment:

```json
{
  "video_id": "dQw4w9WgXcQ",
  "comment_id": "UgxKREWPli4DLtf...",
  "text": "Comment text",
  "author": "Username",
  "published_at": "2023-01-16T14:20:00Z"
}
```

## Configuration

Channel IDs are configured in `main.py`:

```python
channel_ids = [
    'UCuAXFkgsw1L7xaCfnd5JJOw',  # Channel 1
    'UC_x5XG1OV2P6uZZ5FSM9Ttw',  # Channel 2
]
```

## IP Blocking

If you encounter YouTube IP blocks:

1. Connect to Warp VPN (free tier available)
2. The script will automatically use proxy configuration
3. Blocks typically occur after ~60 requests
4. See `../utils/test_captions.py` to check block status
