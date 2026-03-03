# YouTube Channel Scraping and Analysis Pipeline

Research-grade Python pipeline for scraping YouTube channels and analyzing language-engagement relationships.

## Features

- **Video Discovery**: Fetches all videos from a channel's uploads playlist from the last 2 years
- **Metadata Extraction**: Collects video ID, title, published date, duration (ISO 8601), view count, like count, and comment count
- **Transcript Extraction**:
  - First attempts to fetch captions via `youtube-transcript-api`
  - Falls back to local Whisper transcription when captions are unavailable
  - Records transcript source for each video
- **Comment Scraping**: Fetches all top-level comments with full pagination
- **Proxy Support**: Bypass YouTube IP blocks using rotating residential proxies ([See PROXY_GUIDE.md](PROXY_GUIDE.md))
- **Robust Error Handling**: Gracefully handles disabled comments, missing videos, and API failures
- **Idempotent**: Safe to rerun without duplicating results

## Requirements

- Python 3.9+
- YouTube Data API v3 key ([Get one here](https://console.cloud.google.com/apis/credentials))
- FFmpeg (required by yt-dlp for audio extraction)

## Installation

### 1. Install FFmpeg

**Windows (using winget):**

```powershell
winget install FFmpeg
```

**Windows (using Chocolatey):**

```powershell
choco install ffmpeg
```

**macOS:**

```bash
brew install ffmpeg
```

**Linux:**

```bash
sudo apt-get install ffmpeg
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Installing Whisper may take several minutes as it downloads model weights.

### 3. Set Up API Key

1. Create a `.env` file in the project root:

```bash
cp .env.example .env
```

2. Edit `.env` and add your YouTube API key:

```
YOUTUBE_API_KEY=your_actual_api_key_here
```

## Project Structure

```
├── data_collection/              # Data scraping scripts
│   ├── main.py                  # Main pipeline orchestration
│   ├── youtube_api.py           # YouTube API wrappers
│   ├── transcripts.py           # Caption extraction
│   └── comments.py              # Comment scraping
├── analysis/                     # Analysis scripts
│   ├── merge_liwc_results.py           # LIWC integration with metadata
│   ├── run_regression.py               # Linear regression analysis
│   ├── improve_regression.py           # ML model comparison
│   ├── language_engagement_study.py    # Corpus-wide bigram analysis
│   ├── bigram_feature_regression.py    # Per-video bigram analysis
│   ├── prepare_liwc_data_csv.py        # LIWC data preparation
│   └── analyze_metadata.py             # Exploratory analysis
├── scripts/                      # Utility scripts
│   ├── check_all_bigrams.py     # Bigram verification
│   └── create_bigram_summary.py # Bigram summary generation
├── utils/                        # Helper utilities
│   ├── find_channel_ids.py      # Channel ID lookup
│   ├── add_channel_ids.py       # Retrospective channel mapping
│   └── test_captions.py         # IP block detection
├── reports/                      # Final academic reports
│   ├── PROFESSOR_REPORT.md      # Complete academic report
│   └── ONE_PAGE_SUMMARY.md      # Quick reference summary
└── output/                       # Generated data and results
    ├── raw_data/                # Original collected data
    │   ├── videos.jsonl         # Video metadata
    │   ├── transcripts.jsonl    # Video transcripts
    │   └── comments.jsonl       # Raw comments
    ├── liwc_prep/               # LIWC preparation files
    │   ├── youtube_transcripts_for_liwc.csv
    │   ├── youtube_comments_for_liwc.csv
    │   └── youtube_metadata.xlsx
    ├── liwc_results/            # LIWC-22 analysis output
    │   └── LIWC-22 Results - youtube_transcripts_for_liwc - LIWC Analysis.xlsx
    ├── bigram_analysis/         # Bigram analysis results
    │   ├── pervideo_bigrams_combined.csv
    │   ├── pervideo_bigrams_like_coefficients.csv
    │   └── model_performance.txt
    └── regression_results/      # Final regression models
        ├── final_regression_dataset.xlsx (2,654 × 134 variables)
        ├── regression_results.xlsx (42 significant predictors)
        └── improvement_analysis.xlsx (ML comparison)
```

## Usage

### Data Collection

1. Edit `data_collection/main.py` and add your channel IDs to the `channel_ids` list:

```python
channel_ids = [
    'UCuAXFkgsw1L7xaCfnd5JJOw',  # Example channel ID
    'UC_x5XG1OV2P6uZZ5FSM9Ttw',  # Another channel
]
```

2. Run the pipeline:

```bash
python data_collection/main.py
```

### Language-Engagement Analysis

Run either analysis approach:

**Corpus-wide bigrams** (extracts 100 most common bigrams across all videos):

```bash
python analysis/language_engagement_study.py
```

**Per-video bigrams** (extracts statistics from each video's top 100 bigrams):

```bash
python analysis/language_engagement_study_pervideo.py
```

Results are saved to `output/model_performance.txt` and CSV files with coefficients.

### Advanced Usage

**Scrape specific channels programmatically:**

```python
from main import scrape_channels
import os

api_key = os.getenv('YOUTUBE_API_KEY')
channel_ids = ['UCuAXFkgsw1L7xaCfnd5JJOw']

scrape_channels(
    channel_ids=channel_ids,
    api_key=api_key,
    output_dir='output',
    years=2,                    # Look back 2 years
    skip_transcripts=False,     # Set True to skip transcripts
    skip_comments=False         # Set True to skip comments
)
```

**Scrape a single channel:**

```python
from main import scrape_channel
import os

api_key = os.getenv('YOUTUBE_API_KEY')

scrape_channel(
    channel_id='UCuAXFkgsw1L7xaCfnd5JJOw',
    api_key=api_key,
    output_dir='output',
    years=2
)
```

## Output Format

All outputs are stored as JSONL (JSON Lines) files in the `output/` directory:

### videos.jsonl

```json
{
  "video_id": "dQw4w9WgXcQ",
  "title": "Video Title",
  "published_at": "2023-05-15T14:30:00Z",
  "duration": "PT4M33S",
  "viewCount": 1234567,
  "likeCount": 50000,
  "commentCount": 1500
}
```

### transcripts.jsonl

```json
{
  "video_id": "dQw4w9WgXcQ",
  "transcript_source": "youtube_captions",
  "segments": [
    {
      "start": 0.0,
      "duration": 3.5,
      "text": "Hello everyone"
    }
  ]
}
```

For Whisper transcripts, includes additional `language` field.

### comments.jsonl

```json
{
  "video_id": "dQw4w9WgXcQ",
  "comment_text": "Great video!",
  "author_name": "John Doe",
  "like_count": 42,
  "published_at": "2023-05-16T10:20:30Z"
}
```

## Architecture

### Module Overview

- **`youtube_api.py`**: YouTube Data API v3 integration
  - `get_uploads_playlist()`: Get channel's uploads playlist ID
  - `get_recent_videos()`: Fetch videos from last N years with pagination
  - `get_video_metadata()`: Extract comprehensive video metadata
  - `get_video_metadata_batch()`: Batch metadata fetching (up to 50 videos)

- **`transcripts.py`**: Transcript extraction with Whisper fallback
  - `get_transcript()`: Fetch captions via youtube-transcript-api (with proxy support)
  - `whisper_transcribe()`: Local Whisper transcription
  - `get_transcript_with_fallback()`: Automatic fallback logic

- **`comments.py`**: Comment scraping
  - `get_all_comments()`: Fetch all comments with full pagination
  - Handles disabled comments gracefully

- **`main.py`**: Orchestration and idempotency
  - Coordinates all modules
  - Implements idempotency via processed video tracking
  - Handles errors gracefully
  - Progress tracking with tqdm

## API Quota Considerations

YouTube Data API v3 has daily quota limits (default: 10,000 units/day).

**Cost per video:**

- Video list: 1 unit (batch of 50)
- Comment threads: ~1 unit per 100 comments
- Transcript API: No quota cost (separate API)
- Whisper: No quota cost (local processing)

**Example:** Scraping a channel with 100 videos and 10,000 total comments will cost approximately 102 units.



## Troubleshooting

**Issue: "YOUTUBE_API_KEY not found"**

- Ensure `.env` file exists and contains your API key
- Verify the key name is exactly `YOUTUBE_API_KEY`

**Issue: FFmpeg not found**

- Install FFmpeg and ensure it's in your system PATH
- Restart terminal after installation

**Issue: 429 Too Many Requests / IP Blocked / 403 Forbidden**

- YouTube is blocking your IP (common on cloud providers)
- **Solution:** Use proxies to bypass blocks
- **Quick fix:** Use rotating residential proxies (Use WARP)



**Issue: API quota exceeded**

- Wait 24 hours for quota reset
- Request quota increase in Google Cloud Console
- Use `skip_comments=True` to reduce quota usage

## Example: Complete Workflow

```python
from main import scrape_channels
import os
from dotenv import load_dotenv

load_dotenv()

# Your channel IDs
channels = [
    'UCuAXFkgsw1L7xaCfnd5JJOw',
    'UC_x5XG1OV2P6uZZ5FSM9Ttw'
]

# Run scraper
scrape_channels(
    channel_ids=channels,
    api_key=os.getenv('YOUTUBE_API_KEY'),
    output_dir='output',
    years=2
)

print("✓ Scraping complete!")
print("Results saved to output/")
```

## License

MIT

## Credits

Built with:

- [google-api-python-client](https://github.com/googleapis/google-api-python-client)
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [OpenAI Whisper](https://github.com/openai/whisper)
