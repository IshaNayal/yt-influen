"""
LIWC Data Preparation for YouTube Engagement Analysis

Prepares three Excel files for LIWC-22 analysis:
1. youtube_metadata.xlsx - Video metrics
2. youtube_transcripts_for_liwc.xlsx - Raw transcript text
3. youtube_comments_for_liwc.xlsx - Aggregated comment text

Following methodology from:
"Leveraging Machine Learning and Generative AI for Content Engagement"

The professor will compute all linguistic features using LIWC-22.
This script only prepares and cleans the text data.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# Fixed scrape date for consistent recency calculation
SCRAPE_DATE = datetime.now(tz=timezone.utc)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_jsonl(filepath):
    """
    Load JSONL file into list of dictionaries.
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of dictionaries
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Required file not found: {filepath}")
    
    data = []
    print(f"Loading {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
    
    print(f"  ✓ Loaded {len(data)} records")
    return data


def parse_iso_duration(duration_str):
    """
    Parse ISO 8601 duration to total minutes.
    
    Handles formats:
    - PT10M30S
    - PT1H2M3S
    - PT45S
    - PT5M
    - PT1H
    
    Args:
        duration_str: ISO 8601 duration string
        
    Returns:
        Total minutes as float
    """
    if not duration_str or not isinstance(duration_str, str):
        return 0.0
    
    # Pattern: PT(hours)H(minutes)M(seconds)S
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    
    if not match:
        return 0.0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    # Convert to total minutes
    total_minutes = hours * 60 + minutes + seconds / 60.0
    
    return total_minutes


def clean_text_for_liwc(text):
    """
    Minimal cleaning for LIWC analysis.
    
    Preserves:
    - Capitalization
    - Punctuation
    - Pronouns
    - Verb tenses
    - Stopwords
    
    Removes:
    - Emojis
    - Non-ASCII control characters
    - Excessive whitespace
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove emojis (Unicode emoji ranges)
    # Covers most common emoji blocks
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(' ', text)
    
    # Remove control characters (but keep normal whitespace)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize whitespace (collapse multiple spaces/newlines to single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


# ============================================================================
# STEP 1 — LOAD DATA
# ============================================================================

def load_all_data(videos_path, transcripts_path, comments_path):
    """
    Load all three JSONL files.
    
    Returns:
        Tuple of (df_videos, df_transcripts, df_comments)
    """
    print("\n" + "=" * 80)
    print("STEP 1 — LOAD DATA")
    print("=" * 80 + "\n")
    
    # Load videos
    videos = load_jsonl(videos_path)
    df_videos = pd.DataFrame(videos)
    
    # Load transcripts
    transcripts = load_jsonl(transcripts_path)
    df_transcripts = pd.DataFrame(transcripts)
    
    # Load comments
    comments = load_jsonl(comments_path)
    df_comments = pd.DataFrame(comments)
    
    print(f"\nData loaded successfully:")
    print(f"  Videos: {len(df_videos)}")
    print(f"  Transcripts: {len(df_transcripts)}")
    print(f"  Comments: {len(df_comments)}")
    
    return df_videos, df_transcripts, df_comments


# ============================================================================
# STEP 2 — METADATA FILE
# ============================================================================

def create_metadata_file(df_videos):
    """
    Create metadata file with video metrics.
    
    Columns:
    - video_id
    - like_count
    - view_count
    - recency_days
    - duration_minutes
    - duration_dummy
    
    Returns:
        DataFrame ready for export
    """
    print("\n" + "=" * 80)
    print("STEP 2 — CREATE METADATA FILE")
    print("=" * 80 + "\n")
    
    df = df_videos.copy()
    
    # Standardize column names (handle both formats)
    if 'viewCount' in df.columns:
        df = df.rename(columns={
            'viewCount': 'view_count',
            'likeCount': 'like_count',
            'commentCount': 'comment_count'
        })
    
    # Select required columns
    required_cols = ['video_id', 'view_count', 'like_count', 'published_at', 'duration']
    
    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in videos.jsonl: {missing_cols}")
    
    df = df[required_cols].copy()
    
    print("Processing metadata...")
    
    # Handle missing like_count and view_count
    df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce').fillna(0).astype(int)
    df['view_count'] = pd.to_numeric(df['view_count'], errors='coerce').fillna(0).astype(int)
    
    # Drop videos with missing published_at
    initial_count = len(df)
    df = df.dropna(subset=['published_at']).copy()
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(f"  ⚠ Dropped {dropped_count} videos with missing published_at")
    
    # Compute recency_days
    print("  Computing recency_days...")
    df['published_at'] = pd.to_datetime(df['published_at'])
    df['recency_days'] = (SCRAPE_DATE - df['published_at']).dt.days
    df['recency_days'] = df['recency_days'].clip(lower=0)  # Ensure non-negative
    
    # Parse duration
    print("  Parsing duration_minutes...")
    df['duration_minutes'] = df['duration'].apply(parse_iso_duration)
    
    # Create duration_dummy
    print("  Creating duration_dummy...")
    df['duration_dummy'] = (df['duration_minutes'] > 8).astype(int)
    
    # Select final columns
    df_metadata = df[['video_id', 'like_count', 'view_count', 
                      'recency_days', 'duration_minutes', 'duration_dummy']].copy()
    
    # Ensure all numeric
    numeric_cols = ['like_count', 'view_count', 'recency_days', 'duration_minutes', 'duration_dummy']
    for col in numeric_cols:
        df_metadata[col] = pd.to_numeric(df_metadata[col], errors='coerce').fillna(0)
    
    # Convert to appropriate dtypes
    df_metadata['like_count'] = df_metadata['like_count'].astype(int)
    df_metadata['view_count'] = df_metadata['view_count'].astype(int)
    df_metadata['recency_days'] = df_metadata['recency_days'].astype(int)
    df_metadata['duration_dummy'] = df_metadata['duration_dummy'].astype(int)
    
    # Validate no NaN
    if df_metadata.isnull().any().any():
        print("  ⚠ Warning: NaN values detected, filling with 0")
        df_metadata = df_metadata.fillna(0)
    
    print(f"\n✓ Metadata file ready: {len(df_metadata)} videos")
    print("\nMetadata Summary:")
    print(df_metadata.describe())
    
    return df_metadata


# ============================================================================
# STEP 3 — TRANSCRIPT FILE
# ============================================================================

def create_transcript_file(df_transcripts, valid_video_ids):
    """
    Create transcript file for LIWC analysis.
    
    Columns:
    - video_id
    - transcript_text
    
    Returns:
        DataFrame ready for export
    """
    print("\n" + "=" * 80)
    print("STEP 3 — CREATE TRANSCRIPT FILE")
    print("=" * 80 + "\n")
    
    df = df_transcripts.copy()
    
    print("Processing transcripts...")
    
    # Combine segments into full text
    def combine_segments(row):
        """Combine transcript segments with proper spacing."""
        if 'segments' in row and isinstance(row['segments'], list):
            segments = row['segments']
            if not segments:
                return ""
            
            # Extract text from segments
            texts = [seg.get('text', '').strip() for seg in segments if seg.get('text')]
            
            if not texts:
                return ""
            
            # Join segments with spacing
            # Add period if segment doesn't end with punctuation
            combined = []
            for text in texts:
                text = text.strip()
                if text:
                    # Add period if missing end punctuation
                    if text[-1] not in '.!?':
                        text = text + '.'
                    combined.append(text)
            
            return ' '.join(combined)
        
        elif 'transcript' in row and isinstance(row['transcript'], str):
            return row['transcript']
        
        return ""
    
    print("  Combining transcript segments...")
    df['transcript_text'] = df.apply(combine_segments, axis=1)
    
    # Clean text for LIWC
    print("  Cleaning text (removing emojis, control chars)...")
    df['transcript_text'] = df['transcript_text'].apply(clean_text_for_liwc)
    
    # Keep only video_id and transcript_text
    df_transcripts_final = df[['video_id', 'transcript_text']].copy()
    
    # Create dataframe for all videos (including those without transcripts)
    print("  Aligning with metadata video_ids...")
    df_all_videos = pd.DataFrame({'video_id': valid_video_ids})
    df_transcripts_final = df_all_videos.merge(
        df_transcripts_final, 
        on='video_id', 
        how='left'
    )
    
    # Fill missing transcripts with empty string
    df_transcripts_final['transcript_text'] = df_transcripts_final['transcript_text'].fillna('')
    
    # Ensure string dtype
    df_transcripts_final['transcript_text'] = df_transcripts_final['transcript_text'].astype(str)
    
    # Count empty transcripts
    empty_count = (df_transcripts_final['transcript_text'] == '').sum()
    short_count = (df_transcripts_final['transcript_text'].str.split().str.len() < 10).sum()
    
    print(f"\n✓ Transcript file ready: {len(df_transcripts_final)} videos")
    print(f"  Videos with empty transcripts: {empty_count}")
    print(f"  Videos with < 10 words: {short_count}")
    print(f"  Videos with transcripts: {len(df_transcripts_final) - empty_count}")
    
    # Show sample
    non_empty = df_transcripts_final[df_transcripts_final['transcript_text'] != '']
    if len(non_empty) > 0:
        sample = non_empty.iloc[0]['transcript_text']
        print(f"\nSample transcript (first 200 chars):")
        print(f"  {sample[:200]}...")
    
    return df_transcripts_final


# ============================================================================
# STEP 4 — COMMENTS FILE
# ============================================================================

def create_comments_file(df_comments, valid_video_ids):
    """
    Create comments file for LIWC analysis.
    
    Columns:
    - video_id
    - comments_text
    
    Returns:
        DataFrame ready for export
    """
    print("\n" + "=" * 80)
    print("STEP 4 — CREATE COMMENTS FILE")
    print("=" * 80 + "\n")
    
    df = df_comments.copy()
    
    print(f"Processing {len(df)} comments...")
    
    # Clean comment text
    print("  Cleaning comment text...")
    df['comment_cleaned'] = df['comment_text'].apply(clean_text_for_liwc)
    
    # Remove empty comments
    df = df[df['comment_cleaned'] != ''].copy()
    print(f"  Valid comments after cleaning: {len(df)}")
    
    # Aggregate comments per video
    print("  Aggregating comments per video...")
    
    def aggregate_comments(group):
        """Combine all comments for a video with proper separation."""
        comments = group['comment_cleaned'].tolist()
        
        # Add period to each comment if missing
        processed = []
        for comment in comments:
            comment = comment.strip()
            if comment:
                if comment[-1] not in '.!?':
                    comment = comment + '.'
                processed.append(comment)
        
        # Join with space
        return ' '.join(processed)
    
    df_aggregated = df.groupby('video_id').apply(
        aggregate_comments
    ).reset_index()
    df_aggregated.columns = ['video_id', 'comments_text']
    
    # Create dataframe for all videos (including those without comments)
    print("  Aligning with metadata video_ids...")
    df_all_videos = pd.DataFrame({'video_id': valid_video_ids})
    df_comments_final = df_all_videos.merge(
        df_aggregated,
        on='video_id',
        how='left'
    )
    
    # Fill missing comments with empty string
    df_comments_final['comments_text'] = df_comments_final['comments_text'].fillna('')
    
    # Ensure string dtype
    df_comments_final['comments_text'] = df_comments_final['comments_text'].astype(str)
    
    # Count empty comments
    empty_count = (df_comments_final['comments_text'] == '').sum()
    
    print(f"\n✓ Comments file ready: {len(df_comments_final)} videos")
    print(f"  Videos with no comments: {empty_count}")
    print(f"  Videos with comments: {len(df_comments_final) - empty_count}")
    
    # Show sample
    non_empty = df_comments_final[df_comments_final['comments_text'] != '']
    if len(non_empty) > 0:
        sample = non_empty.iloc[0]['comments_text']
        print(f"\nSample comments (first 200 chars):")
        print(f"  {sample[:200]}...")
    
    return df_comments_final


# ============================================================================
# STEP 5 — FINAL VALIDATION
# ============================================================================

def validate_alignment(df_metadata, df_transcripts, df_comments):
    """
    Validate that all three files have aligned video_ids.
    
    Args:
        df_metadata: Metadata dataframe
        df_transcripts: Transcripts dataframe
        df_comments: Comments dataframe
        
    Raises:
        ValueError: If video_ids are not aligned
    """
    print("\n" + "=" * 80)
    print("STEP 5 — FINAL VALIDATION")
    print("=" * 80 + "\n")
    
    print("Validating video_id alignment...")
    
    ids_metadata = set(df_metadata['video_id'])
    ids_transcripts = set(df_transcripts['video_id'])
    ids_comments = set(df_comments['video_id'])
    
    # Check alignment
    if ids_metadata != ids_transcripts or ids_metadata != ids_comments:
        raise ValueError("video_id mismatch across files!")
    
    print("  ✓ All three files have identical video_ids")
    
    # Check for duplicates
    for name, df in [('metadata', df_metadata), 
                     ('transcripts', df_transcripts), 
                     ('comments', df_comments)]:
        dupes = df['video_id'].duplicated().sum()
        if dupes > 0:
            raise ValueError(f"Duplicate video_ids found in {name}: {dupes}")
    
    print("  ✓ No duplicate video_ids found")
    
    # Print summary
    print(f"\n{'='*80}")
    print("DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal videos: {len(df_metadata)}")
    print(f"\nTranscript coverage:")
    print(f"  Videos with transcripts: {(df_transcripts['transcript_text'] != '').sum()}")
    print(f"  Videos without transcripts: {(df_transcripts['transcript_text'] == '').sum()}")
    print(f"\nComment coverage:")
    print(f"  Videos with comments: {(df_comments['comments_text'] != '').sum()}")
    print(f"  Videos without comments: {(df_comments['comments_text'] == '').sum()}")
    
    print("\n✓ Validation passed")


# ============================================================================
# STEP 6 — EXPORT FILES
# ============================================================================

def export_files(df_metadata, df_transcripts, df_comments, output_dir='output'):
    """
    Export all three files to Excel.
    
    Args:
        df_metadata: Metadata dataframe
        df_transcripts: Transcripts dataframe
        df_comments: Comments dataframe
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("STEP 6 — EXPORT FILES")
    print("=" * 80 + "\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    files = [
        (df_metadata, 'youtube_metadata.xlsx'),
        (df_transcripts, 'youtube_transcripts_for_liwc.xlsx'),
        (df_comments, 'youtube_comments_for_liwc.xlsx')
    ]
    
    for df, filename in files:
        filepath = output_path / filename
        print(f"Exporting {filename}...")
        df.to_excel(filepath, index=False, engine='openpyxl')
        
        # Get file size
        size_kb = filepath.stat().st_size / 1024
        print(f"  ✓ {len(df)} rows, {len(df.columns)} columns, {size_kb:.2f} KB")
    
    print(f"\n✓ All files exported to: {output_path.absolute()}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main pipeline to create LIWC-ready files.
    """
    print("\n" + "=" * 80)
    print("LIWC DATA PREPARATION")
    print("YouTube Engagement Analysis")
    print("=" * 80)
    print("\nPreparing data for LIWC-22 analysis")
    print("Professor will compute all linguistic features\n")
    
    # File paths
    videos_path = 'output/videos.jsonl'
    transcripts_path = 'output/transcripts.jsonl'
    comments_path = 'output/comments.jsonl'
    output_dir = 'output'
    
    try:
        # Step 1: Load data
        df_videos, df_transcripts, df_comments = load_all_data(
            videos_path, transcripts_path, comments_path
        )
        
        # Step 2: Create metadata file
        df_metadata = create_metadata_file(df_videos)
        
        # Get valid video IDs from metadata
        valid_video_ids = df_metadata['video_id'].tolist()
        
        # Step 3: Create transcript file
        df_transcripts_final = create_transcript_file(df_transcripts, valid_video_ids)
        
        # Step 4: Create comments file
        df_comments_final = create_comments_file(df_comments, valid_video_ids)
        
        # Step 5: Validate alignment
        validate_alignment(df_metadata, df_transcripts_final, df_comments_final)
        
        # Step 6: Export files
        export_files(df_metadata, df_transcripts_final, df_comments_final, output_dir)
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE")
        print("=" * 80)
        print("\nGenerated files:")
        print("  📄 youtube_metadata.xlsx")
        print("  📄 youtube_transcripts_for_liwc.xlsx")
        print("  📄 youtube_comments_for_liwc.xlsx")
        print("\nReady for LIWC-22 analysis!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Please ensure all input files exist in the output/ directory")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
