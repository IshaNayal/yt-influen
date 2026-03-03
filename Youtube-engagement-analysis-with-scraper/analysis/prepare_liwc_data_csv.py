"""
LIWC Data Preparation for YouTube Engagement Analysis (CSV Version)

Prepares three files for LIWC-22 analysis:
1. youtube_metadata.xlsx - Video metrics (Excel)
2. youtube_transcripts_for_liwc.csv - Raw transcript text (CSV - no char limit)
3. youtube_comments_for_liwc.csv - Aggregated comment text (CSV - no char limit)

Fixes Excel 32,767 character truncation by using CSV for text fields.
"""

import json
import re
import os
import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

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
# STEP 2 — REBUILD METADATA
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
    print("STEP 2 — REBUILD METADATA")
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
    df['recency_days'] = df['recency_days'].clip(lower=0)
    
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
# STEP 3 — REBUILD TRANSCRIPTS (CSV VERSION)
# ============================================================================

def create_transcript_file(df_transcripts, valid_video_ids):
    """
    Create transcript file for LIWC analysis (CSV - no character limit).
    
    Columns:
    - video_id
    - transcript_text
    
    Returns:
        DataFrame ready for export with validation statistics
    """
    print("\n" + "=" * 80)
    print("STEP 3 — REBUILD TRANSCRIPTS (CSV VERSION)")
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
    
    # Compute statistics
    print("\n" + "-" * 80)
    print("TRANSCRIPT VALIDATION & STATISTICS")
    print("-" * 80)
    
    lengths = df_transcripts_final['transcript_text'].str.len()
    non_empty = df_transcripts_final[df_transcripts_final['transcript_text'] != '']
    
    print(f"\nTotal videos: {len(df_transcripts_final)}")
    print(f"Videos with transcripts: {len(non_empty)}")
    print(f"Videos without transcripts: {len(df_transcripts_final) - len(non_empty)}")
    
    if len(non_empty) > 0:
        print(f"\nTranscript Character Lengths:")
        print(f"  Max: {lengths.max():,} characters")
        print(f"  Mean: {lengths[lengths > 0].mean():,.2f} characters")
        print(f"  Median: {lengths[lengths > 0].median():,.0f} characters")
        
        # Check for Excel truncation (sanity check)
        truncated = (lengths == 32767).sum()
        if truncated > 0:
            print(f"  ⚠ WARNING: {truncated} transcripts are exactly 32,767 chars (possible truncation)")
        else:
            print(f"  ✓ No transcripts at Excel truncation limit (32,767 chars)")
        
        # Show distribution
        over_32k = (lengths > 32767).sum()
        if over_32k > 0:
            print(f"  ✓ {over_32k} transcripts exceed Excel limit (CSV handles this)")
    
    print(f"\n✓ Transcript file ready for CSV export (NO CHARACTER LIMITS)")
    
    return df_transcripts_final


# ============================================================================
# STEP 4 — REBUILD COMMENTS (CSV VERSION)
# ============================================================================

def create_comments_file(df_comments, valid_video_ids):
    """
    Create comments file for LIWC analysis (CSV - no character limit).
    
    Columns:
    - video_id
    - comments_text
    
    Returns:
        DataFrame ready for export with validation statistics
    """
    print("\n" + "=" * 80)
    print("STEP 4 — REBUILD COMMENTS (CSV VERSION)")
    print("=" * 80 + "\n")
    
    df = df_comments.copy()
    
    print(f"Processing {len(df)} comments...")
    
    # Clean comment text
    print("  Cleaning comment text...")
    df['comment_cleaned'] = df['comment_text'].apply(clean_text_for_liwc)
    
    # Remove empty comments
    df = df[df['comment_cleaned'] != ''].copy()
    print(f"  Valid comments after cleaning: {len(df)}")
    
    # Aggregate comments per video using efficient groupby
    print("  Aggregating comments per video (efficient groupby)...")
    
    def aggregate_comments(comment_series):
        """Combine all comments for a video with proper separation."""
        comments = comment_series.tolist()
        
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
    
    df_aggregated = df.groupby('video_id')['comment_cleaned'].apply(
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
    
    # Compute statistics
    print("\n" + "-" * 80)
    print("COMMENTS VALIDATION & STATISTICS")
    print("-" * 80)
    
    lengths = df_comments_final['comments_text'].str.len()
    non_empty = df_comments_final[df_comments_final['comments_text'] != '']
    
    print(f"\nTotal videos: {len(df_comments_final)}")
    print(f"Videos with comments: {len(non_empty)}")
    print(f"Videos without comments: {len(df_comments_final) - len(non_empty)}")
    
    if len(non_empty) > 0:
        print(f"\nComment Character Lengths:")
        print(f"  Max: {lengths.max():,} characters")
        print(f"  Mean: {lengths[lengths > 0].mean():,.2f} characters")
        print(f"  Median: {lengths[lengths > 0].median():,.0f} characters")
        
        # Check for Excel truncation (sanity check)
        truncated = (lengths == 32767).sum()
        if truncated > 0:
            print(f"  ⚠ WARNING: {truncated} comments are exactly 32,767 chars (possible truncation)")
        else:
            print(f"  ✓ No comments at Excel truncation limit (32,767 chars)")
        
        # Show distribution
        over_32k = (lengths > 32767).sum()
        if over_32k > 0:
            print(f"  ✓ {over_32k} videos have comments exceeding Excel limit")
            print(f"    (CSV handles this - no truncation!)")
    
    print(f"\n✓ Comments file ready for CSV export (NO CHARACTER LIMITS)")
    
    return df_comments_final


# ============================================================================
# STEP 5 — ALIGNMENT CHECK
# ============================================================================

def validate_alignment(df_metadata, df_transcripts, df_comments):
    """
    Validate that all three files have aligned video_ids.
    """
    print("\n" + "=" * 80)
    print("STEP 5 — ALIGNMENT CHECK")
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
    
    print(f"\n  Videos in metadata: {len(df_metadata)}")
    print(f"  Videos in transcripts: {len(df_transcripts)}")
    print(f"  Videos in comments: {len(df_comments)}")
    
    print("\n✓ Alignment validation passed")


# ============================================================================
# STEP 6 — DELETE OLD EXCEL TEXT FILES
# ============================================================================

def delete_old_excel_files(output_dir='output'):
    """
    Delete old Excel versions of transcript and comment files.
    """
    print("\n" + "=" * 80)
    print("STEP 6 — DELETE OLD EXCEL TEXT FILES")
    print("=" * 80 + "\n")
    
    output_path = Path(output_dir)
    
    old_files = [
        'youtube_transcripts_for_liwc.xlsx',
        'youtube_comments_for_liwc.xlsx'
    ]
    
    deleted_count = 0
    
    for filename in old_files:
        filepath = output_path / filename
        if filepath.exists():
            try:
                os.remove(filepath)
                print(f"✓ Deleted: {filename}")
                deleted_count += 1
            except Exception as e:
                print(f"⚠ Warning: Could not delete {filename}: {e}")
        else:
            print(f"  (Not found: {filename})")
    
    if deleted_count > 0:
        print(f"\n✓ Deleted {deleted_count} old Excel file(s)")
    else:
        print("\n  No old Excel files to delete")


# ============================================================================
# STEP 7 — EXPORT FILES
# ============================================================================

def export_files(df_metadata, df_transcripts, df_comments, output_dir='output'):
    """
    Export files:
    - Metadata as Excel
    - Transcripts as CSV
    - Comments as CSV
    """
    print("\n" + "=" * 80)
    print("STEP 7 — EXPORT FILES")
    print("=" * 80 + "\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Export metadata as Excel
    print("Exporting youtube_metadata.xlsx...")
    metadata_path = output_path / 'youtube_metadata.xlsx'
    df_metadata.to_excel(metadata_path, index=False, engine='openpyxl')
    size_kb = metadata_path.stat().st_size / 1024
    print(f"  ✓ {len(df_metadata)} rows, {len(df_metadata.columns)} columns, {size_kb:.2f} KB")
    
    # Export transcripts as CSV with proper quoting
    print("\nExporting youtube_transcripts_for_liwc.csv...")
    transcripts_path = output_path / 'youtube_transcripts_for_liwc.csv'
    df_transcripts.to_csv(
        transcripts_path,
        index=False,
        encoding='utf-8',
        quoting=csv.QUOTE_ALL
    )
    size_kb = transcripts_path.stat().st_size / 1024
    print(f"  ✓ {len(df_transcripts)} rows, {len(df_transcripts.columns)} columns, {size_kb:.2f} KB")
    
    # Export comments as CSV with proper quoting
    print("\nExporting youtube_comments_for_liwc.csv...")
    comments_path = output_path / 'youtube_comments_for_liwc.csv'
    df_comments.to_csv(
        comments_path,
        index=False,
        encoding='utf-8',
        quoting=csv.QUOTE_ALL
    )
    size_kb = comments_path.stat().st_size / 1024
    print(f"  ✓ {len(df_comments)} rows, {len(df_comments.columns)} columns, {size_kb:.2f} KB")
    
    print(f"\n✓ All files exported to: {output_path.absolute()}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main pipeline to create LIWC-ready files (CSV version - no truncation).
    """
    print("\n" + "=" * 80)
    print("LIWC DATA PREPARATION (CSV VERSION)")
    print("YouTube Engagement Analysis")
    print("=" * 80)
    print("\nFixes Excel 32,767 character truncation")
    print("Uses CSV for text fields (unlimited character length)\n")
    
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
        
        # Step 3: Create transcript file (CSV)
        df_transcripts_final = create_transcript_file(df_transcripts, valid_video_ids)
        
        # Step 4: Create comments file (CSV)
        df_comments_final = create_comments_file(df_comments, valid_video_ids)
        
        # Step 5: Validate alignment
        validate_alignment(df_metadata, df_transcripts_final, df_comments_final)
        
        # Step 6: Delete old Excel files
        delete_old_excel_files(output_dir)
        
        # Step 7: Export files
        export_files(df_metadata, df_transcripts_final, df_comments_final, output_dir)
        
        # Final summary
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE - FINAL SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Total videos processed: {len(df_metadata)}")
        
        transcript_lengths = df_transcripts_final['transcript_text'].str.len()
        comment_lengths = df_comments_final['comments_text'].str.len()
        
        print(f"\n📝 Transcript Statistics:")
        print(f"   Max length: {transcript_lengths.max():,} characters")
        print(f"   Videos with transcripts: {(transcript_lengths > 0).sum()}")
        
        print(f"\n💬 Comment Statistics:")
        print(f"   Max length: {comment_lengths.max():,} characters")
        print(f"   Videos with comments: {(comment_lengths > 0).sum()}")
        
        print(f"\n✅ CSV Format Benefits:")
        print(f"   ✓ No 32,767 character limit")
        print(f"   ✓ All text preserved without truncation")
        print(f"   ✓ Proper UTF-8 encoding")
        print(f"   ✓ Safe handling of commas and quotes")
        
        print(f"\n📄 Generated Files:")
        print(f"   • youtube_metadata.xlsx (numeric data)")
        print(f"   • youtube_transcripts_for_liwc.csv (text data)")
        print(f"   • youtube_comments_for_liwc.csv (text data)")
        
        print(f"\n🗑  Old Excel text files deleted")
        
        print(f"\n✅ Ready for LIWC-22 analysis!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Please ensure all input files exist in the output/ directory")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
