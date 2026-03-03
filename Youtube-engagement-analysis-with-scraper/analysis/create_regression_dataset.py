"""
YouTube Video Engagement Analysis - Regression Dataset Creation

Replicates methodology from:
"Leveraging Machine Learning and Generative AI for Content Engagement: 
Drivers of YouTube Video Success"

Creates one-row-per-video Excel dataset with all regression variables.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from textblob import TextBlob
import textstat
from tqdm import tqdm

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# WORD LISTS (Paper Methodology)
# ============================================================================

AFFECT_WORDS = [
    "happy", "sad", "angry", "excited", "disappointed", "amazing", "terrible", "great",
    "love", "hate", "awesome", "awful", "fantastic", "horrible", "good", "bad",
    "pleasant", "unpleasant", "positive", "negative", "frustrated", "delighted",
    "annoyed", "thrilled", "upset", "satisfied", "unsatisfied", "fun", "boring",
    "incredible", "disgusting", "surprising", "shocking", "fear", "scared", "worried",
    "confident", "proud", "embarrassed", "joy", "pain", "hope", "hopeless"
]

SOCIAL_DISTANCE_WORDS = [
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves"
]

FUTURE_FOCUS_WORDS = [
    "will", "would", "shall", "going", "gonna", "future", "soon", "upcoming",
    "next", "tomorrow", "later", "eventually", "plan", "plans", "planned",
    "expect", "expects", "expected", "anticipate", "anticipated", "launch",
    "releasing", "release", "coming", "ahead", "forecast", "predict", "prediction"
]

PRESENT_FOCUS_WORDS = [
    "is", "are", "am", "was", "were", "be", "being", "been",
    "now", "today", "currently", "present", "right", "here",
    "do", "does", "doing", "have", "has", "having"
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_jsonl(filepath):
    """Load JSONL file into list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
    return data


def clean_text(text):
    """
    Clean text according to paper methodology:
    - Lowercase
    - Remove punctuation
    - Remove numbers
    - Remove extra whitespace
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize(text):
    """Tokenize text by splitting on whitespace."""
    if not text:
        return []
    return text.split()


def parse_iso_duration(duration_str):
    """
    Parse ISO 8601 duration (e.g., PT10M30S) to total minutes.
    
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
    
    total_minutes = hours * 60 + minutes + seconds / 60.0
    
    return total_minutes


def compute_word_percentage(tokens, word_list):
    """
    Compute percentage of tokens that appear in word_list.
    
    Args:
        tokens: List of word tokens
        word_list: List of target words
        
    Returns:
        Percentage (0-100)
    """
    if not tokens:
        return 0.0
    
    word_set = set(word_list)
    count = sum(1 for token in tokens if token in word_set)
    percentage = (count / len(tokens)) * 100.0
    
    return percentage


# ============================================================================
# STEP 1 — LOAD AND MERGE DATA
# ============================================================================

def load_and_merge_data(videos_path, transcripts_path, comments_path):
    """
    Load all data sources and merge into one dataframe.
    
    Returns:
        DataFrame with videos, transcripts, and aggregated comments
    """
    print("=" * 80)
    print("STEP 1 — LOAD AND MERGE DATA")
    print("=" * 80)
    
    # Load videos
    print(f"\nLoading videos from {videos_path}...")
    videos = load_jsonl(videos_path)
    df_videos = pd.DataFrame(videos)
    
    # Standardize column names
    if 'viewCount' in df_videos.columns:
        df_videos = df_videos.rename(columns={
            'viewCount': 'view_count',
            'likeCount': 'like_count',
            'commentCount': 'comment_count'
        })
    
    print(f"  ✓ Loaded {len(df_videos)} videos")
    
    # Load transcripts
    print(f"\nLoading transcripts from {transcripts_path}...")
    transcripts = load_jsonl(transcripts_path)
    df_transcripts = pd.DataFrame(transcripts)
    
    # Combine transcript segments into full text
    def combine_segments(row):
        if 'segments' in row and isinstance(row['segments'], list):
            return ' '.join([seg.get('text', '') for seg in row['segments'] if seg.get('text')])
        elif 'transcript' in row and isinstance(row['transcript'], str):
            return row['transcript']
        return ''
    
    df_transcripts['full_text'] = df_transcripts.apply(combine_segments, axis=1)
    df_transcripts = df_transcripts[df_transcripts['full_text'].notna()].copy()
    df_transcripts = df_transcripts[df_transcripts['full_text'].str.strip() != ''].copy()
    
    print(f"  ✓ Loaded {len(df_transcripts)} transcripts")
    
    # Load comments
    print(f"\nLoading comments from {comments_path}...")
    comments = load_jsonl(comments_path)
    df_comments = pd.DataFrame(comments)
    print(f"  ✓ Loaded {len(df_comments)} comments")
    
    # Merge transcripts to videos
    print("\nMerging transcripts to videos...")
    df = df_videos.merge(
        df_transcripts[['video_id', 'full_text']], 
        on='video_id', 
        how='inner'
    )
    print(f"  ✓ {len(df)} videos with transcripts")
    
    # Store comments separately for later aggregation
    df_comments_processed = df_comments[['video_id', 'comment_text']].copy()
    
    initial_count = len(df)
    df = df.dropna(subset=['view_count', 'like_count'])
    df = df[df['view_count'] > 0].copy()
    print(f"  ✓ {len(df)} videos after removing invalid data")
    print(f"  (Dropped {initial_count - len(df)} videos with missing/zero views)")
    
    return df, df_comments_processed


# ============================================================================
# STEP 2 — BASIC CLEANING
# ============================================================================

def clean_transcript_and_comments(df, df_comments):
    """
    Clean and tokenize transcript and comments.
    
    Returns:
        Updated dataframes with cleaned text and tokens
    """
    print("\n" + "=" * 80)
    print("STEP 2 — BASIC CLEANING")
    print("=" * 80)
    
    print("\nCleaning transcripts...")
    df['transcript_cleaned'] = df['full_text'].apply(clean_text)
    df['transcript_tokens'] = df['transcript_cleaned'].apply(tokenize)
    print(f"  ✓ Cleaned {len(df)} transcripts")
    
    print("\nCleaning comments...")
    df_comments['comment_cleaned'] = df_comments['comment_text'].apply(clean_text)
    df_comments['comment_tokens'] = df_comments['comment_cleaned'].apply(tokenize)
    df_comments['comment_wordcount'] = df_comments['comment_tokens'].apply(len)
    print(f"  ✓ Cleaned {len(df_comments)} comments")
    
    return df, df_comments


# ============================================================================
# STEP 3 — DEPENDENT VARIABLE
# ============================================================================

def extract_dependent_variable(df):
    """Extract dependent variable (like_count)."""
    print("\n" + "=" * 80)
    print("STEP 3 — DEPENDENT VARIABLE")
    print("=" * 80)
    
    print("\nDependent variable: like_count")
    print(f"  Mean: {df['like_count'].mean():.2f}")
    print(f"  Median: {df['like_count'].median():.2f}")
    print(f"  Std: {df['like_count'].std():.2f}")
    print(f"  Min: {df['like_count'].min():.0f}")
    print(f"  Max: {df['like_count'].max():.0f}")
    
    return df


# ============================================================================
# STEP 4 — CONTROL VARIABLES
# ============================================================================

def create_control_variables(df):
    """
    Create control variables:
    - view_count
    - recency_days
    - duration_minutes
    - duration_dummy
    """
    print("\n" + "=" * 80)
    print("STEP 4 — CONTROL VARIABLES")
    print("=" * 80)
    
    # view_count already exists
    print("\n✓ view_count (already present)")
    
    # recency_days
    print("\nComputing recency_days...")
    df['published_at'] = pd.to_datetime(df['published_at'])
    today = pd.Timestamp.now(tz=timezone.utc)
    df['recency_days'] = (today - df['published_at']).dt.days
    df['recency_days'] = df['recency_days'].clip(lower=0)  # Ensure non-negative
    print(f"  Mean recency: {df['recency_days'].mean():.2f} days")
    
    # duration_minutes
    print("\nParsing duration_minutes...")
    df['duration_minutes'] = df['duration'].apply(parse_iso_duration)
    print(f"  Mean duration: {df['duration_minutes'].mean():.2f} minutes")
    
    # duration_dummy (1 if > 8 minutes, else 0)
    print("\nCreating duration_dummy (1 if > 8 minutes)...")
    df['duration_dummy'] = (df['duration_minutes'] > 8).astype(int)
    print(f"  Videos > 8 min: {df['duration_dummy'].sum()} ({df['duration_dummy'].mean()*100:.1f}%)")
    
    print("\n" + "-" * 80)
    print("Control Variables Summary:")
    print(df[['view_count', 'recency_days', 'duration_minutes', 'duration_dummy']].describe())
    
    return df


# ============================================================================
# STEP 5 — TRANSCRIPT VARIABLES
# ============================================================================

def compute_transcript_variables(df):
    """
    Compute all transcript-based variables:
    1. Polarity and Subjectivity (TextBlob)
    2. Affect
    3. Social Distance
    4. Future Focus
    5. Readability
    """
    print("\n" + "=" * 80)
    print("STEP 5 — TRANSCRIPT VARIABLES")
    print("=" * 80)
    
    # Word count
    print("\nComputing total_word_count_transcript...")
    df['total_word_count_transcript'] = df['transcript_tokens'].apply(len)
    print(f"  Mean word count: {df['total_word_count_transcript'].mean():.2f}")
    
    # 1️⃣ Polarity and Subjectivity
    print("\n1️⃣ Computing polarity and subjectivity (TextBlob)...")
    
    def get_sentiment(text):
        if not text:
            return 0.0, 0.0
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except:
            return 0.0, 0.0
    
    print("  Processing sentiment (this may take a moment)...")
    sentiments = []
    for text in tqdm(df['transcript_cleaned'], desc="  Analyzing sentiment"):
        sentiments.append(get_sentiment(text))
    
    df['polarity_transcript'] = [s[0] for s in sentiments]
    df['subjectivity_transcript'] = [s[1] for s in sentiments]
    
    print(f"  Mean polarity: {df['polarity_transcript'].mean():.4f}")
    print(f"  Mean subjectivity: {df['subjectivity_transcript'].mean():.4f}")
    
    # 2️⃣ Affect
    print("\n2️⃣ Computing affect_transcript...")
    df['affect_transcript'] = df['transcript_tokens'].apply(
        lambda tokens: compute_word_percentage(tokens, AFFECT_WORDS)
    )
    print(f"  Mean affect: {df['affect_transcript'].mean():.4f}%")
    
    # 3️⃣ Social Distance
    print("\n3️⃣ Computing social_distance_transcript...")
    df['social_distance_transcript'] = df['transcript_tokens'].apply(
        lambda tokens: compute_word_percentage(tokens, SOCIAL_DISTANCE_WORDS)
    )
    print(f"  Mean social distance: {df['social_distance_transcript'].mean():.4f}%")
    
    # 4️⃣ Future Focus
    print("\n4️⃣ Computing future_focus...")
    df['future_focus'] = df['transcript_tokens'].apply(
        lambda tokens: compute_word_percentage(tokens, FUTURE_FOCUS_WORDS)
    )
    print(f"  Mean future focus: {df['future_focus'].mean():.4f}%")
    
    # 5️⃣ Readability
    print("\n5️⃣ Computing readability_score (Flesch Reading Ease)...")
    
    def get_readability(text):
        if not text or len(text) < 10:
            return 0.0
        try:
            return textstat.flesch_reading_ease(text)
        except:
            return 0.0
    
    readability_scores = []
    for text in tqdm(df['transcript_cleaned'], desc="  Computing readability"):
        readability_scores.append(get_readability(text))
    
    df['readability_score'] = readability_scores
    print(f"  Mean readability: {df['readability_score'].mean():.2f}")
    
    print("\n" + "-" * 80)
    print("Transcript Variables Summary:")
    print(df[[
        'polarity_transcript', 'subjectivity_transcript', 'affect_transcript',
        'social_distance_transcript', 'future_focus', 'readability_score'
    ]].describe())
    
    return df


# ============================================================================
# STEP 6 — COMMENT VARIABLES
# ============================================================================

def compute_comment_variables(df, df_comments):
    """
    Compute aggregated comment variables per video:
    1. avg_comment_wordcount
    2. affect_comment
    3. social_distance_comment
    4. present_focus_comment
    """
    print("\n" + "=" * 80)
    print("STEP 6 — COMMENT VARIABLES (Aggregate Per Video)")
    print("=" * 80)
    
    # Filter out empty comments
    df_comments = df_comments[df_comments['comment_wordcount'] > 0].copy()
    print(f"\nValid comments (wordcount > 0): {len(df_comments)}")
    
    # 1️⃣ Average word count per video
    print("\n1️⃣ Computing avg_comment_wordcount...")
    avg_wordcount = df_comments.groupby('video_id')['comment_wordcount'].mean().reset_index()
    avg_wordcount.columns = ['video_id', 'avg_comment_wordcount']
    
    # 2️⃣ Affect (Comments)
    print("\n2️⃣ Computing affect_comment...")
    df_comments['affect_pct'] = df_comments['comment_tokens'].apply(
        lambda tokens: compute_word_percentage(tokens, AFFECT_WORDS)
    )
    affect_comment = df_comments.groupby('video_id')['affect_pct'].mean().reset_index()
    affect_comment.columns = ['video_id', 'affect_comment']
    
    # 3️⃣ Social Distance (Comments)
    print("\n3️⃣ Computing social_distance_comment...")
    df_comments['social_distance_pct'] = df_comments['comment_tokens'].apply(
        lambda tokens: compute_word_percentage(tokens, SOCIAL_DISTANCE_WORDS)
    )
    social_distance_comment = df_comments.groupby('video_id')['social_distance_pct'].mean().reset_index()
    social_distance_comment.columns = ['video_id', 'social_distance_comment']
    
    # 4️⃣ Present Focus (Comments)
    print("\n4️⃣ Computing present_focus_comment...")
    df_comments['present_focus_pct'] = df_comments['comment_tokens'].apply(
        lambda tokens: compute_word_percentage(tokens, PRESENT_FOCUS_WORDS)
    )
    present_focus_comment = df_comments.groupby('video_id')['present_focus_pct'].mean().reset_index()
    present_focus_comment.columns = ['video_id', 'present_focus_comment']
    
    # Merge all comment variables to main dataframe
    print("\nMerging comment variables to main dataset...")
    df = df.merge(avg_wordcount, on='video_id', how='left')
    df = df.merge(affect_comment, on='video_id', how='left')
    df = df.merge(social_distance_comment, on='video_id', how='left')
    df = df.merge(present_focus_comment, on='video_id', how='left')
    
    # Fill NaN with 0 for videos without comments
    comment_cols = ['avg_comment_wordcount', 'affect_comment', 
                    'social_distance_comment', 'present_focus_comment']
    df[comment_cols] = df[comment_cols].fillna(0)
    
    print(f"  Videos with comments: {(df['avg_comment_wordcount'] > 0).sum()}")
    print(f"  Videos without comments: {(df['avg_comment_wordcount'] == 0).sum()}")
    
    print("\n" + "-" * 80)
    print("Comment Variables Summary:")
    print(df[comment_cols].describe())
    
    return df


# ============================================================================
# STEP 7 — FINAL DATASET STRUCTURE
# ============================================================================

def create_final_dataset(df):
    """
    Create final regression-ready dataset with required columns only.
    """
    print("\n" + "=" * 80)
    print("STEP 7 — FINAL DATASET STRUCTURE")
    print("=" * 80)
    
    # Select only required columns in specified order
    final_columns = [
        'video_id',
        'like_count',
        'view_count',
        'recency_days',
        'duration_minutes',
        'duration_dummy',
        'polarity_transcript',
        'subjectivity_transcript',
        'affect_transcript',
        'social_distance_transcript',
        'future_focus',
        'readability_score',
        'avg_comment_wordcount',
        'affect_comment',
        'social_distance_comment',
        'present_focus_comment'
    ]
    
    df_final = df[final_columns].copy()
    
    print(f"\nFinal dataset shape: {df_final.shape}")
    print(f"  Rows (videos): {len(df_final)}")
    print(f"  Columns: {len(df_final.columns)}")
    
    # Check for missing values
    print("\nMissing values check:")
    missing = df_final.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values")
    else:
        print(missing[missing > 0])
    
    # Data types check
    print("\nData types:")
    print(df_final.dtypes)
    
    # Summary statistics
    print("\n" + "-" * 80)
    print("FINAL DATASET SUMMARY STATISTICS:")
    print("-" * 80)
    print(df_final.describe())
    
    return df_final


# ============================================================================
# STEP 8 — EXPORT
# ============================================================================

def export_to_excel(df_final, output_path):
    """
    Export final dataset to Excel.
    """
    print("\n" + "=" * 80)
    print("STEP 8 — EXPORT")
    print("=" * 80)
    
    print(f"\nExporting to {output_path}...")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True)
    
    # Export to Excel
    df_final.to_excel(output_path, index=False, engine='openpyxl')
    
    print(f"  ✓ Successfully exported {len(df_final)} rows to Excel")
    print(f"  ✓ File location: {output_path}")
    
    # File size
    file_size = Path(output_path).stat().st_size / 1024  # KB
    print(f"  ✓ File size: {file_size:.2f} KB")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main pipeline to create regression-ready dataset.
    """
    print("\n" + "=" * 80)
    print("YOUTUBE VIDEO ENGAGEMENT ANALYSIS")
    print("REGRESSION DATASET CREATION")
    print("=" * 80)
    print("\nReplicating methodology from:")
    print("'Leveraging Machine Learning and Generative AI for Content Engagement:")
    print(" Drivers of YouTube Video Success'\n")
    
    # File paths
    videos_path = 'output/videos.jsonl'
    transcripts_path = 'output/transcripts.jsonl'
    comments_path = 'output/comments.jsonl'
    output_path = 'output/youtube_regression_dataset.xlsx'
    
    # STEP 1: Load and merge
    df, df_comments = load_and_merge_data(videos_path, transcripts_path, comments_path)
    
    # STEP 2: Basic cleaning
    df, df_comments = clean_transcript_and_comments(df, df_comments)
    
    # STEP 3: Dependent variable
    df = extract_dependent_variable(df)
    
    # STEP 4: Control variables
    df = create_control_variables(df)
    
    # STEP 5: Transcript variables
    df = compute_transcript_variables(df)
    
    # STEP 6: Comment variables
    df = compute_comment_variables(df, df_comments)
    
    # STEP 7: Final dataset structure
    df_final = create_final_dataset(df)
    
    # STEP 8: Export
    export_to_excel(df_final, output_path)
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nRegression-ready dataset created:")
    print(f"  📊 {len(df_final)} videos")
    print(f"  📊 {len(df_final.columns)} variables")
    print(f"  📄 {output_path}")
    print("\nReady for regression analysis!")


if __name__ == '__main__':
    main()
