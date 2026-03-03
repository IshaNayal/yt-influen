"""
Bigram Feature Regression: Per-Video Top 100 Approach

For each video:
1. Extract its top 100 bigrams
2. Create binary/count features for presence of each bigram
3. Regress on engagement controlling for video characteristics
4. Identify which bigrams are most predictive
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, hstack
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from tqdm import tqdm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_jsonl(filepath):
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def parse_iso_duration(duration_str):
    """Parse ISO 8601 duration to seconds."""
    if not duration_str or not isinstance(duration_str, str):
        return 0
    
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    
    if not match:
        return 0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds


def clean_text(text):
    """Clean transcript text."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_top_bigrams_per_video(text, top_n=100):
    """
    Extract top N bigrams from a single transcript.
    
    Returns:
        List of (bigram, count) tuples
    """
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
                'those', 'it', 'its', 'my', 'your', 'their', 'our'}
    
    words = text.split()
    bigrams = []
    
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        if len(w1) >= 2 and len(w2) >= 2 and w1 not in stopwords and w2 not in stopwords:
            bigrams.append(f"{w1} {w2}")
    
    if not bigrams:
        return []
    
    bigram_counts = Counter(bigrams)
    return bigram_counts.most_common(top_n)


def load_and_prepare_data():
    """Load and merge data."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # Load videos
    videos = load_jsonl('output/videos.jsonl')
    df_videos = pd.DataFrame(videos)
    df_videos = df_videos.rename(columns={
        'viewCount': 'view_count',
        'likeCount': 'like_count',
        'commentCount': 'comment_count'
    })
    print(f"Loaded {len(df_videos)} videos")
    
    # Load transcripts
    transcripts = load_jsonl('output/transcripts.jsonl')
    df_transcripts = pd.DataFrame(transcripts)
    
    # Combine segments
    def combine_segments(row):
        if 'segments' in row and isinstance(row['segments'], list):
            return ' '.join([seg.get('text', '') for seg in row['segments']])
        elif 'transcript' in row:
            return row['transcript']
        return ''
    
    df_transcripts['full_text'] = df_transcripts.apply(combine_segments, axis=1)
    df_transcripts = df_transcripts[df_transcripts['full_text'].notna()].copy()
    print(f"Valid transcripts: {len(df_transcripts)}")
    
    # Merge
    df = df_videos.merge(df_transcripts[['video_id', 'full_text']], on='video_id', how='inner')
    
    # Check column names
    if 'view_count' not in df.columns and 'viewCount' in df.columns:
        df = df.rename(columns={'viewCount': 'view_count', 'likeCount': 'like_count', 
                                'commentCount': 'comment_count'})
    
    df = df.dropna(subset=['view_count', 'like_count', 'comment_count'])
    df = df[df['view_count'] > 0].copy()
    print(f"Final dataset: {len(df)} videos\n")
    
    return df


def stratified_sampling(df, frac=0.5):
    """Sample videos stratified by channel."""
    print("=" * 60)
    print("STRATIFIED SAMPLING")
    print("=" * 60)
    
    if 'channel_id' in df.columns:
        print(f"Sampling {frac*100:.0f}% from each channel...")
        df_sampled = df.groupby('channel_id', group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=RANDOM_SEED)
        ).reset_index(drop=True)
    else:
        print(f"Sampling {frac*100:.0f}% of all videos...")
        df_sampled = df.sample(frac=frac, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"Sampled videos: {len(df_sampled)}\n")
    return df_sampled


def extract_bigram_features(df, top_n=100, min_videos=5):
    """
    Extract top N bigrams from each video, create feature matrix.
    
    Args:
        df: DataFrame with transcripts
        top_n: Number of top bigrams to extract per video
        min_videos: Minimum number of videos a bigram must appear in
        
    Returns:
        Sparse matrix of bigram features, list of bigram names
    """
    print("=" * 60)
    print("EXTRACTING BIGRAM FEATURES")
    print("=" * 60)
    
    print(f"Extracting top {top_n} bigrams per video...")
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    # Step 1: Extract top bigrams for each video
    all_video_bigrams = []
    bigram_vocabulary = Counter()
    
    print("Phase 1: Extracting bigrams from each transcript...")
    for idx, text in enumerate(tqdm(df['cleaned_text'], desc="Processing transcripts")):
        top_bigrams = extract_top_bigrams_per_video(text, top_n)
        video_bigrams = {bigram: count for bigram, count in top_bigrams}
        all_video_bigrams.append(video_bigrams)
        
        # Track which bigrams appear in which videos
        for bigram in video_bigrams:
            bigram_vocabulary[bigram] += 1
    
    # Step 2: Filter bigrams by min_videos threshold
    print(f"\nPhase 2: Filtering bigrams (min {min_videos} videos)...")
    valid_bigrams = {bigram for bigram, video_count in bigram_vocabulary.items() 
                     if video_count >= min_videos}
    
    print(f"Total unique bigrams found: {len(bigram_vocabulary)}")
    print(f"Bigrams appearing in {min_videos}+ videos: {len(valid_bigrams)}")
    
    # Step 3: Create bigram-to-index mapping
    bigram_to_idx = {bigram: idx for idx, bigram in enumerate(sorted(valid_bigrams))}
    bigram_names = sorted(valid_bigrams)
    
    # Step 4: Build sparse feature matrix
    print(f"\nPhase 3: Building feature matrix...")
    n_samples = len(df)
    n_features = len(bigram_to_idx)
    
    X_bigrams = lil_matrix((n_samples, n_features), dtype=np.float32)
    
    for video_idx, video_bigrams in enumerate(tqdm(all_video_bigrams, desc="Creating matrix")):
        for bigram, count in video_bigrams.items():
            if bigram in bigram_to_idx:
                bigram_idx = bigram_to_idx[bigram]
                X_bigrams[video_idx, bigram_idx] = count
    
    X_bigrams = X_bigrams.tocsr()
    
    print(f"\nFeature matrix shape: {X_bigrams.shape}")
    print(f"Number of bigram features: {n_features}")
    
    # Show most common bigrams
    print(f"\nTop 20 most frequent bigrams across all videos:")
    most_common = bigram_vocabulary.most_common(20)
    for i, (bigram, count) in enumerate(most_common, 1):
        if bigram in valid_bigrams:
            print(f"  {i:2d}. '{bigram}' - appears in {count} videos")
    print()
    
    return X_bigrams, bigram_names


def create_control_variables(df):
    """Create control variables."""
    print("=" * 60)
    print("CREATING CONTROL VARIABLES")
    print("=" * 60)
    
    df['log_views'] = np.log1p(df['view_count'])
    df['published_at'] = pd.to_datetime(df['published_at'])
    now = pd.Timestamp.now(tz='UTC')
    df['video_age_days'] = (now - df['published_at']).dt.days
    df['video_duration_seconds'] = df['duration'].apply(parse_iso_duration)
    
    print("Control variables created:")
    print(df[['log_views', 'video_age_days', 'video_duration_seconds']].describe())
    print()
    
    return df


def create_target_variables(df):
    """Create engagement rate targets."""
    print("=" * 60)
    print("CREATING TARGET VARIABLES")
    print("=" * 60)
    
    df['like_rate'] = df['like_count'] / df['view_count']
    df['comment_rate'] = df['comment_count'] / df['view_count']
    df['log_like_rate'] = np.log1p(df['like_rate'])
    df['log_comment_rate'] = np.log1p(df['comment_rate'])
    
    print(df[['like_rate', 'comment_rate']].describe())
    print()
    
    return df


def prepare_features(df, X_bigrams):
    """Prepare feature matrices."""
    print("=" * 60)
    print("PREPARING FEATURE MATRICES")
    print("=" * 60)
    
    numeric_controls = df[['log_views', 'video_age_days', 'video_duration_seconds']].values
    
    if 'channel_id' in df.columns:
        encoder = OneHotEncoder(sparse_output=True, drop='first')
        channel_encoded = encoder.fit_transform(df[['channel_id']])
        X_controls = hstack([numeric_controls, channel_encoded])
    else:
        X_controls = numeric_controls
    
    X_full = hstack([X_controls, X_bigrams])
    
    print(f"Controls matrix: {X_controls.shape}")
    print(f"Full matrix (controls + bigrams): {X_full.shape}\n")
    
    return X_controls, X_full


def train_models(X_controls, X_full, y, target_name):
    """Train and evaluate models."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {target_name}")
    print(f"{'='*60}")
    
    X_controls_train, X_controls_test, y_train, y_test = train_test_split(
        X_controls, y, test_size=0.2, random_state=RANDOM_SEED
    )
    X_full_train, X_full_test = train_test_split(
        X_full, test_size=0.2, random_state=RANDOM_SEED
    )[0:2]
    
    # Controls only
    model_controls = LinearRegression()
    model_controls.fit(X_controls_train, y_train)
    r2_controls = r2_score(y_test, model_controls.predict(X_controls_test))
    print(f"Controls only R²: {r2_controls:.4f}")
    
    # Full model (Lasso)
    model_lasso = Lasso(alpha=0.0001, random_state=RANDOM_SEED, max_iter=10000)
    model_lasso.fit(X_full_train, y_train)
    r2_lasso = r2_score(y_test, model_lasso.predict(X_full_test))
    print(f"Full model (Lasso) R²: {r2_lasso:.4f}")
    print(f"Improvement: {r2_lasso - r2_controls:.4f}\n")
    
    return model_controls, model_lasso, r2_controls, r2_lasso


def extract_top_bigrams(model, bigram_names, X_controls_shape, top_n=50):
    """Extract top bigrams by coefficient."""
    n_controls = X_controls_shape[1]
    bigram_coefs = model.coef_[n_controls:]
    
    df_coefs = pd.DataFrame({
        'bigram': bigram_names,
        'coefficient': bigram_coefs
    })
    
    df_coefs = df_coefs[df_coefs['coefficient'] != 0]  # Remove zeros
    df_coefs = df_coefs.reindex(df_coefs['coefficient'].abs().sort_values(ascending=False).index)
    
    return df_coefs


def main():
    """Main pipeline."""
    print("\n" + "=" * 60)
    print("BIGRAM FEATURE REGRESSION: PER-VIDEO TOP 100")
    print("=" * 60 + "\n")
    
    # Load data
    df = load_and_prepare_data()
    
    # Sample
    df = stratified_sampling(df, frac=0.5)
    
    # Extract bigram features
    X_bigrams, bigram_names = extract_bigram_features(df, top_n=100, min_videos=5)
    
    # Create variables
    df = create_control_variables(df)
    df = create_target_variables(df)
    
    # Prepare features
    X_controls, X_full = prepare_features(df, X_bigrams)
    
    # Train models
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    _, model_like, r2_like_ctrl, r2_like_full = train_models(
        X_controls, X_full, df['log_like_rate'].values, "LIKE RATE"
    )
    
    _, model_comment, r2_comm_ctrl, r2_comm_full = train_models(
        X_controls, X_full, df['log_comment_rate'].values, "COMMENT RATE"
    )
    
    # Extract top bigrams
    print("\n" + "=" * 60)
    print("TOP BIGRAMS BY COEFFICIENT")
    print("=" * 60)
    
    df_like_coefs = extract_top_bigrams(model_like, bigram_names, X_controls.shape)
    df_comment_coefs = extract_top_bigrams(model_comment, bigram_names, X_controls.shape)
    
    print("\n📊 LIKE RATE - Top 30 Bigrams:")
    print(df_like_coefs.head(30).to_string(index=False))
    
    print("\n\n📊 COMMENT RATE - Top 30 Bigrams:")
    print(df_comment_coefs.head(30).to_string(index=False))
    
    # Combine coefficients
    df_combined = df_like_coefs.merge(
        df_comment_coefs, on='bigram', how='outer', suffixes=('_like', '_comment')
    ).fillna(0)
    
    df_combined['total_impact'] = (
        df_combined['coefficient_like'].abs() + 
        df_combined['coefficient_comment'].abs()
    )
    df_combined = df_combined.sort_values('total_impact', ascending=False)
    
    print("\n\n📊 COMBINED - Top 50 Most Impactful Bigrams:")
    print(df_combined.head(50).to_string(index=False))
    
    # Save results
    output_dir = Path('output')
    df_like_coefs.to_csv(output_dir / 'pervideo_bigrams_like_coefficients.csv', index=False)
    df_comment_coefs.to_csv(output_dir / 'pervideo_bigrams_comment_coefficients.csv', index=False)
    df_combined.to_csv(output_dir / 'pervideo_bigrams_combined.csv', index=False)
    
    print("\n\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nResults saved to output/:")
    print("  - pervideo_bigrams_like_coefficients.csv")
    print("  - pervideo_bigrams_comment_coefficients.csv")
    print("  - pervideo_bigrams_combined.csv")
    print(f"\nTotal bigrams analyzed: {len(bigram_names)}")
    print(f"Non-zero coefficients (like): {len(df_like_coefs)}")
    print(f"Non-zero coefficients (comment): {len(df_comment_coefs)}")


if __name__ == '__main__':
    main()
