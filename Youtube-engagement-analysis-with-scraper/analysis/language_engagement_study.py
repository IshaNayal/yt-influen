"""
YouTube Transcript Language vs Engagement Study

Computational social science analysis to understand how language used in 
YouTube video transcripts affects engagement, controlling for exposure, 
time, and channel differences.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_jsonl(filepath):
    """Load JSONL file into list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
    return data


def parse_iso_duration(duration_str):
    """
    Parse ISO 8601 duration (e.g., PT10M30S) to seconds.
    
    Args:
        duration_str: ISO 8601 duration string
        
    Returns:
        Total seconds as integer
    """
    if not duration_str or not isinstance(duration_str, str):
        return 0
    
    # Pattern: PT(hours)H(minutes)M(seconds)S
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    
    if not match:
        return 0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds


def clean_text(text):
    """
    Clean transcript text.
    
    Args:
        text: Raw transcript text
        
    Returns:
        Cleaned text (lowercase, no punctuation/numbers, normalized whitespace)
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_and_merge_data(videos_path, transcripts_path):
    """
    Load videos and transcripts, merge them.
    
    Returns:
        Merged DataFrame with video metadata and full transcript text
    """
    print("=" * 60)
    print("1️⃣ LOADING AND MERGING DATA")
    print("=" * 60)
    
    # Load videos
    print(f"Loading videos from {videos_path}...")
    videos = load_jsonl(videos_path)
    df_videos = pd.DataFrame(videos)
    
    # Rename columns to snake_case for consistency
    df_videos = df_videos.rename(columns={
        'viewCount': 'view_count',
        'likeCount': 'like_count',
        'commentCount': 'comment_count'
    })
    
    print(f"  Loaded {len(df_videos)} videos")
    
    # Load transcripts
    print(f"Loading transcripts from {transcripts_path}...")
    transcripts = load_jsonl(transcripts_path)
    df_transcripts = pd.DataFrame(transcripts)
    print(f"  Loaded {len(df_transcripts)} transcripts")
    
    # Combine transcript segments into full text
    print("Combining transcript segments...")
    def combine_segments(row):
        if row['transcript_source'] == 'failed' or not row.get('segments'):
            return None
        segments = row['segments']
        return ' '.join([seg['text'] for seg in segments if seg.get('text')])
    
    df_transcripts['full_text'] = df_transcripts.apply(combine_segments, axis=1)
    
    # Keep only videos with valid transcripts
    df_transcripts = df_transcripts[df_transcripts['full_text'].notna()].copy()
    print(f"  Valid transcripts: {len(df_transcripts)}")
    
    # Merge
    print("Merging datasets...")
    df = df_videos.merge(df_transcripts[['video_id', 'full_text']], on='video_id', how='inner')
    print(f"  Merged dataset: {len(df)} videos with transcripts")
    
    # Drop rows with missing critical fields or zero views
    initial_count = len(df)
    df = df.dropna(subset=['view_count', 'like_count', 'comment_count'])
    df = df[df['view_count'] > 0].copy()
    print(f"  After removing zero views: {len(df)} videos")
    print(f"  Dropped {initial_count - len(df)} videos\n")
    
    return df


def stratified_sampling(df, channel_col='channel_id', frac=0.5):
    """
    Sample videos stratified by channel to prevent large channels from dominating.
    If channel_col doesn't exist, perform simple random sampling instead.
    
    Args:
        df: Input DataFrame
        channel_col: Column name for channel grouping
        frac: Fraction to sample (0.5 = 50%)
        
    Returns:
        Sampled DataFrame
    """
    print("=" * 60)
    print("2️⃣ STRATIFIED SAMPLING")
    print("=" * 60)
    
    if channel_col not in df.columns:
        print(f"Warning: '{channel_col}' column not found. Performing simple random sampling instead.")
        print(f"Sampling {frac*100:.0f}% of all videos...")
        df_sampled = df.sample(frac=frac, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"Total videos after sampling: {len(df_sampled)}\n")
        return df_sampled
    
    print(f"Sampling {frac*100:.0f}% of videos from each channel...")
    
    # Show distribution before
    print("\nVideos per channel (before sampling):")
    print(df[channel_col].value_counts().to_string())
    
    df_sampled = df.groupby(channel_col, group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=RANDOM_SEED)
    ).reset_index(drop=True)
    
    print(f"\nTotal videos after sampling: {len(df_sampled)}")
    print("\nVideos per channel (after sampling):")
    print(df_sampled[channel_col].value_counts().to_string())
    print()
    
    return df_sampled


def preprocess_text(df):
    """
    Clean transcript text.
    
    Args:
        df: DataFrame with 'full_text' column
        
    Returns:
        DataFrame with added 'cleaned_text' column
    """
    print("=" * 60)
    print("3️⃣ TEXT PREPROCESSING")
    print("=" * 60)
    
    print("Cleaning transcript text...")
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    # Show example
    print("\nExample cleaned text (first 200 chars):")
    print(df['cleaned_text'].iloc[0][:200])
    print()
    
    return df


def extract_bigram_features(df, max_features=100, min_df=5):
    """
    Extract bigram features using CountVectorizer.
    
    Args:
        df: DataFrame with 'cleaned_text' column
        max_features: Maximum number of bigrams to extract
        min_df: Minimum document frequency (ignore rare phrases)
        
    Returns:
        Tuple of (DataFrame, sparse matrix of bigram features, vectorizer)
    """
    print("=" * 60)
    print("4️⃣ EXTRACTING BIGRAM FEATURES")
    print("=" * 60)
    
    print(f"Extracting top {max_features} bigrams...")
    print(f"  Min document frequency: {min_df}")
    
    vectorizer = CountVectorizer(
        ngram_range=(2, 2),  # Only bigrams
        max_features=max_features,
        min_df=min_df,
        stop_words='english',
        lowercase=True
    )
    
    bigram_matrix = vectorizer.fit_transform(df['cleaned_text'])
    
    print(f"  Bigram matrix shape: {bigram_matrix.shape}")
    print(f"  Number of bigrams: {len(vectorizer.get_feature_names_out())}")
    
    # Show some example bigrams
    print("\nExample bigrams (first 10):")
    for i, bigram in enumerate(vectorizer.get_feature_names_out()[:10]):
        print(f"  {i+1}. '{bigram}'")
    print()
    
    return df, bigram_matrix, vectorizer


def create_control_variables(df):
    """
    Create control variables for regression.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with added control variables
    """
    print("=" * 60)
    print("5️⃣ CREATING CONTROL VARIABLES")
    print("=" * 60)
    
    # Log views
    print("Creating log_views...")
    df['log_views'] = np.log1p(df['view_count'])
    
    # Video age in days
    print("Calculating video age in days...")
    df['published_at'] = pd.to_datetime(df['published_at'])
    now = pd.Timestamp.now(tz='UTC')  # Make timezone-aware
    df['video_age_days'] = (now - df['published_at']).dt.days
    
    # Video duration in seconds
    print("Parsing video duration...")
    df['video_duration_seconds'] = df['duration'].apply(parse_iso_duration)
    
    # Summary statistics
    print("\nControl variable statistics:")
    print(df[['log_views', 'video_age_days', 'video_duration_seconds']].describe())
    print()
    
    return df


def create_target_variables(df):
    """
    Create engagement rate targets.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with added target variables
    """
    print("=" * 60)
    print("6️⃣ CREATING TARGET VARIABLES")
    print("=" * 60)
    
    print("Computing engagement rates...")
    
    # Engagement rates
    df['like_rate'] = df['like_count'] / df['view_count']
    df['comment_rate'] = df['comment_count'] / df['view_count']
    
    # Log-transformed targets (more stable for regression)
    df['log_like_rate'] = np.log1p(df['like_rate'])
    df['log_comment_rate'] = np.log1p(df['comment_rate'])
    
    print("\nTarget variable statistics:")
    print(df[['like_rate', 'comment_rate', 'log_like_rate', 'log_comment_rate']].describe())
    print()
    
    return df


def prepare_features(df, bigram_matrix):
    """
    Prepare feature matrices for modeling.
    
    Args:
        df: DataFrame with control variables
        bigram_matrix: Sparse matrix of bigram features
        
    Returns:
        Tuple of (controls_only matrix, controls+bigrams matrix, channel encoder or None)
    """
    print("=" * 60)
    print("7️⃣ PREPARING FEATURE MATRICES")
    print("=" * 60)
    
    # Numeric control variables
    numeric_controls = df[['log_views', 'video_age_days', 'video_duration_seconds']].values
    print(f"Numeric controls shape: {numeric_controls.shape}")
    
    # One-hot encode channel_id if it exists
    if 'channel_id' in df.columns:
        print("One-hot encoding channel_id...")
        encoder = OneHotEncoder(sparse_output=True, drop='first')  # Drop first to avoid multicollinearity
        channel_encoded = encoder.fit_transform(df[['channel_id']])
        print(f"Channel encoding shape: {channel_encoded.shape}")
        
        # Combine controls
        X_controls = hstack([numeric_controls, channel_encoded])
    else:
        print("No channel_id column found - using only numeric controls")
        encoder = None
        X_controls = numeric_controls
    
    print(f"Controls matrix shape: {X_controls.shape}")
    
    # Combine controls + bigrams
    X_full = hstack([X_controls, bigram_matrix])
    print(f"Full feature matrix shape: {X_full.shape}")
    print()
    
    return X_controls, X_full, encoder


def train_and_evaluate_models(X_controls, X_full, y, target_name):
    """
    Train and evaluate both control-only and full models.
    
    Args:
        X_controls: Control variables only
        X_full: Controls + bigram features
        y: Target variable
        target_name: Name of target for reporting
        
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*60}")
    print(f"MODEL: {target_name}")
    print(f"{'='*60}")
    
    # Train/test split
    print("Splitting data (80/20 train/test)...")
    X_controls_train, X_controls_test, y_train, y_test = train_test_split(
        X_controls, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    X_full_train, X_full_test = train_test_split(
        X_full, test_size=0.2, random_state=RANDOM_SEED
    )[0:2]
    
    print(f"  Train size: {len(y_train)}")
    print(f"  Test size: {len(y_test)}")
    
    # Model A: Controls Only
    print("\n📊 Model A: Controls Only")
    model_controls = LinearRegression()
    model_controls.fit(X_controls_train, y_train)
    y_pred_controls = model_controls.predict(X_controls_test)
    r2_controls = r2_score(y_test, y_pred_controls)
    print(f"  R² score: {r2_controls:.4f}")
    
    # Model B: Controls + Bigrams (Linear Regression)
    print("\n📊 Model B: Controls + Bigrams (Linear Regression)")
    model_full_lr = LinearRegression()
    model_full_lr.fit(X_full_train, y_train)
    y_pred_full_lr = model_full_lr.predict(X_full_test)
    r2_full_lr = r2_score(y_test, y_pred_full_lr)
    print(f"  R² score: {r2_full_lr:.4f}")
    
    # Model B: Controls + Bigrams (Lasso Regression for feature selection)
    print("\n📊 Model B: Controls + Bigrams (Lasso Regression)")
    model_full_lasso = Lasso(alpha=0.001, random_state=RANDOM_SEED, max_iter=5000)
    model_full_lasso.fit(X_full_train, y_train)
    y_pred_full_lasso = model_full_lasso.predict(X_full_test)
    r2_full_lasso = r2_score(y_test, y_pred_full_lasso)
    print(f"  R² score: {r2_full_lasso:.4f}")
    
    # Improvement
    improvement_lr = r2_full_lr - r2_controls
    improvement_lasso = r2_full_lasso - r2_controls
    print(f"\n✨ Language Effect (Linear): {improvement_lr:.4f} ({improvement_lr/r2_controls*100:.1f}% improvement)")
    print(f"✨ Language Effect (Lasso): {improvement_lasso:.4f} ({improvement_lasso/r2_controls*100:.1f}% improvement)")
    
    return {
        'target': target_name,
        'r2_controls': r2_controls,
        'r2_full_lr': r2_full_lr,
        'r2_full_lasso': r2_full_lasso,
        'improvement_lr': improvement_lr,
        'improvement_lasso': improvement_lasso,
        'model_controls': model_controls,
        'model_full_lr': model_full_lr,
        'model_full_lasso': model_full_lasso
    }


def interpret_coefficients(model, vectorizer, X_controls_shape, top_n=15):
    """
    Extract and interpret bigram coefficients.
    
    Args:
        model: Fitted regression model
        vectorizer: CountVectorizer used for bigrams
        X_controls_shape: Shape of control variable matrix (to skip those coefficients)
        top_n: Number of top bigrams to show
        
    Returns:
        DataFrame of bigram coefficients
    """
    print("\n" + "=" * 60)
    print("9️⃣ INTERPRETING BIGRAM COEFFICIENTS")
    print("=" * 60)
    
    # Get coefficients
    coefs = model.coef_
    
    # Extract bigram coefficients (skip control variable coefficients)
    n_controls = X_controls_shape[1]
    bigram_coefs = coefs[n_controls:]
    
    # Get bigram names
    bigram_names = vectorizer.get_feature_names_out()
    
    # Create dataframe
    df_coefs = pd.DataFrame({
        'bigram': bigram_names,
        'coefficient': bigram_coefs
    })
    
    # Sort by coefficient
    df_coefs = df_coefs.sort_values('coefficient', ascending=False).reset_index(drop=True)
    
    print(f"\nTop {top_n} bigrams that INCREASE engagement:")
    print(df_coefs.head(top_n).to_string(index=False))
    
    print(f"\nTop {top_n} bigrams that DECREASE engagement:")
    print(df_coefs.tail(top_n).to_string(index=False))
    
    return df_coefs


def save_results(results_dict, df_coefs_like, df_coefs_comment, output_dir='output'):
    """
    Save model results and coefficient tables.
    
    Args:
        results_dict: Dictionary containing model results
        df_coefs_like: Bigram coefficients for like rate
        df_coefs_comment: Bigram coefficients for comment rate
        output_dir: Directory to save results
    """
    print("\n" + "=" * 60)
    print("🔟 SAVING RESULTS")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save bigram coefficients
    like_coef_path = output_path / 'bigram_coefficients_like_rate.csv'
    df_coefs_like.to_csv(like_coef_path, index=False)
    print(f"✓ Saved like rate bigram coefficients to {like_coef_path}")
    
    comment_coef_path = output_path / 'bigram_coefficients_comment_rate.csv'
    df_coefs_comment.to_csv(comment_coef_path, index=False)
    print(f"✓ Saved comment rate bigram coefficients to {comment_coef_path}")
    
    # Save model performance metrics
    metrics_path = output_path / 'model_performance.txt'
    with open(metrics_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("YOUTUBE LANGUAGE-ENGAGEMENT STUDY - MODEL PERFORMANCE\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results_dict:
            f.write(f"{result['target']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"R² (Controls Only):                {result['r2_controls']:.4f}\n")
            f.write(f"R² (Controls + Bigrams - Linear):  {result['r2_full_lr']:.4f}\n")
            f.write(f"R² (Controls + Bigrams - Lasso):   {result['r2_full_lasso']:.4f}\n")
            f.write(f"Language Effect (Linear):           {result['improvement_lr']:.4f}\n")
            f.write(f"Language Effect (Lasso):            {result['improvement_lasso']:.4f}\n")
            f.write(f"Relative Improvement (Linear):      {result['improvement_lr']/result['r2_controls']*100:.1f}%\n")
            f.write(f"Relative Improvement (Lasso):       {result['improvement_lasso']/result['r2_controls']*100:.1f}%\n")
            f.write("\n")
    
    print(f"✓ Saved model performance metrics to {metrics_path}")
    print()


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 60)
    print("YOUTUBE TRANSCRIPT LANGUAGE vs ENGAGEMENT STUDY")
    print("=" * 60 + "\n")
    
    # Paths
    videos_path = 'output/videos.jsonl'
    transcripts_path = 'output/transcripts.jsonl'
    
    # 1. Load and merge
    df = load_and_merge_data(videos_path, transcripts_path)
    
    # 2. Stratified sampling
    df = stratified_sampling(df, channel_col='channel_id', frac=0.5)
    
    # 3. Text preprocessing
    df = preprocess_text(df)
    
    # 4. Extract bigram features
    df, bigram_matrix, vectorizer = extract_bigram_features(df, max_features=100, min_df=5)
    
    # 5. Create control variables
    df = create_control_variables(df)
    
    # 6. Create target variables
    df = create_target_variables(df)
    
    # 7. Prepare feature matrices
    X_controls, X_full, encoder = prepare_features(df, bigram_matrix)
    
    # 8. Train and evaluate models
    print("\n" + "=" * 60)
    print("8️⃣ TRAINING AND EVALUATING MODELS")
    print("=" * 60)
    
    results = []
    
    # Like rate model
    result_like = train_and_evaluate_models(
        X_controls, X_full, 
        df['log_like_rate'].values,
        "LOG LIKE RATE"
    )
    results.append(result_like)
    
    # Comment rate model
    result_comment = train_and_evaluate_models(
        X_controls, X_full,
        df['log_comment_rate'].values,
        "LOG COMMENT RATE"
    )
    results.append(result_comment)
    
    # 9. Interpret coefficients (use Lasso model for clearer interpretation)
    df_coefs_like = interpret_coefficients(
        result_like['model_full_lasso'], 
        vectorizer, 
        X_controls.shape,
        top_n=15
    )
    
    df_coefs_comment = interpret_coefficients(
        result_comment['model_full_lasso'],
        vectorizer,
        X_controls.shape,
        top_n=15
    )
    
    # 10. Save results
    save_results(results, df_coefs_like, df_coefs_comment)
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nResults saved to output/ directory:")
    print("  - bigram_coefficients_like_rate.csv")
    print("  - bigram_coefficients_comment_rate.csv")
    print("  - model_performance.txt")
    print()


if __name__ == '__main__':
    main()
