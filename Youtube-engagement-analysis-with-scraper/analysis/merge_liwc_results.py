"""
Final Regression Dataset Creation

Combines:
1. Metadata (views, likes, recency, duration)
2. LIWC-22 features from transcripts
3. Sentiment and emotion features from comments (TextBlob + NRCLex)

Creates final regression-ready dataset for analysis.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from nrclex import NRCLex
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("FINAL REGRESSION DATASET CREATION")
print("=" * 80)
print("\nCombining metadata + LIWC features + comment sentiment\n")

# ============================================================================
# STEP 1 — LOAD ALL DATA
# ============================================================================

print("=" * 80)
print("STEP 1 — LOAD ALL DATA")
print("=" * 80 + "\n")

# Load metadata
print("Loading metadata...")
df_metadata = pd.read_excel('output/youtube_metadata.xlsx')
print(f"  ✓ Loaded {len(df_metadata)} videos")

# Load LIWC results
print("\nLoading LIWC-22 results...")
df_liwc = pd.read_excel('LIWC-22 Results - youtube_transcripts_for_liwc - LIWC Analysis.xlsx')
print(f"  ✓ Loaded {len(df_liwc)} rows")
print(f"  Columns: {len(df_liwc.columns)}")

# Show LIWC columns
print(f"\n  LIWC variables available:")
liwc_cols = [col for col in df_liwc.columns if col not in ['Filename', 'Segment', 'video_id']]
print(f"  {len(liwc_cols)} linguistic features")
if len(liwc_cols) <= 20:
    for col in liwc_cols:
        print(f"    - {col}")
else:
    print(f"    First 20: {', '.join(liwc_cols[:20])}")
    print(f"    ... and {len(liwc_cols) - 20} more")

# Load comments for sentiment analysis
print("\nLoading comments...")
df_comments = pd.read_csv('output/youtube_comments_for_liwc.csv')
print(f"  ✓ Loaded {len(df_comments)} videos")

# ============================================================================
# STEP 2 — PROCESS LIWC RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2 — PROCESS LIWC RESULTS")
print("=" * 80 + "\n")

# Identify video_id column (might be 'Filename' or already 'video_id')
if 'video_id' not in df_liwc.columns and 'Filename' in df_liwc.columns:
    print("Renaming 'Filename' to 'video_id'...")
    df_liwc = df_liwc.rename(columns={'Filename': 'video_id'})

# Drop unnecessary columns
cols_to_drop = ['Segment', 'WC', 'Analytic', 'Clout', 'Authentic', 'Tone']
cols_to_drop = [col for col in cols_to_drop if col in df_liwc.columns]
if cols_to_drop:
    print(f"Dropping columns: {', '.join(cols_to_drop)}")
    df_liwc = df_liwc.drop(columns=cols_to_drop)

# Keep only video_id and linguistic features
liwc_feature_cols = [col for col in df_liwc.columns if col != 'video_id']
print(f"\nRetaining {len(liwc_feature_cols)} LIWC features")

# ============================================================================
# STEP 3 — COMPUTE COMMENT SENTIMENT AND EMOTIONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3 — COMPUTE COMMENT SENTIMENT AND EMOTIONS")
print("=" * 80 + "\n")

print("Computing sentiment and emotion for comments...")

def get_comment_sentiment_emotions(text):
    """
    Compute sentiment and emotion features for comments.
    
    Returns:
        tuple: (polarity, subjectivity, fear, anger, anticipation, trust,
                surprise, positive, negative, sadness, disgust, joy)
    """
    if not text or pd.isna(text) or text == '':
        return (0.0,) * 12
    
    try:
        # Basic sentiment with TextBlob
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Emotion analysis with NRCLex
        emotion = NRCLex(str(text))
        emotions = emotion.raw_emotion_scores
        
        # Extract 10 emotion scores (NRC emotions)
        fear = emotions.get('fear', 0)
        anger = emotions.get('anger', 0)
        anticipation = emotions.get('anticipation', 0)
        trust = emotions.get('trust', 0)
        surprise = emotions.get('surprise', 0)
        positive = emotions.get('positive', 0)
        negative = emotions.get('negative', 0)
        sadness = emotions.get('sadness', 0)
        disgust = emotions.get('disgust', 0)
        joy = emotions.get('joy', 0)
        
        return (polarity, subjectivity, fear, anger, anticipation, trust,
                surprise, positive, negative, sadness, disgust, joy)
    except:
        return (0.0,) * 12

print("  Analyzing comment sentiment and emotions (this may take a few minutes)...")
results = []
for text in tqdm(df_comments['comments_text'], desc="  Processing"):
    results.append(get_comment_sentiment_emotions(text))

# Unpack results
df_comments['comment_polarity'] = [r[0] for r in results]
df_comments['comment_subjectivity'] = [r[1] for r in results]
df_comments['emotion_fear'] = [r[2] for r in results]
df_comments['emotion_anger'] = [r[3] for r in results]
df_comments['emotion_anticipation'] = [r[4] for r in results]
df_comments['emotion_trust'] = [r[5] for r in results]
df_comments['emotion_surprise'] = [r[6] for r in results]
df_comments['emotion_positive'] = [r[7] for r in results]
df_comments['emotion_negative'] = [r[8] for r in results]
df_comments['emotion_sadness'] = [r[9] for r in results]
df_comments['emotion_disgust'] = [r[10] for r in results]
df_comments['emotion_joy'] = [r[11] for r in results]

# Compute comment length
df_comments['comment_length'] = df_comments['comments_text'].fillna('').astype(str).str.len()
df_comments['has_comments'] = (df_comments['comment_length'] > 0).astype(int)

print(f"\n✓ Comment sentiment and emotions computed")
print(f"  Mean polarity: {df_comments['comment_polarity'].mean():.4f}")
print(f"  Mean subjectivity: {df_comments['comment_subjectivity'].mean():.4f}")
print(f"  Mean emotion scores:")
print(f"    Joy: {df_comments['emotion_joy'].mean():.2f}")
print(f"    Sadness: {df_comments['emotion_sadness'].mean():.2f}")
print(f"    Anger: {df_comments['emotion_anger'].mean():.2f}")
print(f"    Fear: {df_comments['emotion_fear'].mean():.2f}")
print(f"    Surprise: {df_comments['emotion_surprise'].mean():.2f}")
print(f"    Anticipation: {df_comments['emotion_anticipation'].mean():.2f}")
print(f"    Trust: {df_comments['emotion_trust'].mean():.2f}")
print(f"    Disgust: {df_comments['emotion_disgust'].mean():.2f}")
print(f"  Videos with comments: {df_comments['has_comments'].sum()}")

# Keep all features
emotion_features = ['comment_polarity', 'comment_subjectivity', 
                   'emotion_fear', 'emotion_anger', 'emotion_anticipation', 
                   'emotion_trust', 'emotion_surprise', 'emotion_positive', 
                   'emotion_negative', 'emotion_sadness', 'emotion_disgust', 
                   'emotion_joy', 'comment_length', 'has_comments']
df_comments_features = df_comments[['video_id'] + emotion_features].copy()

# ============================================================================
# STEP 4 — MERGE ALL DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4 — MERGE ALL DATA")
print("=" * 80 + "\n")

print("Merging metadata + LIWC + comments...")

# Start with metadata
df_final = df_metadata.copy()
print(f"  Starting with metadata: {len(df_final)} videos")

# Merge LIWC features
df_final = df_final.merge(df_liwc, on='video_id', how='left')
print(f"  After merging LIWC: {len(df_final)} videos, {len(df_final.columns)} columns")

# Merge comment features
df_final = df_final.merge(df_comments_features, on='video_id', how='left')
print(f"  After merging comments: {len(df_final)} videos, {len(df_final.columns)} columns")

# Fill any missing values
numeric_cols = df_final.select_dtypes(include=[np.number]).columns
df_final[numeric_cols] = df_final[numeric_cols].fillna(0)

print(f"\n✓ Merge complete")
print(f"  Total videos: {len(df_final)}")
print(f"  Total features: {len(df_final.columns) - 1}")  # Excluding video_id

# ============================================================================
# STEP 5 — DATA VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5 — DATA VALIDATION")
print("=" * 80 + "\n")

# Check for missing values
missing_counts = df_final.isnull().sum()
if missing_counts.sum() > 0:
    print("⚠ Missing values detected:")
    print(missing_counts[missing_counts > 0])
else:
    print("✓ No missing values")

# Check data types
print(f"\nData types:")
print(f"  Numeric columns: {len(df_final.select_dtypes(include=[np.number]).columns)}")
print(f"  Object columns: {len(df_final.select_dtypes(include=['object']).columns)}")

# Summary statistics for key variables
print(f"\nKey variable statistics:")
key_vars = ['like_count', 'view_count', 'comment_polarity', 'comment_subjectivity']
key_vars = [v for v in key_vars if v in df_final.columns]
print(df_final[key_vars].describe())

# ============================================================================
# STEP 6 — EXPORT FINAL DATASET
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6 — EXPORT FINAL DATASET")
print("=" * 80 + "\n")

output_file = 'output/final_regression_dataset.xlsx'
print(f"Exporting to {output_file}...")

df_final.to_excel(output_file, index=False, engine='openpyxl')

import os
file_size = os.path.getsize(output_file) / 1024
print(f"  ✓ Exported successfully")
print(f"  Rows: {len(df_final)}")
print(f"  Columns: {len(df_final.columns)}")
print(f"  File size: {file_size:.2f} KB")

# ============================================================================
# STEP 7 — NEXT STEPS SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ DATASET READY - NEXT STEPS")
print("=" * 80)

print(f"\n📊 Final Dataset: {output_file}")
print(f"   - {len(df_final)} videos")
print(f"   - {len(df_final.columns)} variables")

print(f"\n📋 Variable Categories:")
print(f"   1. Dependent Variable: like_count")
print(f"   2. Control Variables: view_count, recency_days, duration_minutes, duration_dummy")
print(f"   3. LIWC Features ({len(liwc_feature_cols)}): {', '.join(liwc_feature_cols[:5])}...")
print(f"   4. Comment Sentiment: comment_polarity, comment_subjectivity")
print(f"   5. Comment Emotions (10): joy, sadness, anger, fear, surprise, disgust, anticipation, trust, positive, negative")
print(f"   6. Comment Meta: comment_length, has_comments")

print(f"\n🎯 NEXT STEPS:")
print(f"\n   OPTION A: Run Regression in Python")
print(f"   ----------------------------------")
print(f"   1. Load final_regression_dataset.xlsx")
print(f"   2. Run OLS regression with statsmodels")
print(f"   3. Identify significant predictors")
print(f"   4. Report R², coefficients, p-values")
print(f"\n   Command: python analysis/run_regression.py")

print(f"\n   OPTION B: Run Regression in Excel/SPSS/R")
print(f"   -----------------------------------------")
print(f"   1. Open final_regression_dataset.xlsx")
print(f"   2. Set like_count as dependent variable")
print(f"   3. Add controls: view_count, recency_days, duration_minutes, duration_dummy")
print(f"   4. Add LIWC features as independent variables")
print(f"   5. Add comment features as independent variables")
print(f"   6. Run multiple regression")

print(f"\n   OPTION C: Create Visualizations")
print(f"   --------------------------------")
print(f"   1. Correlation heatmap of top predictors")
print(f"   2. Scatter plots: LIWC features vs engagement")
print(f"   3. Feature importance plots")
print(f"\n   Command: python analysis/visualize_results.py")

print(f"\n📝 Recommended Analysis Sequence:")
print(f"   1. Run descriptive statistics")
print(f"   2. Check correlations between predictors")
print(f"   3. Run baseline model (controls only)")
print(f"   4. Run full model (controls + LIWC + comments)")
print(f"   5. Compare R² improvement")
print(f"   6. Identify top significant LIWC features")
print(f"   7. Visualize results")

print(f"\n" + "=" * 80)
print("Would you like me to:")
print("  A) Create a Python regression script?")
print("  B) Create a visualization script?")
print("  C) Generate summary statistics report?")
print("=" * 80 + "\n")
