"""
Quick Test: Do Normalized Emotions Improve R²?

Fast comparison of raw vs normalized emotion scores.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("QUICK TEST: NORMALIZED vs RAW EMOTIONS")
print("=" * 80 + "\n")

# Load data
df = pd.read_excel('output/final_regression_dataset.xlsx')
df_clean = df[df['transcript_text'].notna()].copy()

# Estimate comment count
df_clean['estimated_comments'] = (df_clean['comment_length'] / 100).clip(lower=1)

# Normalize emotions
emotions = ['emotion_fear', 'emotion_anger', 'emotion_anticipation', 
            'emotion_trust', 'emotion_surprise', 'emotion_positive', 
            'emotion_negative', 'emotion_sadness', 'emotion_disgust', 'emotion_joy']

for col in emotions:
    df_clean[f'{col}_norm'] = df_clean[col] / df_clean['estimated_comments']

# Check correlations
print("Correlation Check:")
print(f"  emotion_trust ↔ comment_length:      {df_clean['emotion_trust'].corr(df_clean['comment_length']):.3f}")
print(f"  emotion_trust_norm ↔ comment_length: {df_clean['emotion_trust_norm'].corr(df_clean['comment_length']):.3f}")
print(f"  like_count ↔ emotion_trust:          {df_clean['like_count'].corr(df_clean['emotion_trust']):.3f}")
print(f"  like_count ↔ emotion_trust_norm:     {df_clean['like_count'].corr(df_clean['emotion_trust_norm']):.3f}")

# Prepare features
controls = ['view_count', 'recency_days', 'duration_minutes', 'duration_dummy']
liwc = [c for c in df_clean.columns if c not in 
        ['video_id', 'transcript_text', 'like_count'] + controls
        and not c.startswith('comment_') and not c.startswith('emotion_')
        and c != 'has_comments' and '_norm' not in c and 'estimated' not in c]

# Raw emotions model
raw_comments = emotions + ['comment_polarity', 'comment_subjectivity', 'comment_length', 'has_comments']
X_raw = df_clean[controls + liwc + raw_comments].fillna(0)

# Normalized emotions model  
norm_emotions = [f'{e}_norm' for e in emotions]
norm_comments = norm_emotions + ['comment_polarity', 'comment_subjectivity', 'has_comments']
X_norm = df_clean[controls + liwc + norm_comments].fillna(0)

y = np.log1p(df_clean['like_count'])

# Fit models
scaler = StandardScaler()

print(f"\n" + "=" * 80)
print("MODEL 1: Raw Emotions (Current)")
print("-" * 80)
X_raw_scaled = scaler.fit_transform(X_raw)
X_raw_sm = sm.add_constant(X_raw_scaled)
m1 = sm.OLS(y, X_raw_sm).fit()
print(f"R² = {m1.rsquared:.4f}")
print(f"Adjusted R² = {m1.rsquared_adj:.4f}")

print(f"\n" + "=" * 80)
print("MODEL 2: Normalized Emotions (Per Comment)")
print("-" * 80)
X_norm_scaled = scaler.fit_transform(X_norm)
X_norm_sm = sm.add_constant(X_norm_scaled)
m2 = sm.OLS(y, X_norm_sm).fit()
print(f"R² = {m2.rsquared:.4f}")
print(f"Adjusted R² = {m2.rsquared_adj:.4f}")
print(f"Change: {m2.rsquared - m1.rsquared:+.4f}")

print(f"\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80 + "\n")

if m2.rsquared > m1.rsquared:
    print(f"✓ NORMALIZED EMOTIONS ARE BETTER (+{(m2.rsquared - m1.rsquared)*100:.2f}%)")
    print(f"\n  USE NORMALIZED: Emotional intensity matters more than volume")
    print(f"  INTERPRETATION: Videos with higher RATES of trust/joy get more likes")
    print(f"  ACTION: Regenerate dataset with normalized emotions")
else:
    print(f"✓ RAW EMOTIONS ARE BETTER (normalized is {(m1.rsquared - m2.rsquared)*100:.2f}% worse)")
    print(f"\n  KEEP RAW: Volume of emotional comments is the true signal")
    print(f"  INTERPRETATION: Videos with MORE trust-expressing comments get more likes")
    print(f"  ACTION: Keep current dataset (no changes needed)")
    
print(f"\n  Raw emotions work because:")
print(f"  1. Many comments = high engagement = popular video")
print(f"  2. Many trust/joy comments = strong community = deserves more likes")
print(f"  3. The volume IS the signal, not just the proportion")
print(f"\n  Normalized would only be better if we wanted to control for volume")
print(f"  and measure pure emotional 'quality' independent of popularity")

print("\n" + "=" * 80 + "\n")
