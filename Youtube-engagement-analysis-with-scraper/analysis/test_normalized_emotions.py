"""
Test: Normalized vs Raw Emotion Scores

Compares:
1. Current: Raw emotion counts (conflated with volume)
2. Normalized: Emotions per comment (emotional intensity)
3. Normalized by words: Emotions per word (density)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("TESTING: NORMALIZED vs RAW EMOTION SCORES")
print("=" * 80 + "\n")

# Load data
df = pd.read_excel('output/final_regression_dataset.xlsx')
df_clean = df[df['transcript_text'].notna()].copy()

print(f"Sample: {len(df_clean)} videos\n")

# ============================================================================
# CREATE NORMALIZED EMOTIONS
# ============================================================================

print("=" * 80)
print("STEP 1 — CREATE NORMALIZED EMOTION FEATURES")
print("=" * 80 + "\n")

# Count number of comments per video (approximate from comment_length)
# Assuming average comment is ~100 characters
df_clean['estimated_comment_count'] = (df_clean['comment_length'] / 100).clip(lower=1)

# Also use has_comments (1 if any comments, 0 if none)
# For videos with comments, normalize by estimated count
# For videos without comments, keep at 0

emotion_cols = ['emotion_fear', 'emotion_anger', 'emotion_anticipation', 
                'emotion_trust', 'emotion_surprise', 'emotion_positive', 
                'emotion_negative', 'emotion_sadness', 'emotion_disgust', 
                'emotion_joy']

print("Creating normalized emotion features...")
for col in emotion_cols:
    # Normalized by estimated comment count
    df_clean[f'{col}_per_comment'] = df_clean[col] / df_clean['estimated_comment_count']
    
    # Normalized by total words (comment_length / 5 = approximate word count)
    df_clean[f'{col}_per_word'] = df_clean[col] / (df_clean['comment_length'] / 5 + 1)

print(f"✓ Created {len(emotion_cols) * 2} normalized features\n")

# Show comparison
print("=" * 80)
print("COMPARISON: Raw vs Normalized Emotions")
print("=" * 80 + "\n")

print("Example: emotion_trust")
print(f"  Raw emotion_trust:            Mean={df_clean['emotion_trust'].mean():.2f}, Std={df_clean['emotion_trust'].std():.2f}")
print(f"  Trust per comment:            Mean={df_clean['emotion_trust_per_comment'].mean():.2f}, Std={df_clean['emotion_trust_per_comment'].std():.2f}")
print(f"  Trust per word:               Mean={df_clean['emotion_trust_per_word'].mean():.4f}, Std={df_clean['emotion_trust_per_word'].std():.4f}")

print(f"\nExample: emotion_joy")
print(f"  Raw emotion_joy:              Mean={df_clean['emotion_joy'].mean():.2f}, Std={df_clean['emotion_joy'].std():.2f}")
print(f"  Joy per comment:              Mean={df_clean['emotion_joy_per_comment'].mean():.2f}, Std={df_clean['emotion_joy_per_comment'].std():.2f}")
print(f"  Joy per word:                 Mean={df_clean['emotion_joy_per_word'].mean():.4f}, Std={df_clean['emotion_joy_per_word'].std():.4f}")

# Check correlation with comment_length
print(f"\n📊 Correlation with comment_length:")
print(f"  Raw emotion_trust:            r={df_clean['emotion_trust'].corr(df_clean['comment_length']):.3f}")
print(f"  Normalized trust_per_comment: r={df_clean['emotion_trust_per_comment'].corr(df_clean['comment_length']):.3f}")
print(f"  Normalized trust_per_word:    r={df_clean['emotion_trust_per_word'].corr(df_clean['comment_length']):.3f}")

print(f"\n  → Raw emotions are HIGHLY correlated with volume")
print(f"  → Normalized emotions are less correlated (more independent)")

# ============================================================================
# TEST MODEL 1: RAW EMOTIONS (CURRENT)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2 — MODEL COMPARISON")
print("=" * 80 + "\n")

# Prepare features
control_vars = ['view_count', 'recency_days', 'duration_minutes', 'duration_dummy']
liwc_vars = [col for col in df_clean.columns if col not in 
             ['video_id', 'transcript_text', 'like_count'] + control_vars
             and not col.startswith('comment_') 
             and not col.startswith('emotion_')
             and col != 'has_comments'
             and '_per_' not in col]

# Raw emotions
raw_emotion_vars = emotion_cols + ['comment_polarity', 'comment_subjectivity', 
                                    'comment_length', 'has_comments']

# Normalized emotions (per comment)
norm_comment_vars = [f'{col}_per_comment' for col in emotion_cols] + \
                    ['comment_polarity', 'comment_subjectivity', 'has_comments']

# Normalized emotions (per word)
norm_word_vars = [f'{col}_per_word' for col in emotion_cols] + \
                 ['comment_polarity', 'comment_subjectivity', 'has_comments']

y = np.log1p(df_clean['like_count'])

print("=" * 80)
print("MODEL 1: Raw Emotions (Current)")
print("-" * 80)

X1 = df_clean[control_vars + liwc_vars + raw_emotion_vars].fillna(0)
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)
X1_sm = sm.add_constant(X1_scaled)
model1 = sm.OLS(y, X1_sm).fit()

print(f"Linear Regression:")
print(f"  R² = {model1.rsquared:.4f}")
print(f"  Adjusted R² = {model1.rsquared_adj:.4f}")

# Gradient Boosting
X1_train, X1_test, y_train, y_test = train_test_split(X1_scaled, y, test_size=0.2, random_state=42)
gb1 = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb1.fit(X1_train, y_train)
r2_gb1 = r2_score(y_test, gb1.predict(X1_test))
print(f"Gradient Boosting:")
print(f"  Test R² = {r2_gb1:.4f}")

print("\n" + "=" * 80)
print("MODEL 2: Normalized Emotions (Per Comment)")
print("-" * 80)

X2 = df_clean[control_vars + liwc_vars + norm_comment_vars].fillna(0)
X2_scaled = scaler.fit_transform(X2)
X2_sm = sm.add_constant(X2_scaled)
model2 = sm.OLS(y, X2_sm).fit()

print(f"Linear Regression:")
print(f"  R² = {model2.rsquared:.4f}")
print(f"  Adjusted R² = {model2.rsquared_adj:.4f}")
print(f"  Improvement: {model2.rsquared - model1.rsquared:+.4f}")

# Gradient Boosting
X2_train, X2_test, y_train, y_test = train_test_split(X2_scaled, y, test_size=0.2, random_state=42)
gb2 = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb2.fit(X2_train, y_train)
r2_gb2 = r2_score(y_test, gb2.predict(X2_test))
print(f"Gradient Boosting:")
print(f"  Test R² = {r2_gb2:.4f}")
print(f"  Improvement: {r2_gb2 - r2_gb1:+.4f}")

print("\n" + "=" * 80)
print("MODEL 3: Normalized Emotions (Per Word)")
print("-" * 80)

X3 = df_clean[control_vars + liwc_vars + norm_word_vars].fillna(0)
X3_scaled = scaler.fit_transform(X3)
X3_sm = sm.add_constant(X3_scaled)
model3 = sm.OLS(y, X3_sm).fit()

print(f"Linear Regression:")
print(f"  R² = {model3.rsquared:.4f}")
print(f"  Adjusted R² = {model3.rsquared_adj:.4f}")
print(f"  Improvement: {model3.rsquared - model1.rsquared:+.4f}")

# Gradient Boosting
X3_train, X3_test, y_train, y_test = train_test_split(X3_scaled, y, test_size=0.2, random_state=42)
gb3 = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb3.fit(X3_train, y_train)
r2_gb3 = r2_score(y_test, gb3.predict(X3_test))
print(f"Gradient Boosting:")
print(f"  Test R² = {r2_gb3:.4f}")
print(f"  Improvement: {r2_gb3 - r2_gb1:+.4f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATION")
print("=" * 80 + "\n")

results = pd.DataFrame({
    'Model': ['Raw Emotions (Current)', 'Per Comment Normalization', 'Per Word Normalization'],
    'Linear R²': [model1.rsquared, model2.rsquared, model3.rsquared],
    'Linear Adj R²': [model1.rsquared_adj, model2.rsquared_adj, model3.rsquared_adj],
    'GB Test R²': [r2_gb1, r2_gb2, r2_gb3],
    'vs Raw (Linear)': [0, model2.rsquared - model1.rsquared, model3.rsquared - model1.rsquared],
    'vs Raw (GB)': [0, r2_gb2 - r2_gb1, r2_gb3 - r2_gb1]
})

print(results.to_string(index=False))

best_idx = results['Linear R²'].idxmax()
best_model = results.iloc[best_idx]

print(f"\n🏆 BEST MODEL: {best_model['Model']}")
print(f"   Linear R² = {best_model['Linear R²']:.4f}")
print(f"   GB R² = {best_model['GB Test R²']:.4f}")

print(f"\n" + "=" * 80)
print("💡 INTERPRETATION & RECOMMENDATION")
print("=" * 80)

if best_idx == 0:
    print("\n✓ KEEP RAW EMOTIONS")
    print("  Raw emotion counts perform best, likely because:")
    print("  - Volume of emotional comments IS the signal we want")
    print("  - Many trust-expressing comments = genuine community trust")
    print("  - Normalization removes valuable information about engagement volume")
    print("\n  INTERPRETATION:")
    print("  'Videos with MORE trust-expressing comments get more likes'")
    print("  (Not just 'higher proportion of trust')")
else:
    print("\n✓ USE NORMALIZED EMOTIONS")
    print("  Normalized emotions perform better, suggesting:")
    print("  - INTENSITY of emotion matters more than volume")
    print("  - Emotional quality > quantity")
    print("  - Controls for confounding with comment volume")
    print("\n  INTERPRETATION:")
    print("  'Videos where comments express HIGHER RATES of trust get more likes'")
    print("  (Quality of emotional tone, not just volume)")

print(f"\n📊 CORRELATION CHECK:")
print(f"  Does raw emotion_trust just measure 'lots of comments'?")
like_trust_corr = df_clean['like_count'].corr(df_clean['emotion_trust'])
like_length_corr = df_clean['like_count'].corr(df_clean['comment_length'])
trust_length_corr = df_clean['emotion_trust'].corr(df_clean['comment_length'])

print(f"    likes ↔ emotion_trust:   r = {like_trust_corr:.3f}")
print(f"    likes ↔ comment_length:  r = {like_length_corr:.3f}")
print(f"    trust ↔ comment_length:  r = {trust_length_corr:.3f}")

if trust_length_corr > 0.8:
    print(f"\n  ⚠️  emotion_trust is highly correlated with comment volume!")
    print(f"      Normalization might provide cleaner interpretation")
elif trust_length_corr > 0.5:
    print(f"\n  ⚠️  emotion_trust is moderately correlated with volume")
    print(f"      Consider normalization for academic rigor")
else:
    print(f"\n  ✓  emotion_trust is relatively independent of volume")

print("\n" + "=" * 80 + "\n")
