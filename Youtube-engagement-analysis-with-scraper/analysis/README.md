# Analysis Scripts

This directory contains scripts for analyzing the relationship between language features and video engagement metrics.

## Scripts

### language_engagement_study.py

**Corpus-wide bigram analysis**

Extracts the 100 most common bigrams across the entire dataset and uses them as features to predict engagement.

**Methodology:**

- Uses `CountVectorizer` with `ngram_range=(2,2)`, `max_features=100`, `min_df=5`
- Extracts bigrams that appear in at least 5 videos
- Creates binary features: does video contain each bigram?
- Compares models with/without channel fixed effects

**Models:**

1. Baseline: `engagement ~ channel_id` (channel fixed effects only)
2. Language: `engagement ~ bigrams` (language features only)
3. Full: `engagement ~ channel_id + bigrams` (both)

**Dependent variables:**

- Like rate: likes / views
- Comment rate: comments / views

**Results:**

- Channel identity dominates engagement patterns
- Language adds minimal predictive power after controlling for channel
- Like Rate: R² 0.46 → 0.47 (+0.5%)
- Comment Rate: R² 0.51 → 0.48 (-3%)

**Output:**

- `output/model_performance.txt` - R² scores for all models
- `output/bigram_coefficients_like_rate.csv` - Coefficients for like prediction
- `output/bigram_coefficients_comment_rate.csv` - Coefficients for comment prediction

**Usage:**

```bash
python analysis/language_engagement_study.py
```

### language_engagement_study_pervideo.py

**Per-video bigram statistics analysis**

Extracts the top 100 bigrams from EACH video individually and computes statistical features.

**Features created:**

- `total_unique_bigrams`: Count of unique bigrams in video
- `avg_bigram_freq`: Average frequency of top bigrams
- `max_bigram_freq`: Frequency of most repeated bigram
- `bigram_diversity`: Normalized entropy of bigram distribution

**Methodology:**

- Stratified sampling: 50% of videos from each channel (prevents large channel bias)
- Same model comparison as corpus-wide approach
- Channel stratification ensures balanced representation

**Results:**

- Language complexity adds significant predictive power
- Like Rate: R² 0.46 → 0.51 (+11.3%)
- Comment Rate: R² 0.51 → 0.64 (+24.9%)

**Key Finding:**
Videos with repetitive language (high `max_bigram_freq`) receive MORE comments. This counterintuitive result suggests that repetition may drive engagement through catchphrases or memorable hooks.

**Output:**

- `output/model_performance_pervideo.txt` - R² scores for all models
- `output/bigram_feature_coefficients_like_rate.csv` - Feature importance for likes
- `output/bigram_feature_coefficients_comment_rate.csv` - Feature importance for comments

**Usage:**

```bash
python analysis/language_engagement_study_pervideo.py
```

### analyze_metadata.py

**Exploratory data analysis**

Early exploratory script for understanding the dataset structure. Superseded by the language engagement studies.

## Comparison: Corpus-Wide vs Per-Video

| Approach        | Method                                               | Result                                | Interpretation                                   |
| --------------- | ---------------------------------------------------- | ------------------------------------- | ------------------------------------------------ |
| **Corpus-Wide** | Extract 100 most common bigrams across all videos    | Language adds <1% predictive power    | Channel-specific audience relationships dominate |
| **Per-Video**   | Extract statistics from each video's top 100 bigrams | Language adds 11-25% predictive power | Language complexity/repetition matters           |

**Why the difference?**

The corpus-wide approach captures which topics/words are universally popular, but this is dominated by channel identity. The per-video approach captures how language is USED within each video - complexity, repetition, and stylistic features that transcend specific words.

## Dataset Statistics

- **Videos:** 2,654 (with channel_id)
- **Transcripts:** 2,567 (combined segments)
- **Comments:** 320,166
- **Channels:** 7 fashion/beauty YouTubers
- **Time range:** Last 2 years

## Methodological Notes

**Channel Stratification:**
Both analyses use stratified sampling (50% of videos per channel) to prevent large channels from dominating the dataset. This ensures that results reflect language effects, not channel size.

**Channel Fixed Effects:**
Models include channel-specific indicator variables to control for channel identity. This isolates the effect of language features from audience loyalty and channel-specific engagement patterns.

**Dependent Variables:**
Engagement metrics are normalized by views (rates rather than counts) to account for video popularity. This focuses on engagement intensity rather than reach.
