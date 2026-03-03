"""
Extract ALL unique bigrams from the corpus to see the full scope.
"""

import json
import re
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def clean_text(text):
    """Clean transcript text."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load data
print("Loading data...")
videos = []
with open('output/videos.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            try:
                videos.append(json.loads(line))
            except:
                pass

transcripts = []
with open('output/transcripts.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            try:
                transcripts.append(json.loads(line))
            except:
                pass

# Merge
df_v = pd.DataFrame(videos)
df_t = pd.DataFrame(transcripts)
df = pd.merge(df_v, df_t, on='video_id', how='inner')
if 'view_count' in df.columns:
    df = df[df['view_count'] > 0]
elif 'viewCount' in df.columns:
    df = df[df['viewCount'] > 0]

# Sample 50% from each channel
df = df.groupby('channel_id', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=42))

print(f"Total videos: {len(df)}")

# Combine transcript segments
def combine_segments(row):
    if 'segments' in row and isinstance(row['segments'], list):
        return ' '.join([seg.get('text', '') for seg in row['segments']])
    elif 'transcript' in row:
        return row['transcript']
    return ''

df['full_text'] = df.apply(combine_segments, axis=1)
df['cleaned_text'] = df['full_text'].apply(clean_text)

# Extract ALL bigrams (no max_features limit)
print("\nExtracting ALL bigrams with min_df=5...")
vectorizer = CountVectorizer(
    ngram_range=(2, 2),
    min_df=5,
    stop_words='english',
    lowercase=True
)

bigram_matrix = vectorizer.fit_transform(df['cleaned_text'])
all_bigrams = vectorizer.get_feature_names_out()

print(f"\nTotal unique bigrams (appearing in 5+ videos): {len(all_bigrams)}")

# Get bigram frequencies
bigram_counts = bigram_matrix.sum(axis=0).A1
bigram_freq = list(zip(all_bigrams, bigram_counts))
bigram_freq.sort(key=lambda x: x[1], reverse=True)

print("\nTop 30 most frequent bigrams:")
for i, (bigram, count) in enumerate(bigram_freq[:30], 1):
    print(f"{i:3}. '{bigram}' - appears {int(count)} times")

print(f"\n\n{'='*60}")
print("EXPLANATION:")
print("="*60)
print(f"• Total videos analyzed: {len(df)}")
print(f"• Total unique bigrams (min 5 videos): {len(all_bigrams)}")
print(f"• For analysis, we used TOP 100 most frequent bigrams")
print(f"• These 100 bigrams represent the most common phrases")
print(f"• Rare bigrams were excluded to reduce noise")
print("="*60)
