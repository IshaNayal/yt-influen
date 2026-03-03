import pandas as pd

# Load coefficient files
like_df = pd.read_csv('output/bigram_coefficients_like_rate.csv')
comment_df = pd.read_csv('output/bigram_coefficients_comment_rate.csv')

# Combine data
result = pd.DataFrame({
    'bigram': like_df['bigram'],
    'like_coefficient': like_df['coefficient'],
    'comment_coefficient': comment_df['coefficient']
})

# Calculate total impact
result['total_abs_impact'] = result['like_coefficient'].abs() + result['comment_coefficient'].abs()

# Sort by impact
result = result.sort_values('total_abs_impact', ascending=False)

# Save
result.to_csv('output/significant_bigrams_data.csv', index=False)

print("✓ Saved to: output/significant_bigrams_data.csv")
print("\nTop 20 bigrams by total impact:\n")
print(result.head(20).to_string(index=False))
