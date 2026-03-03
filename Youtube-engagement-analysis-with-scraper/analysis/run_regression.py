"""
Regression Analysis for YouTube Engagement

Analyzes which linguistic features (LIWC) and comment emotions predict video likes.

Models:
1. Baseline: Controls only
2. LIWC Model: Controls + LIWC features
3. Comment Model: Controls + Comment features
4. Full Model: All features
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("YOUTUBE ENGAGEMENT REGRESSION ANALYSIS")
print("=" * 80)
print("\nPredicting like_count using LIWC features and comment emotions\n")

# ============================================================================
# STEP 1 — LOAD AND PREPARE DATA
# ============================================================================

print("=" * 80)
print("STEP 1 — LOAD AND PREPARE DATA")
print("=" * 80 + "\n")

print("Loading dataset...")
df = pd.read_excel('output/final_regression_dataset.xlsx')
print(f"  ✓ Loaded {len(df)} videos with {len(df.columns)} variables")

# Remove rows with missing transcript_text (no LIWC features available)
print(f"\nRemoving videos without transcripts...")
df_clean = df[df['transcript_text'].notna()].copy()
print(f"  ✓ Removed {len(df) - len(df_clean)} videos")
print(f"  ✓ Final sample: {len(df_clean)} videos")

# Identify feature groups
control_vars = ['view_count', 'recency_days', 'duration_minutes', 'duration_dummy']

# LIWC features (excluding text column)
liwc_vars = [col for col in df_clean.columns if col not in 
             ['video_id', 'transcript_text', 'like_count'] + control_vars
             and not col.startswith('comment_') 
             and not col.startswith('emotion_')
             and col != 'has_comments']

# Comment features
comment_vars = [col for col in df_clean.columns if 
                col.startswith('comment_') or col.startswith('emotion_') or col == 'has_comments']

print(f"\n📊 Feature Groups:")
print(f"  Controls: {len(control_vars)} variables")
print(f"  LIWC Features: {len(liwc_vars)} variables")
print(f"  Comment Features: {len(comment_vars)} variables")

# Handle outliers in dependent variable (using IQR method)
Q1 = df_clean['like_count'].quantile(0.25)
Q3 = df_clean['like_count'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 3 * IQR  # More conservative (3x IQR instead of 1.5x)

print(f"\n📈 Dependent Variable (like_count):")
print(f"  Mean: {df_clean['like_count'].mean():.0f}")
print(f"  Median: {df_clean['like_count'].median():.0f}")
print(f"  Std Dev: {df_clean['like_count'].std():.0f}")
print(f"  Outlier threshold (Q3 + 3*IQR): {outlier_threshold:.0f}")

outliers = df_clean['like_count'] > outlier_threshold
print(f"  Videos above threshold: {outliers.sum()}")
print(f"  Keeping all observations for robustness")

# Log transform dependent variable (common for count data)
print(f"\n🔄 Transforming dependent variable...")
df_clean['log_likes'] = np.log1p(df_clean['like_count'])  # log(1 + x) to handle zeros
print(f"  ✓ Created log_likes (log-transformed like_count)")

# ============================================================================
# STEP 2 — PREPARE REGRESSION DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2 — PREPARE REGRESSION DATA")
print("=" * 80 + "\n")

# Standardize all predictors (mean=0, sd=1) for coefficient comparison
print("Standardizing independent variables...")

scaler = StandardScaler()

# Standardize controls
X_controls = scaler.fit_transform(df_clean[control_vars])
X_controls_df = pd.DataFrame(X_controls, columns=control_vars, index=df_clean.index)

# Standardize LIWC
X_liwc = scaler.fit_transform(df_clean[liwc_vars])
X_liwc_df = pd.DataFrame(X_liwc, columns=liwc_vars, index=df_clean.index)

# Standardize comments
X_comments = scaler.fit_transform(df_clean[comment_vars])
X_comments_df = pd.DataFrame(X_comments, columns=comment_vars, index=df_clean.index)

print(f"  ✓ Standardized {len(control_vars)} control variables")
print(f"  ✓ Standardized {len(liwc_vars)} LIWC features")
print(f"  ✓ Standardized {len(comment_vars)} comment features")

# Dependent variable
y = df_clean['log_likes']

print(f"\n✓ Data ready for regression")
print(f"  Sample size: {len(y)}")
print(f"  Dependent variable: log_likes (log-transformed)")

# ============================================================================
# STEP 3 — MODEL 1: BASELINE (CONTROLS ONLY)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3 — MODEL 1: BASELINE (CONTROLS ONLY)")
print("=" * 80 + "\n")

X1 = sm.add_constant(X_controls_df)
model1 = sm.OLS(y, X1).fit()

print("Model 1 Results:")
print(f"  R²: {model1.rsquared:.4f}")
print(f"  Adjusted R²: {model1.rsquared_adj:.4f}")
print(f"  F-statistic: {model1.fvalue:.2f} (p={model1.f_pvalue:.4f})")
print(f"  N: {int(model1.nobs)}")

print(f"\n  Significant predictors (p < 0.05):")
sig_vars1 = model1.pvalues[model1.pvalues < 0.05].index.tolist()
if 'const' in sig_vars1:
    sig_vars1.remove('const')
for var in sig_vars1:
    coef = model1.params[var]
    pval = model1.pvalues[var]
    print(f"    {var:30s} β={coef:7.4f} (p={pval:.4f})")

# ============================================================================
# STEP 4 — MODEL 2: LIWC MODEL
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4 — MODEL 2: LIWC MODEL (CONTROLS + LIWC)")
print("=" * 80 + "\n")

X2 = sm.add_constant(pd.concat([X_controls_df, X_liwc_df], axis=1))
print(f"Running regression with {X2.shape[1]-1} predictors...")
model2 = sm.OLS(y, X2).fit()

print(f"\nModel 2 Results:")
print(f"  R²: {model2.rsquared:.4f}")
print(f"  Adjusted R²: {model2.rsquared_adj:.4f}")
print(f"  F-statistic: {model2.fvalue:.2f} (p={model2.f_pvalue:.4f})")
print(f"  N: {int(model2.nobs)}")

print(f"\n  Improvement over baseline:")
print(f"    ΔR²: {model2.rsquared - model1.rsquared:.4f}")
print(f"    ΔAdjusted R²: {model2.rsquared_adj - model1.rsquared_adj:.4f}")

print(f"\n  Significant LIWC predictors (p < 0.05):")
sig_vars2 = model2.pvalues[model2.pvalues < 0.05].index.tolist()
if 'const' in sig_vars2:
    sig_vars2.remove('const')

liwc_sig = [v for v in sig_vars2 if v in liwc_vars]
print(f"  Found {len(liwc_sig)} significant LIWC features\n")

# Sort by absolute coefficient size
liwc_sig_sorted = sorted(liwc_sig, key=lambda x: abs(model2.params[x]), reverse=True)
for i, var in enumerate(liwc_sig_sorted[:20], 1):  # Top 20
    coef = model2.params[var]
    pval = model2.pvalues[var]
    print(f"    {i:2d}. {var:30s} β={coef:7.4f} (p={pval:.4f})")

if len(liwc_sig_sorted) > 20:
    print(f"    ... and {len(liwc_sig_sorted) - 20} more")

# ============================================================================
# STEP 5 — MODEL 3: COMMENT MODEL
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5 — MODEL 3: COMMENT MODEL (CONTROLS + COMMENTS)")
print("=" * 80 + "\n")

X3 = sm.add_constant(pd.concat([X_controls_df, X_comments_df], axis=1))
print(f"Running regression with {X3.shape[1]-1} predictors...")
model3 = sm.OLS(y, X3).fit()

print(f"\nModel 3 Results:")
print(f"  R²: {model3.rsquared:.4f}")
print(f"  Adjusted R²: {model3.rsquared_adj:.4f}")
print(f"  F-statistic: {model3.fvalue:.2f} (p={model3.f_pvalue:.4f})")
print(f"  N: {int(model3.nobs)}")

print(f"\n  Improvement over baseline:")
print(f"    ΔR²: {model3.rsquared - model1.rsquared:.4f}")
print(f"    ΔAdjusted R²: {model3.rsquared_adj - model1.rsquared_adj:.4f}")

print(f"\n  Significant comment predictors (p < 0.05):")
sig_vars3 = model3.pvalues[model3.pvalues < 0.05].index.tolist()
if 'const' in sig_vars3:
    sig_vars3.remove('const')

comment_sig = [v for v in sig_vars3 if v in comment_vars]
print(f"  Found {len(comment_sig)} significant comment features\n")

for var in sorted(comment_sig, key=lambda x: abs(model3.params[x]), reverse=True):
    coef = model3.params[var]
    pval = model3.pvalues[var]
    print(f"    {var:30s} β={coef:7.4f} (p={pval:.4f})")

# ============================================================================
# STEP 6 — MODEL 4: FULL MODEL
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6 — MODEL 4: FULL MODEL (ALL FEATURES)")
print("=" * 80 + "\n")

X4 = sm.add_constant(pd.concat([X_controls_df, X_liwc_df, X_comments_df], axis=1))
print(f"Running regression with {X4.shape[1]-1} predictors...")
model4 = sm.OLS(y, X4).fit()

print(f"\nModel 4 Results:")
print(f"  R²: {model4.rsquared:.4f}")
print(f"  Adjusted R²: {model4.rsquared_adj:.4f}")
print(f"  F-statistic: {model4.fvalue:.2f} (p={model4.f_pvalue:.4f})")
print(f"  N: {int(model4.nobs)}")

print(f"\n  Improvement over baseline:")
print(f"    ΔR²: {model4.rsquared - model1.rsquared:.4f}")
print(f"    ΔAdjusted R²: {model4.rsquared_adj - model1.rsquared_adj:.4f}")

print(f"\n  Top 30 significant predictors (p < 0.05):")
sig_vars4 = model4.pvalues[model4.pvalues < 0.05].index.tolist()
if 'const' in sig_vars4:
    sig_vars4.remove('const')

print(f"  Found {len(sig_vars4)} significant features total\n")

sig_vars4_sorted = sorted(sig_vars4, key=lambda x: abs(model4.params[x]), reverse=True)
for i, var in enumerate(sig_vars4_sorted[:30], 1):
    coef = model4.params[var]
    pval = model4.pvalues[var]
    var_type = "CONTROL" if var in control_vars else "LIWC" if var in liwc_vars else "COMMENT"
    print(f"    {i:2d}. [{var_type:7s}] {var:30s} β={coef:7.4f} (p={pval:.4f})")

if len(sig_vars4_sorted) > 30:
    print(f"    ... and {len(sig_vars4_sorted) - 30} more")

# ============================================================================
# STEP 7 — MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7 — MODEL COMPARISON")
print("=" * 80 + "\n")

comparison_data = {
    'Model': ['1. Baseline (Controls)', '2. LIWC Model', '3. Comment Model', '4. Full Model'],
    'Predictors': [len(control_vars), len(control_vars) + len(liwc_vars), 
                   len(control_vars) + len(comment_vars), 
                   len(control_vars) + len(liwc_vars) + len(comment_vars)],
    'R²': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared],
    'Adj R²': [model1.rsquared_adj, model2.rsquared_adj, model3.rsquared_adj, model4.rsquared_adj],
    'F-stat': [model1.fvalue, model2.fvalue, model3.fvalue, model4.fvalue],
    'Sig Features': [len([v for v in sig_vars1 if v != 'const']),
                     len([v for v in sig_vars2 if v != 'const']),
                     len([v for v in sig_vars3 if v != 'const']),
                     len([v for v in sig_vars4 if v != 'const'])]
}

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# ============================================================================
# STEP 8 — EXPORT RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8 — EXPORT RESULTS")
print("=" * 80 + "\n")

# Export all significant predictors from full model
print("Exporting detailed results...")

sig_results = []
for var in sig_vars4_sorted:
    var_type = "Control" if var in control_vars else "LIWC" if var in liwc_vars else "Comment"
    sig_results.append({
        'Variable': var,
        'Type': var_type,
        'Coefficient': model4.params[var],
        'Std_Error': model4.bse[var],
        'T_Statistic': model4.tvalues[var],
        'P_Value': model4.pvalues[var],
        'CI_Lower': model4.conf_int().loc[var, 0],
        'CI_Upper': model4.conf_int().loc[var, 1]
    })

df_significant = pd.DataFrame(sig_results)

# Create Excel writer
output_file = 'output/regression_results.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    
    # Sheet 1: Model Comparison
    df_comparison.to_excel(writer, sheet_name='Model Comparison', index=False)
    
    # Sheet 2: Significant Predictors (Full Model)
    df_significant.to_excel(writer, sheet_name='Significant Predictors', index=False)
    
    # Sheet 3: All Coefficients (Full Model)
    coef_data = {
        'Variable': model4.params.index.tolist(),
        'Coefficient': model4.params.values,
        'Std_Error': model4.bse.values,
        'P_Value': model4.pvalues.values
    }
    df_all_coef = pd.DataFrame(coef_data)
    df_all_coef.to_excel(writer, sheet_name='All Coefficients', index=False)
    
    # Sheet 4: Model Summary Stats
    summary_stats = {
        'Metric': ['R²', 'Adjusted R²', 'F-statistic', 'F p-value', 'N', 
                   'Significant Features', 'Total Features'],
        'Baseline': [model1.rsquared, model1.rsquared_adj, model1.fvalue, model1.f_pvalue,
                    int(model1.nobs), len([v for v in sig_vars1 if v != 'const']), len(control_vars)],
        'LIWC Model': [model2.rsquared, model2.rsquared_adj, model2.fvalue, model2.f_pvalue,
                      int(model2.nobs), len([v for v in sig_vars2 if v != 'const']), len(control_vars) + len(liwc_vars)],
        'Comment Model': [model3.rsquared, model3.rsquared_adj, model3.fvalue, model3.f_pvalue,
                         int(model3.nobs), len([v for v in sig_vars3 if v != 'const']), len(control_vars) + len(comment_vars)],
        'Full Model': [model4.rsquared, model4.rsquared_adj, model4.fvalue, model4.f_pvalue,
                      int(model4.nobs), len([v for v in sig_vars4 if v != 'const']), 
                      len(control_vars) + len(liwc_vars) + len(comment_vars)]
    }
    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_excel(writer, sheet_name='Summary Statistics', index=False)

print(f"  ✓ Exported to {output_file}")
print(f"    - Model Comparison")
print(f"    - Significant Predictors ({len(df_significant)} features)")
print(f"    - All Coefficients")
print(f"    - Summary Statistics")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ REGRESSION ANALYSIS COMPLETE")
print("=" * 80)

print(f"\n📊 KEY FINDINGS:")
print(f"\n  Best Model: Full Model (Controls + LIWC + Comments)")
print(f"  - R² = {model4.rsquared:.4f} ({model4.rsquared*100:.1f}% variance explained)")
print(f"  - Adjusted R² = {model4.rsquared_adj:.4f}")
print(f"  - {len(sig_vars4)} significant predictors out of {X4.shape[1]-1} total")

print(f"\n  Model Performance Comparison:")
print(f"  - Baseline R² = {model1.rsquared:.4f}")
print(f"  - Adding LIWC: ΔR² = +{model2.rsquared - model1.rsquared:.4f}")
print(f"  - Adding Comments: ΔR² = +{model3.rsquared - model1.rsquared:.4f}")
print(f"  - Full Model: ΔR² = +{model4.rsquared - model1.rsquared:.4f}")

print(f"\n  🎯 Top 5 Predictors (Full Model):")
for i, var in enumerate(sig_vars4_sorted[:5], 1):
    coef = model4.params[var]
    var_type = "CONTROL" if var in control_vars else "LIWC" if var in liwc_vars else "COMMENT"
    print(f"    {i}. [{var_type}] {var}: β={coef:.4f}")

print(f"\n📁 Results saved to: {output_file}")
print(f"   Open in Excel to explore all findings")

print("\n" + "=" * 80 + "\n")
