# YouTube Engagement Study - One-Page Summary

---

## 📊 RESEARCH OVERVIEW

**Question:** What linguistic patterns in transcripts and emotions in comments predict YouTube video engagement?

**Sample:** 2,431 beauty videos | 320,166 comments | 130 predictive features

---

## 🔬 METHODOLOGY

| Component | Details |
|-----------|---------|
| **Linguistic Analysis** | LIWC-22 Professional Tool (114 features) |
| **Emotion Analysis** | NRC Lexicon + TextBlob (12 features) |
| **Controls** | Views, recency, duration (4 features) |
| **Statistical Method** | OLS Regression + Gradient Boosting ML |
| **Dependent Variable** | log(like_count) |

---

## 🎯 RESULTS

### Model Performance
```
Baseline (controls only):        R² = 0.141 (14%)
+ LIWC linguistic features:      R² = 0.503 (50%) → +36% improvement
+ Comment emotions:              R² = 0.499 (50%) → +36% improvement
FULL LINEAR MODEL:               R² = 0.635 (64%) ✓ REPORT THIS
Gradient Boosting (ML):          R² = 0.881 (88%) → Shows non-linearity
```

### Top 10 Significant Predictors

| Rank | Feature | Type | β | p-value | Impact |
|------|---------|------|---|---------|--------|
| 1 | **emotion_trust** | Comment | +1.17 | <0.001 | Trust drives engagement ✅ |
| 2 | **emotion_joy** | Comment | +0.86 | <0.001 | Happiness increases likes ✅ |
| 3 | **has_comments** | Meta | +0.74 | <0.001 | Comments signal engagement ✅ |
| 4 | **function words** | LIWC | +0.62 | 0.003 | Conversational style works ✅ |
| 5 | **recency_days** | Control | -0.64 | <0.001 | New content performs better ✅ |
| 6 | **emotion_disgust** | Comment | +0.60 | <0.001 | Controversial = engagement ✅ |
| 7 | **Social words** | LIWC | +0.67 | 0.047 | Community language helps ✅ |
| 8 | **verb usage** | LIWC | +0.42 | 0.003 | Action words engage ✅ |
| 9 | **emotion_anger** | Comment | -1.07 | <0.001 | Anger hurts engagement ❌ |
| 10 | **comment_polarity** | Comment | -0.36 | <0.001 | Extreme sentiment backfires ❌ |

---

## 💡 KEY INSIGHTS

### 1. Emotions Trump Language
- Comment emotions (trust, joy) are **strongest predictors**
- Community sentiment > what creator says in video
- Relationship quality drives engagement

### 2. Language Style Matters
- ✅ Simple, conversational > complex, formal
- ✅ Social, action-oriented language
- ✅ Short sentences (low WPS)
- ❌ Conjunctions and long sentences reduce engagement

### 3. Non-Linear Relationships
- Linear model: 63.5%
- ML model: 88.1%
- **+24.6% improvement** shows complex patterns

---

## 🎬 ACTIONABLE RECOMMENDATIONS

**For Content Creators:**
1. **Build trust** in your community (most important factor)
2. Create **joyful experiences** (2nd most important)
3. Use **conversational, simple language**
4. Include **social and physical descriptions**
5. Post **regularly** (recency matters)
6. Keep videos **concise** (duration_dummy effect)
7. Moderate comments to minimize anger/negativity

**Formula:** Trust + Joy + Conversational Style = High Engagement

---

## 📈 STATISTICAL RIGOR

- ✅ Log-transformed dependent variable (appropriate for count data)
- ✅ Standardized predictors (β coefficients comparable)
- ✅ Multiple model specifications tested
- ✅ Multicollinearity diagnostics (VIF analysis)
- ✅ Non-linear model comparison
- ✅ Professional tools (LIWC-22, NRC Lexicon)
- ✅ Large sample (2,431 videos, 320K comments)

---

## ⚠️ LIMITATIONS

1. **Missing YouTube Analytics**: No watch time, CTR, retention data
2. **Niche-specific**: Beauty content only (generalizability?)
3. **Temporal**: No time-of-day or seasonal analysis
4. **Causality**: Correlation ≠ causation
5. **Sample**: 223 videos without transcripts excluded

---

## 📊 COMPARISON TO LITERATURE

**Typical Social Media R²:** 0.30-0.50 (30-50%)

**This Study:**
- Linear: **0.635** (above average ✓)
- ML: **0.881** (excellent ✓✓)

**Achievement:** Successfully replicated academic methodology and extended with emotion analysis

---

## 🎓 ACADEMIC CONTRIBUTION

1. **Replicated** paper methodology with beauty content niche
2. **Extended** analysis with 12 emotion dimensions (NRC)
3. **Discovered** comment emotions > transcript language
4. **Demonstrated** non-linear patterns in engagement
5. **Provided** actionable creator insights

---

## 📁 DELIVERABLES

### Analysis Files:
- `output/regression_results.xlsx` - Full linear model (42 significant predictors)
- `output/final_regression_dataset.xlsx` - Complete data (2,654 × 134)
- `output/improvement_analysis.xlsx` - ML comparison

### Code:
- `analysis/merge_liwc_results.py` - Data integration
- `analysis/run_regression.py` - Linear models
- `analysis/improve_regression.py` - ML models

---

## 🗣️ ELEVATOR PITCH (30 seconds)

*"I analyzed 2,431 YouTube videos to predict engagement using linguistic analysis of transcripts and emotion detection in 320,000 comments. The model achieved 63.5% R² with traditional regression and 88.1% with machine learning. Key finding: **trust and joy in comments predict engagement better than what's said in the video itself**, suggesting community relationship quality is the most important factor for YouTube success."*

---

## ✅ BOTTOM LINE

**What to Report:**
- **Primary Model:** Full Linear Model (R² = 0.635)
  - Interpretable, statistically rigorous, academically appropriate
  
- **Supplementary Finding:** ML model (R² = 0.881)
  - Shows non-linear potential, practical applications

**Key Takeaway:** 
Language and emotions significantly predict YouTube engagement. **Trust + Joy = Likes.**

---

**RECOMMENDATION: Lead with R² = 0.635 for academic credibility, mention 0.881 as "future work" potential**
