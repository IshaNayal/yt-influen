# Project Organization Summary

## ✅ Organized Folder Structure

### 📁 Root Level
```
youtube transcripts/
├── reports/              ✓ Academic reports and summaries
├── scripts/              ✓ Utility scripts
├── analysis/             ✓ Main analysis scripts
├── data_collection/      ✓ YouTube data scraping
├── utils/                ✓ Helper functions
├── output/               ✓ All results (organized below)
├── .venv/                Virtual environment
└── README.md             Project documentation
```

---

## 📊 Output Folder Organization

### output/raw_data/
**Original collected data from YouTube**
- `videos.jsonl` - 2,654 videos metadata
- `transcripts.jsonl` - 2,481 transcripts
- `comments.jsonl` - 320,166 comments
- `videos_backup.jsonl` - Backup

### output/liwc_prep/
**Data prepared for LIWC-22 analysis**
- `youtube_transcripts_for_liwc.csv` - Cleaned transcripts
- `youtube_comments_for_liwc.csv` - Aggregated comments per video
- `youtube_metadata.xlsx` - Video metrics

### output/liwc_results/
**Professional LIWC-22 linguistic analysis**
- `LIWC-22 Results - youtube_transcripts_for_liwc - LIWC Analysis.xlsx`
  - 114 linguistic features per video

### output/bigram_analysis/
**Bigram-based engagement prediction**
- `pervideo_bigrams_combined.csv` - All bigrams
- `pervideo_bigrams_like_coefficients.csv` - 232 significant predictors
- `pervideo_bigrams_comment_coefficients.csv` - 75 significant predictors
- `model_performance.txt` - Corpus-wide results (R² = 0.51)
- `model_performance_pervideo.txt` - Per-video results (R² = 0.53)

### output/regression_results/
**Final regression models and datasets**
- `final_regression_dataset.xlsx` - Complete dataset (2,654 × 134)
- `regression_results.xlsx` - Linear model results (R² = 0.635)
- `improvement_analysis.xlsx` - ML comparison (R² = 0.881)

---

## 📄 Reports Folder

### reports/PROFESSOR_REPORT.md
**Complete academic report including:**
- Research question and methodology
- Data collection details
- Statistical analysis (4 models tested)
- Key findings (42 significant predictors)
- Limitations and future work
- Presentation talking points

### reports/ONE_PAGE_SUMMARY.md
**Quick reference with:**
- Model performance comparison
- Top 10 significant predictors
- Key insights and recommendations
- Elevator pitch (30 seconds)

---

## 🔧 Scripts Folder

### scripts/
**Utility scripts for verification**
- `check_all_bigrams.py` - Verify bigram methodology
- `create_bigram_summary.py` - Generate bigram summaries

---

## 📈 Analysis Folder

### Key Analysis Scripts
- `merge_liwc_results.py` - Integrate LIWC with metadata and emotions
- `run_regression.py` - Linear regression (R² = 0.635)
- `improve_regression.py` - Test ML models (R² = 0.881)
- `prepare_liwc_data_csv.py` - Prepare data for LIWC-22
- `language_engagement_study.py` - Corpus-wide bigram analysis
- `bigram_feature_regression.py` - Per-video bigram analysis

---

## 🎯 Quick Access Guide

### For Your Professor
→ `reports/PROFESSOR_REPORT.md` - Full academic report
→ `reports/ONE_PAGE_SUMMARY.md` - Quick summary

### For Results
→ `output/regression_results/regression_results.xlsx` - All significant predictors
→ `output/regression_results/final_regression_dataset.xlsx` - Complete dataset

### For Data
→ `output/raw_data/` - Original YouTube data
→ `output/liwc_results/` - LIWC-22 linguistic features

### For Methodology
→ `analysis/` - All analysis scripts with documentation
→ `README.md` - Project overview and usage

---

## 📊 Key Results Summary

### Model Performance
| Model | R² | Significant Predictors |
|-------|-----|----------------------|
| Baseline | 0.141 | 4 |
| + LIWC | 0.503 | 32 |
| + Comments | 0.499 | 9 |
| **Full Linear** | **0.635** | **42** |
| **Gradient Boosting** | **0.881** | — |

### Top 5 Predictors
1. **emotion_trust** (β=1.173, p<0.001) - Trust drives engagement
2. **emotion_joy** (β=0.857, p<0.001) - Happiness increases likes
3. **has_comments** (β=0.736, p<0.001) - Comments signal engagement
4. **function words** (β=0.620, p=0.003) - Conversational style works
5. **recency_days** (β=-0.644, p<0.001) - New content performs better

---

## ✅ Benefits of This Organization

1. **Clear Separation**: Raw data → Prep → Results → Reports
2. **Easy Navigation**: Each folder has its own README
3. **Academic Ready**: Reports folder contains presentation materials
4. **Reproducible**: Clear pipeline from data collection to final results
5. **Professional**: Organized structure suitable for sharing or archiving

---

**Everything is now organized and documented!** 🎉
