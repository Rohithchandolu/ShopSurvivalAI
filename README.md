# 🏪 ShopSurvival — AI Business Survival Predictor
### Data Science Mini Project | 45 Days Syllabus Coverage

---

## 📁 Project Structure

```
shopsurvival/
├── phase1_data_engine.py     # Python + NumPy + Pandas
├── phase2_eda.py             # EDA + Matplotlib + Seaborn + Statistics
├── phase3_ml_models.py       # Hypothesis testing + All ML models
├── phase4_clusters.py        # KMeans + PCA + Hierarchical + ANN
├── phase5_nlp_cv.py          # NLP + Sentiment + CNN
├── phase6_app.py             # Streamlit web app (final deliverable)
├── data/                     # Auto-created: CSVs + plots
└── models/                   # Auto-created: saved model files
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib streamlit
pip install tensorflow          # for CNN in Phase 5 (optional)
```

### Step 2 — Get real data (free)
1. Go to https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset
2. Download `yelp_academic_dataset_business.json`
3. Place it in the `shopsurvival/` folder

### Step 3 — Run phases in order
```bash
python phase1_data_engine.py   # Creates data/cleaned_businesses.csv
python phase2_eda.py           # Creates 5 plots in data/
python phase3_ml_models.py     # Trains + compares all ML models
python phase4_clusters.py      # KMeans, PCA, ANN
python phase5_nlp_cv.py        # NLP + CNN
streamlit run phase6_app.py    # Launch the final web app!
```

> ✅ **No Yelp data?** All phases work with auto-generated mock data. Just run them directly.

---

## 🗺 Syllabus Coverage Map

| Phase | Topics Covered |
|-------|---------------|
| Phase 1 | Syntax, Variables, Data Types, Functions, *args/**kwargs, NumPy arrays, Pandas GroupBy, Merge, Null handling |
| Phase 2 | Matplotlib (line, bar, histogram), Seaborn (heatmap, pairplot), EDA (describe, value_counts), Mean/Std/Kurtosis/Covariance, Probability |
| Phase 3 | Hypothesis testing (p-value, t-test), Linear regression (MSE/RMSE/R²), Logistic regression (sigmoid), Decision tree (entropy), Random forest (ensemble), SVM, Naive Bayes, KNN, Classification metrics |
| Phase 4 | KMeans (elbow method), PCA (eigenvalues/eigenvectors), Hierarchical clustering (dendrogram), ANN (forward + backpropagation) |
| Phase 5 | NLP basics, Text processing (TF-IDF, tokenization), Sentiment analysis, CNN (Conv2D, MaxPooling, Flatten), Computer vision basics |
| Phase 6 | End-to-end Streamlit app, deployment |

---

## 🎯 Unique Features That Impress Interviewers

1. **Hypothesis test on real data** — "Does star rating significantly predict closure?" (p-value answer)
2. **Business Archetype Clustering** — KMeans assigns every business to: Struggling / Steady / Popular / Competitive
3. **PCA scatter** — visualises 7-dimensional data in 2D — always gets "wow" reactions
4. **Review Brain** — paste any review → instant sentiment → survival impact score
5. **Live prediction tab** — enter your business idea → get survival % in real time
6. **CNN storefront classifier** — upload a photo → model predicts if it looks like a survivor

---

## 📦 Data Sources (all free)
- **Yelp Business Dataset** → kaggle.com/datasets/yelp-dataset/yelp-dataset
- **Yelp Reviews Dataset** → same Kaggle page (for Phase 5 NLP)

---

## 🌐 Deploy for Free
```bash
# 1. Push to GitHub
# 2. Go to share.streamlit.io
# 3. Connect your repo → select phase6_app.py → Deploy!
# → Get a public link to share on your resume
```
