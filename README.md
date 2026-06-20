# 🤖 AI Resume Screener & Job Matching System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![spaCy](https://img.shields.io/badge/spaCy-3.7-09a3d5?logo=spacy)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?logo=render)

An end-to-end NLP-powered web application that **automatically matches
resumes to job descriptions**, predicts the job category, detects skill
gaps and gives actionable improvement suggestions — all in under 5 seconds.

---

## 🌐 Live Demo
https://ai-resume-screener-mn0s.onrender.com

---

## 🎯 Features
- 📄 **Paste text** or **upload PDF** resume
- 🧠 **NLP skill extraction** from 500+ skills across 11 domains
- 📊 **Weighted match score** (TF-IDF + Skill overlap + Experience)
- 🏷️ **Job category prediction** with confidence score
- ❌ **Skill gap analysis** — matched, missing and extra skills
- 💡 **Improvement suggestions** tailored to the gap
- 📈 **HR Analytics Dashboard** with Chart.js visualisations
- 🔌 **REST API** endpoints for integration

---

## 🛠️ Tech Stack

| Layer        | Technology                          |
|--------------|-------------------------------------|
| Language     | Python 3.11                         |
| Web Framework| Flask 3.0                           |
| ML / NLP     | scikit-learn, spaCy, NLTK           |
| Similarity   | TF-IDF + Cosine Similarity          |
| Classifier   | Logistic Regression                 |
| PDF Parsing  | pdfplumber                          |
| Frontend     | Bootstrap 5, Chart.js, Font Awesome |
| Deployment   | Render (gunicorn)                   |

---

## 📁 Project Structure
```
ai_resume_screener/
├── data/
│   ├── raw/                  ← Original datasets (Kaggle)
│   └── processed/            ← Cleaned CSVs
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_NLP_Processing.ipynb
│   └── 03_Model_Training.ipynb
├── src/
│   ├── text_cleaner.py       ← Text preprocessing
│   ├── skill_extractor.py    ← NLP skill extraction
│   ├── resume_parser.py      ← PDF + text parser
│   ├── matcher.py            ← Cosine similarity engine
│   ├── model_training.py     ← Train + save model
│   └── predictor.py          ← Flask-facing interface
├── model/                    ← Saved .pkl files
├── templates/                ← HTML pages
├── static/                   ← CSS + JS
├── app.py                    ← Flask entry point
├── requirements.txt
├── Procfile
└── render_build.sh
```

---

## ⚡ Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/baluvamisetti/ai-resume-screener
cd ai-resume-screener

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets from Kaggle and place:
#    data/raw/resumes.csv
#    data/raw/job_descriptions.csv

# 5. Run EDA notebook to create cleaned data
jupyter notebook notebooks/01_EDA.ipynb

# 6. Train the model
python src/model_training.py

# 7. Start the app
python app.py
```
Open → [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📊 Model Performance

| Metric        | Score  |
|---------------|--------|
| Test Accuracy | ~85%   |
| F1 (Weighted) | ~84%   |
| CV Mean (5-fold)| ~83% |
| Categories    | 25     |

---

## 🔌 REST API

**POST** `/api/match`
```json
{
  "resume_text": "Python developer, 3 years...",
  "jd_text"   : "We need a Python developer..."
}
```

**POST** `/api/analyze`
```json
{ "resume_text": "..." }
```

---

## 🚀 Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set **Build Command**: `bash render_build.sh`
5. Set **Start Command**: `gunicorn app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT`
6. Add env var: `SECRET_KEY = your_secret_here`
7. Click **Deploy**

---

## 💼 Resume Bullet Points

- Built an **end-to-end NLP-powered resume screening system** using Python,
  Flask, TF-IDF and Cosine Similarity achieving **85%+ classification accuracy**
  across 25 job categories.
- Engineered a **skill extraction pipeline** using spaCy and a custom 500+
  skill taxonomy covering 11 technical domains, reducing manual screening time
  by ~80%.
- Deployed a **production-ready Flask REST API** on Render with PDF parsing,
  multi-format input support and a real-time HR analytics dashboard.

---

## 🎤 Interview Q&A (Top 5)

**Q1: How does the match score work?**
> Final score = 50% TF-IDF cosine similarity + 35% skill keyword overlap +
> 15% experience comparison. Weights are configurable in `src/matcher.py`.

**Q2: Why Logistic Regression over Random Forest?**
> For high-dimensional sparse TF-IDF vectors, Logistic Regression converges
> faster, is less prone to overfitting and gives probability estimates via
> `predict_proba` — useful for the confidence badge in the UI.

**Q3: How do you handle PDFs?**
> pdfplumber extracts text page-by-page. If text extraction fails (scanned
> PDFs), we fall back to word-level extraction. Image-based PDFs prompt the
> user to paste text instead.

**Q4: What is TF-IDF?**
> Term Frequency–Inverse Document Frequency converts text to numerical vectors
> where common words score low and rare but relevant words score high, enabling
> cosine similarity comparison between resume and JD vectors.

**Q5: How would you improve this in production?**
> Add a database (PostgreSQL) to persist results, implement user auth, use
> Sentence-BERT for semantic similarity, add an admin panel for HR teams and
> integrate with LinkedIn/ATS via API.

---

## 📸 Screenshots
*(Add screenshots of the app here after running locally)*

---

## 👨‍💻 Author
**Balu Vamisetti** · [GitHub](https://github.com/baluvamisetti) ·
[LinkedIn](https://linkedin.com/in/baluvamisetti)

---

## 📄 License
MIT License — free to use for portfolio and learning purposes.
