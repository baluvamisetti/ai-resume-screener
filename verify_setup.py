# ============================================================
# verify_setup.py
# PURPOSE : Verify every library, model file, folder and
#           dataset is in place before running the app
# USAGE   : python verify_setup.py
# ============================================================

import os
import sys

print("=" * 55)
print("  AI Resume Screener — Environment Verification")
print("=" * 55)

errors = []

# ── Libraries ─────────────────────────────────────────────
LIBS = {
    "numpy"      : "numpy",
    "pandas"     : "pandas",
    "sklearn"    : "scikit-learn",
    "nltk"       : "nltk",
    "spacy"      : "spaCy",
    "pdfplumber" : "pdfplumber",
    "flask"      : "Flask",
    "matplotlib" : "matplotlib",
    "seaborn"    : "seaborn",
    "joblib"     : "joblib",
}

print("\n📦 Libraries:")
for lib, name in LIBS.items():
    try:
        __import__(lib)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name}  ← pip install {name.lower()}")
        errors.append(name)

# ── spaCy model ────────────────────────────────────────────
print("\n🧠 spaCy model:")
try:
    import spacy
    spacy.load("en_core_web_sm")
    print("   ✅ en_core_web_sm")
except Exception:
    print("   ❌ en_core_web_sm  ← python -m spacy download en_core_web_sm")
    errors.append("spaCy model")

# ── NLTK data ──────────────────────────────────────────────
print("\n📚 NLTK data:")
try:
    from nltk.corpus import stopwords
    stopwords.words("english")
    print("   ✅ stopwords")
except Exception:
    print("   ❌ stopwords  ← python -c \"import nltk; nltk.download('stopwords')\"")
    errors.append("NLTK stopwords")

# ── Folders ────────────────────────────────────────────────
print("\n📁 Folders:")
FOLDERS = [
    "data/raw", "data/processed", "notebooks",
    "src", "model", "templates", "static", "screenshots",
]
for folder in FOLDERS:
    status = "✅" if os.path.isdir(folder) else "❌"
    print(f"   {status} {folder}/")
    if status == "❌":
        errors.append(f"folder: {folder}")

# ── Source files ───────────────────────────────────────────
print("\n🐍 Source files:")
SRC_FILES = [
    "app.py",
    "src/text_cleaner.py",
    "src/skill_extractor.py",
    "src/resume_parser.py",
    "src/matcher.py",
    "src/model_training.py",
    "src/predictor.py",
    "templates/index.html",
    "templates/match.html",
    "templates/dashboard.html",
    "static/style.css",
    "static/dashboard.js",
]
for f in SRC_FILES:
    status = "✅" if os.path.isfile(f) else "❌"
    print(f"   {status} {f}")
    if status == "❌":
        errors.append(f)

# ── Datasets ───────────────────────────────────────────────
print("\n🗂️  Datasets:")
datasets = [
    ("data/raw/resumes.csv",          "Resume Dataset (from Kaggle)"),
    ("data/raw/job_descriptions.csv", "Job Descriptions (from Kaggle)"),
    ("data/processed/cleaned_resumes.csv", "Cleaned Resumes (run EDA notebook)"),
]
for path, label in datasets:
    if os.path.isfile(path):
        import pandas as pd
        df   = pd.read_csv(path)
        size = f"{df.shape[0]} rows"
        print(f"   ✅ {label} — {size}")
    else:
        print(f"   ⚠️  {label} — NOT FOUND")

# ── Model files ────────────────────────────────────────────
print("\n🤖 Saved model:")
MODEL_FILES = [
    "model/full_pipeline.pkl",
    "model/label_encoder.pkl",
    "model/model_metadata.pkl",
]
for mf in MODEL_FILES:
    if os.path.isfile(mf):
        kb = os.path.getsize(mf) / 1024
        print(f"   ✅ {mf} ({kb:.1f} KB)")
    else:
        print(f"   ⚠️  {mf} — run: python src/model_training.py")

# ── Summary ────────────────────────────────────────────────
print("\n" + "=" * 55)
if not errors:
    print("  🎉 All checks passed!")
    print("  Run: python app.py → http://127.0.0.1:5000")
else:
    print(f"  ⚠️  {len(errors)} issue(s) found:")
    for e in errors:
        print(f"     • {e}")
    print("  Fix the issues above and re-run verify_setup.py")
print("=" * 55)
