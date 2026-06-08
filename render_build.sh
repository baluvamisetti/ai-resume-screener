#!/usr/bin/env bash
set -o errexit

echo "========================================"
echo "  AI Resume Screener — Build Script"
echo "========================================"

pip install --upgrade pip
pip install -r requirements.txt

python -c "
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
print('NLTK OK')
"

python auto_setup.py

echo "========================================"
echo "  Build complete!"
echo "========================================"