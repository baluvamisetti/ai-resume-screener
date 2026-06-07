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
print('NLTK downloads complete')
"

python -c "
import os, sys
sys.path.insert(0, '.')
model_path = 'model/full_pipeline.pkl'
if not os.path.exists(model_path):
    print('Model not found — training now...')
    from src.model_training import load_and_prepare_data, train_and_evaluate, save_model
    DATA_PATH = 'data/processed/cleaned_resumes.csv'
    if os.path.exists(DATA_PATH):
        X, y, le, df = load_and_prepare_data(DATA_PATH)
        result = train_and_evaluate(X, y, le, model_type='logistic')
        metadata = {
            'model_type'    : 'logistic',
            'test_accuracy' : str(round(result[\"test_accuracy\"]*100,2))+'%',
            'f1_weighted'   : str(round(result[\"f1_weighted\"]*100,2))+'%',
            'cv_mean'       : str(round(result[\"cv_mean\"]*100,2))+'%',
            'n_classes'     : len(le.classes_),
            'classes'       : list(le.classes_),
        }
        save_model(result['pipeline'], le, metadata)
        print('Model trained and saved!')
    else:
        print('WARNING: Training data not found.')
else:
    print('Model already exists — skipping training')
"

echo "========================================"
echo "  Build complete!"
echo "========================================"
