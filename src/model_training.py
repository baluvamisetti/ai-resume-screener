# ============================================================
# src/model_training.py
# PURPOSE : Train a resume category classifier
#           TF-IDF vectorizer + Logistic Regression pipeline
#           Saves artifacts to model/ for use in Flask app
# ============================================================

import os
import sys
import time
import warnings
import numpy  as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.pipeline          import Pipeline
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.svm               import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection   import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.metrics           import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.preprocessing     import LabelEncoder

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from src.text_cleaner import advanced_clean

# ── Paths ──────────────────────────────────────────────────
MODEL_DIR       = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model"
)
os.makedirs(MODEL_DIR, exist_ok=True)

PIPELINE_PATH   = os.path.join(MODEL_DIR, "full_pipeline.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "resume_classifier.pkl")
LABEL_ENC_PATH  = os.path.join(MODEL_DIR, "label_encoder.pkl")
METADATA_PATH   = os.path.join(MODEL_DIR, "model_metadata.pkl")


# ============================================================
# FUNCTION 1 — Load & Prepare Data
# ============================================================
def load_and_prepare_data(csv_path: str) -> tuple:
    """
    Loads cleaned_resumes.csv, standardises column names,
    applies text cleaning and label-encodes categories.

    Returns: (X_text, y, label_encoder, dataframe)
    """
    print("📥 Loading dataset...")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Accept common column name variants
    for col in ["category", "resume"]:
        if col not in df.columns:
            alts = {
                "category": ["Category", "label", "Label"],
                "resume"  : ["Resume", "text", "Text", "content"],
            }
            for alt in alts.get(col, []):
                if alt.lower() in df.columns:
                    df = df.rename(columns={alt.lower(): col})
                    break

    df = df[["category", "resume"]].dropna()
    df = df[df["resume"].str.strip().str.len() > 50]
    df = df.reset_index(drop=True)

    print(f"   ✅ {len(df)} samples | {df['category'].nunique()} categories")

    # Use pre-cleaned column if available (faster)
    if "resume_clean_advanced" in df.columns:
        X_text = df["resume_clean_advanced"].fillna("").tolist()
        print("   ✅ Using pre-cleaned text")
    else:
        print("   ⏳ Cleaning text (may take ~60 s)...")
        X_text = df["resume"].apply(advanced_clean).tolist()

    le = LabelEncoder()
    y  = le.fit_transform(df["category"])

    print("\n📊 Class distribution (top 10):")
    counts = pd.Series(df["category"]).value_counts()
    for cat, cnt in counts.head(10).items():
        bar = "█" * (cnt // 4)
        print(f"   {cat:<35} {bar} ({cnt})")

    return X_text, y, le, df


# ============================================================
# FUNCTION 2 — Build Pipeline
# ============================================================
def build_pipeline(model_type: str = "logistic") -> Pipeline:
    """
    Returns a sklearn Pipeline: TfidfVectorizer + Classifier.

    model_type options: "logistic" | "random_forest" | "svm"
    """
    vectorizer = TfidfVectorizer(
        max_features  = 10000,
        ngram_range   = (1, 2),
        min_df        = 2,
        max_df        = 0.95,
        sublinear_tf  = True,
        strip_accents = "unicode",
        analyzer      = "word",
    )
    classifiers = {
        "logistic": LogisticRegression(
            C            = 1.0,
            max_iter     = 1000,
            class_weight = "balanced",
            random_state = 42,
            n_jobs       = -1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators = 200,
            class_weight = "balanced",
            random_state = 42,
            n_jobs       = -1,
        ),
        "svm": LinearSVC(
            C            = 1.0,
            max_iter     = 2000,
            class_weight = "balanced",
            random_state = 42,
        ),
    }
    if model_type not in classifiers:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose: {list(classifiers.keys())}"
        )
    return Pipeline([
        ("tfidf",      vectorizer),
        ("classifier", classifiers[model_type]),
    ])


# ============================================================
# FUNCTION 3 — Train & Evaluate
# ============================================================
def train_and_evaluate(
    X_text     : list,
    y          : np.ndarray,
    le         : LabelEncoder,
    model_type : str   = "logistic",
    test_size  : float = 0.20,
) -> dict:
    """
    Runs full training pipeline with evaluation metrics
    and 5-fold cross-validation.
    """
    print(f"\n{'='*50}")
    print(f"  Training: {model_type.upper()}")
    print(f"{'='*50}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y,
        test_size    = test_size,
        random_state = 42,
        stratify     = y,
    )
    print(f"\n   Train: {len(X_train)} | Test: {len(X_test)}")

    pipeline = build_pipeline(model_type)

    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred         = pipeline.predict(X_test)
    train_accuracy = accuracy_score(y_train, pipeline.predict(X_train))
    test_accuracy  = accuracy_score(y_test, y_pred)
    f1_weighted    = f1_score(y_test, y_pred, average="weighted")
    f1_macro       = f1_score(y_test, y_pred, average="macro")

    print(f"\n📈 Results:")
    print(f"   Train Accuracy : {train_accuracy*100:.2f}%")
    print(f"   Test  Accuracy : {test_accuracy*100:.2f}%")
    print(f"   F1 (Weighted)  : {f1_weighted*100:.2f}%")
    print(f"   F1 (Macro)     : {f1_macro*100:.2f}%")
    print(f"   Train Time     : {train_time:.2f}s")

    gap = train_accuracy - test_accuracy
    flag = "⚠️  possible overfit" if gap > 0.15 else "✅ good generalisation"
    print(f"   Train-Test Gap : {gap*100:.1f}%  {flag}")

    # Cross-validation
    print("\n⏳ 5-fold cross-validation...")
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipeline, X_text, y, cv=cv, scoring="accuracy", n_jobs=-1
    )
    print(f"   CV Scores : {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"   CV Mean   : {cv_scores.mean()*100:.2f}%")
    print(f"   CV Std    : {cv_scores.std()*100:.2f}%")

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return {
        "pipeline"         : pipeline,
        "X_train"          : X_train,
        "X_test"           : X_test,
        "y_train"          : y_train,
        "y_test"           : y_test,
        "y_pred"           : y_pred,
        "train_accuracy"   : train_accuracy,
        "test_accuracy"    : test_accuracy,
        "f1_weighted"      : f1_weighted,
        "f1_macro"         : f1_macro,
        "confusion_matrix" : confusion_matrix(y_test, y_pred),
        "cv_scores"        : cv_scores,
        "cv_mean"          : cv_scores.mean(),
        "cv_std"           : cv_scores.std(),
        "report_dict"      : classification_report(
                                 y_test, y_pred,
                                 target_names=le.classes_,
                                 output_dict=True
                             ),
        "model_type"       : model_type,
        "train_time"       : train_time,
    }


# ============================================================
# FUNCTION 4 — Save Model
# ============================================================
def save_model(pipeline: Pipeline,
               le: LabelEncoder,
               metadata: dict) -> None:
    """Saves pipeline, individual components, label encoder and metadata."""
    print("\n💾 Saving model artifacts...")

    joblib.dump(pipeline,                         PIPELINE_PATH)
    joblib.dump(pipeline.named_steps["tfidf"],    VECTORIZER_PATH)
    joblib.dump(pipeline.named_steps["classifier"],CLASSIFIER_PATH)
    joblib.dump(le,                               LABEL_ENC_PATH)
    joblib.dump(metadata,                         METADATA_PATH)

    for path in [PIPELINE_PATH, VECTORIZER_PATH,
                 CLASSIFIER_PATH, LABEL_ENC_PATH]:
        kb = os.path.getsize(path) / 1024
        print(f"   ✅ {os.path.basename(path)} ({kb:.1f} KB)")


# ============================================================
# FUNCTION 5 — Load Model
# ============================================================
def load_model() -> tuple:
    """
    Loads saved pipeline and label encoder from disk.
    Returns: (pipeline, label_encoder, metadata)
    """
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            f"Model not found: {PIPELINE_PATH}\n"
            "Run model_training.py first."
        )
    pipeline = joblib.load(PIPELINE_PATH)
    le       = joblib.load(LABEL_ENC_PATH)
    metadata = (
        joblib.load(METADATA_PATH)
        if os.path.exists(METADATA_PATH) else {}
    )
    print(
        f"✅ Model loaded | "
        f"Accuracy: {metadata.get('test_accuracy','N/A')} | "
        f"Classes: {len(le.classes_)}"
    )
    return pipeline, le, metadata


# ── Entry Point ────────────────────────────────────────────
if __name__ == "__main__":
    DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "processed", "cleaned_resumes.csv"
    )
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found: {DATA_PATH}")
        print("   Run the EDA notebook (Stage 2) first.")
        sys.exit(1)

    X_text, y, le, df = load_and_prepare_data(DATA_PATH)

    result = train_and_evaluate(
        X_text, y, le,
        model_type = "logistic",
        test_size  = 0.20,
    )

    metadata = {
        "model_type"     : result["model_type"],
        "test_accuracy"  : f"{result['test_accuracy']*100:.2f}%",
        "f1_weighted"    : f"{result['f1_weighted']*100:.2f}%",
        "cv_mean"        : f"{result['cv_mean']*100:.2f}%",
        "n_classes"      : len(le.classes_),
        "classes"        : list(le.classes_),
        "n_train_samples": len(result["X_train"]),
        "n_test_samples" : len(result["X_test"]),
        "train_time"     : result["train_time"],
    }
    save_model(result["pipeline"], le, metadata)

    print("\n" + "="*50)
    print("  ✅ Training complete! Ready for Flask app.")
    print("="*50)
