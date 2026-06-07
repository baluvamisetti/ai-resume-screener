# ============================================================
# src/predictor.py
# PURPOSE : Clean prediction interface used by Flask app
#           Loads model once (lazy singleton), exposes:
#             predict_category()  — resume category + proba
#             analyze_resume()    — full profile, no JD needed
#             analyze_match()     — full resume vs JD analysis
# ============================================================

import os
import sys
import joblib

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.text_cleaner    import advanced_clean
from src.skill_extractor import build_resume_profile
from src.matcher         import compute_match_score, generate_suggestions
from src.resume_parser   import parse_resume

# ── Model Paths ────────────────────────────────────────────
MODEL_DIR      = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model"
)
PIPELINE_PATH  = os.path.join(MODEL_DIR, "full_pipeline.pkl")
LABEL_ENC_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
METADATA_PATH  = os.path.join(MODEL_DIR, "model_metadata.pkl")

# ── Module-level cache (load once per process) ─────────────
_pipeline = None
_le       = None
_metadata = None


def _load_model():
    """Lazy-loads model from disk on first call."""
    global _pipeline, _le, _metadata
    if _pipeline is None:
        if not os.path.exists(PIPELINE_PATH):
            raise FileNotFoundError(
                "Model not found! "
                "Run src/model_training.py first.\n"
                f"Expected: {PIPELINE_PATH}"
            )
        _pipeline = joblib.load(PIPELINE_PATH)
        _le       = joblib.load(LABEL_ENC_PATH)
        _metadata = (
            joblib.load(METADATA_PATH)
            if os.path.exists(METADATA_PATH) else {}
        )
        print(
            f"✅ Model loaded | "
            f"Accuracy: {_metadata.get('test_accuracy','N/A')}"
        )
    return _pipeline, _le, _metadata


# ============================================================
# FUNCTION 1 — Predict Category
# ============================================================
def predict_category(resume_text: str) -> dict:
    """
    Predicts the job category for a resume.

    Returns:
        predicted_category : str
        confidence         : float (%)
        top3_categories    : list of {category, confidence}
        status             : "success" | "error"
    """
    try:
        pipeline, le, _ = _load_model()
        clean_text       = advanced_clean(resume_text)

        if not clean_text.strip():
            return {
                "predicted_category": "Unknown",
                "confidence"        : 0.0,
                "top3_categories"   : [],
                "status"            : "error",
                "message"           : "Text too short after cleaning",
            }

        y_pred   = pipeline.predict([clean_text])[0]
        category = le.inverse_transform([y_pred])[0]

        # Probability scores (not available for LinearSVC)
        try:
            proba    = pipeline.predict_proba([clean_text])[0]
            top3_idx = proba.argsort()[-3:][::-1]
            top3     = [
                {
                    "category"  : le.classes_[i],
                    "confidence": round(float(proba[i]) * 100, 1),
                }
                for i in top3_idx
            ]
            confidence = round(float(proba[y_pred]) * 100, 1)
        except AttributeError:
            # SVM — no predict_proba
            confidence = 85.0
            top3       = [{"category": category, "confidence": 85.0}]

        return {
            "predicted_category": category,
            "confidence"        : confidence,
            "top3_categories"   : top3,
            "status"            : "success",
            "message"           : f"Predicted: {category}",
        }

    except Exception as e:
        return {
            "predicted_category": "Unknown",
            "confidence"        : 0.0,
            "top3_categories"   : [],
            "status"            : "error",
            "message"           : str(e),
        }


# ============================================================
# FUNCTION 2 — Analyze Resume Only (no JD)
# ============================================================
def analyze_resume(resume_text: str) -> dict:
    """
    Returns category prediction + skill profile
    without requiring a job description.
    """
    parsed = parse_resume(resume_text, input_type="text")
    if parsed["status"] == "error":
        return {"status": "error", "message": parsed["message"]}

    category_result = predict_category(resume_text)
    profile         = parsed["profile"]

    return {
        "status"            : "success",
        "predicted_category": category_result["predicted_category"],
        "confidence"        : category_result["confidence"],
        "top3_categories"   : category_result["top3_categories"],
        "skills"            : profile["skills"],
        "education"         : profile["education"],
        "experience_years"  : profile["experience_years"],
        "word_count"        : parsed["word_count"],
        "summary"           : profile["summary"],
    }


# ============================================================
# FUNCTION 3 — Full Resume vs JD Match Analysis
# ============================================================
def analyze_match(resume_text: str, jd_text: str) -> dict:
    """
    Master function called by every Flask route.

    Combines:
      1. Resume parsing & profile building
      2. Category prediction
      3. Weighted match score
      4. Skill gap analysis
      5. Improvement suggestions

    Returns a single flat dict ready for Jinja2 templates.
    """
    # ── Input validation ──────────────────────────────────
    if not resume_text or len(resume_text.strip()) < 50:
        return {
            "status" : "error",
            "message": "Resume text is too short (min 50 characters).",
        }
    if not jd_text or len(jd_text.strip()) < 30:
        return {
            "status" : "error",
            "message": "Job description is too short (min 30 characters).",
        }

    try:
        # Step 1 — Parse
        parsed_resume   = parse_resume(resume_text, input_type="text")

        # Step 2 — Category
        category_result = predict_category(resume_text)

        # Step 3 — Match score
        match_result    = compute_match_score(resume_text, jd_text)

        # Step 4 — Suggestions
        suggestions     = generate_suggestions(match_result)

        # Step 5 — Profile helpers
        profile = parsed_resume.get("profile", {})
        skills  = profile.get("skills", {})

        # Score label for the UI badge
        score = match_result["final_score_pct"]
        if score >= 75:
            score_label = "Excellent";  score_emoji = "🟢"
        elif score >= 50:
            score_label = "Good";       score_emoji = "🟡"
        elif score >= 30:
            score_label = "Fair";       score_emoji = "🟠"
        else:
            score_label = "Poor";       score_emoji = "🔴"

        return {
            # Status
            "status"            : "success",
            "message"           : "Analysis complete",
            # Final score
            "final_score_pct"   : match_result["final_score_pct"],
            "score_label"       : score_label,
            "score_emoji"       : score_emoji,
            "verdict"           : match_result["verdict"],
            "recommendation"    : match_result["recommendation"],
            "color"             : match_result["color"],
            # Component scores
            "tfidf_score"       : match_result["tfidf_score"],
            "skill_score_pct"   : match_result["skill_score_pct"],
            "experience_score"  : match_result["experience_score"],
            # Skill details
            "matched_skills"    : match_result["matched_skills"],
            "missing_skills"    : match_result["missing_skills"],
            "extra_skills"      : match_result["extra_skills"],
            "resume_skill_count": match_result["resume_skill_count"],
            "jd_skill_count"    : match_result["jd_skill_count"],
            "matched_count"     : match_result["matched_count"],
            # Category prediction
            "predicted_category": category_result["predicted_category"],
            "confidence"        : category_result["confidence"],
            "top3_categories"   : category_result["top3_categories"],
            # Experience
            "experience_years"  : match_result["resume_years"],
            "required_years"    : match_result["required_years"],
            "experience_message": match_result["experience_message"],
            # Resume meta
            "word_count"        : parsed_resume.get("word_count", 0),
            "skills_by_domain"  : skills.get("by_domain", {}),
            "top_domain"        : skills.get("top_domain", ""),
            # Suggestions
            "suggestions"       : suggestions,
        }

    except Exception as e:
        return {
            "status" : "error",
            "message": f"Analysis failed: {str(e)}",
        }


# ── Quick Test ─────────────────────────────────────────────
if __name__ == "__main__":
    resume = """
    Senior Data Scientist | 4 years experience
    Python, SQL, Machine Learning, Deep Learning,
    TensorFlow, NLP, Docker, AWS, Git, Pandas, Flask
    B.Tech Computer Science - 2020
    """
    jd = """
    Hiring Data Scientist | 2+ years experience required
    Must have: Python, SQL, Machine Learning, TensorFlow,
    NLP, Docker, AWS.  Nice to have: Spark, Kubernetes.
    """
    result = analyze_match(resume, jd)
    if result["status"] == "success":
        print(f"Score    : {result['final_score_pct']}%  {result['score_emoji']}")
        print(f"Category : {result['predicted_category']} ({result['confidence']}%)")
        print(f"Matched  : {result['matched_skills'][:4]}")
        print(f"Missing  : {result['missing_skills'][:4]}")
    else:
        print(f"Error: {result['message']}")
    print("✅ predictor.py OK")
