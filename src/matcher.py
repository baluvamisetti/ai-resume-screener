# ============================================================
# src/matcher.py
# PURPOSE : Core matching engine
#           1. TF-IDF + Cosine Similarity  (50%)
#           2. Skill keyword overlap        (35%)
#           3. Experience years comparison  (15%)
#           Outputs a final weighted match score 0–100%
# ============================================================

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity

from src.text_cleaner    import advanced_clean
from src.skill_extractor import compute_skill_match, extract_experience_years

# ── Scoring Weights ────────────────────────────────────────
WEIGHTS = {
    "tfidf_cosine": 0.50,   # 50% — text similarity
    "skill_match" : 0.35,   # 35% — skill keyword overlap
    "experience"  : 0.15,   # 15% — years of experience
}


# ============================================================
# FUNCTION 1 — TF-IDF Cosine Similarity
# ============================================================
def compute_tfidf_similarity(resume_text: str,
                              jd_text: str) -> float:
    """
    Converts both texts to TF-IDF vectors and returns
    the cosine similarity score (0.0 – 1.0).
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    try:
        vectorizer  = TfidfVectorizer(
            max_features  = 5000,
            ngram_range   = (1, 2),
            min_df        = 1,
            sublinear_tf  = True,
            strip_accents = "unicode",
        )
        tfidf_matrix = vectorizer.fit_transform([
            advanced_clean(resume_text),
            advanced_clean(jd_text),
        ])
        score = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:2]
        )[0][0]
        return round(float(score), 4)
    except Exception:
        return 0.0


# ============================================================
# FUNCTION 2 — Experience Score
# ============================================================
def compute_experience_score(resume_text: str,
                              jd_text: str) -> dict:
    """
    Compares years of experience in resume vs JD.

    Score logic:
      resume >= required  → 1.0
      resume <  required  → proportional
      no requirement      → 0.8 (neutral)
    """
    resume_years   = extract_experience_years(resume_text)
    required_years = extract_experience_years(jd_text)

    if required_years == 0:
        return {
            "score"         : 0.8,
            "resume_years"  : resume_years,
            "required_years": 0,
            "message"       : "No experience requirement in JD",
        }
    if resume_years == 0:
        return {
            "score"         : 0.5,
            "resume_years"  : 0,
            "required_years": required_years,
            "message"       : "Experience not stated in resume",
        }
    if resume_years >= required_years:
        score   = 1.0
        message = (
            f"✅ Meets requirement "
            f"({resume_years}yr >= {required_years}yr)"
        )
    else:
        score   = round(resume_years / required_years, 2)
        message = (
            f"⚠️  Below requirement "
            f"({resume_years}yr < {required_years}yr)"
        )
    return {
        "score"         : score,
        "resume_years"  : resume_years,
        "required_years": required_years,
        "message"       : message,
    }


# ============================================================
# FUNCTION 3 — Master Match Score
# ============================================================
def compute_match_score(resume_text: str,
                         jd_text: str) -> dict:
    """
    Computes the final weighted match score.

    Returns a dict with scores, skill lists, verdicts
    and all data needed to render the results page.
    """
    tfidf_score = compute_tfidf_similarity(resume_text, jd_text)
    skill_data  = compute_skill_match(resume_text, jd_text)
    skill_score = skill_data["skill_match_pct"] / 100.0
    exp_data    = compute_experience_score(resume_text, jd_text)
    exp_score   = exp_data["score"]

    final_pct = round(
        (
            WEIGHTS["tfidf_cosine"] * tfidf_score +
            WEIGHTS["skill_match"]  * skill_score +
            WEIGHTS["experience"]   * exp_score
        ) * 100,
        1,
    )

    # Verdict bands
    if final_pct >= 75:
        verdict        = "🟢 Strong Match"
        recommendation = "Highly recommended for interview"
        color          = "green"
    elif final_pct >= 50:
        verdict        = "🟡 Moderate Match"
        recommendation = "Consider with skill development"
        color          = "orange"
    elif final_pct >= 30:
        verdict        = "🟠 Weak Match"
        recommendation = "Significant skill gaps present"
        color          = "darkorange"
    else:
        verdict        = "🔴 Poor Match"
        recommendation = "Resume does not match this role"
        color          = "red"

    return {
        # Final result
        "final_score_pct"   : final_pct,
        "verdict"           : verdict,
        "recommendation"    : recommendation,
        "color"             : color,
        # Component scores (as percentages)
        "tfidf_score"       : round(tfidf_score * 100, 1),
        "skill_score_pct"   : skill_data["skill_match_pct"],
        "experience_score"  : round(exp_score * 100, 1),
        # Skill details
        "matched_skills"    : skill_data["matched_skills"],
        "missing_skills"    : skill_data["missing_skills"],
        "extra_skills"      : skill_data["extra_skills"],
        "resume_skill_count": skill_data["resume_skill_count"],
        "jd_skill_count"    : skill_data["jd_skill_count"],
        "matched_count"     : skill_data["matched_count"],
        # Experience details
        "resume_years"      : exp_data["resume_years"],
        "required_years"    : exp_data["required_years"],
        "experience_message": exp_data["message"],
        # Weights used
        "weights"           : WEIGHTS,
    }


# ============================================================
# FUNCTION 4 — Rank Multiple Resumes
# ============================================================
def rank_resumes(resumes: list, jd_text: str) -> list:
    """
    Scores and ranks a list of resumes against one JD.

    Args:
        resumes  : [{"name": str, "text": str}, ...]
        jd_text  : job description string

    Returns:
        list of result dicts sorted by score descending
    """
    ranked = []
    for i, resume in enumerate(resumes):
        score_data = compute_match_score(
            resume.get("text", ""), jd_text
        )
        ranked.append({
            "rank"           : 0,
            "name"           : resume.get("name", f"Candidate {i+1}"),
            "final_score_pct": score_data["final_score_pct"],
            "verdict"        : score_data["verdict"],
            "matched_skills" : score_data["matched_skills"],
            "missing_skills" : score_data["missing_skills"],
            "resume_years"   : score_data["resume_years"],
            "tfidf_score"    : score_data["tfidf_score"],
            "skill_score_pct": score_data["skill_score_pct"],
        })

    ranked.sort(key=lambda x: x["final_score_pct"], reverse=True)
    for i, item in enumerate(ranked):
        item["rank"] = i + 1
    return ranked


# ============================================================
# FUNCTION 5 — Generate Suggestions
# ============================================================
def generate_suggestions(match_result: dict) -> list:
    """
    Returns actionable improvement tips based on match gaps.
    """
    suggestions = []
    score   = match_result.get("final_score_pct", 0)
    missing = match_result.get("missing_skills", [])

    if missing:
        suggestions.append(
            f"📚 Learn missing skills: {', '.join(missing[:5])}"
        )
    res_yrs = match_result.get("resume_years", 0)
    req_yrs = match_result.get("required_years", 0)
    if req_yrs > 0 and res_yrs < req_yrs:
        suggestions.append(
            f"💼 Gain {req_yrs - res_yrs} more year(s) of experience"
        )
    if match_result.get("tfidf_score", 0) < 30:
        suggestions.append(
            "✍️  Mirror keywords from the job description in your resume"
        )
    if score < 50:
        suggestions.append(
            "🎯 Tailor your resume specifically for this role"
        )
    if score < 75:
        suggestions.append(
            "📝 Add quantifiable achievements to strengthen your profile"
        )
    if not suggestions:
        suggestions.append(
            "🌟 Excellent match! Focus on interview preparation"
        )
    return suggestions


# ── Quick Test ─────────────────────────────────────────────
if __name__ == "__main__":
    resume = """
    Data Scientist with 3 years experience.
    Python, SQL, Machine Learning, TensorFlow,
    NLP, Docker, AWS, Git, Pandas, Flask
    """
    jd = """
    Hiring Data Scientist, 2+ years experience required.
    Must have: Python, SQL, Machine Learning, TensorFlow,
    NLP, Docker, AWS. Nice to have: Spark, Kubernetes.
    """
    result = compute_match_score(resume, jd)
    print(f"Final Score : {result['final_score_pct']}%")
    print(f"Verdict     : {result['verdict']}")
    for s in generate_suggestions(result):
        print(f"  {s}")
    print("✅ matcher.py OK")
