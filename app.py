# ============================================================
# app.py
# PURPOSE : Main Flask application — all routes, file upload,
#           predictions and template rendering
# ============================================================

import os
import sys
import uuid
import traceback
from datetime import datetime

from flask import (
    Flask, render_template, request,
    jsonify, redirect, url_for, flash, session
)
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predictor     import analyze_match, analyze_resume
from src.resume_parser import extract_text_from_pdf, validate_upload

# ── App Setup ──────────────────────────────────────────────
app            = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY", "ai_resume_screener_secret_2024"
)

UPLOAD_FOLDER  = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "uploads"
)
ALLOWED_EXT    = {"pdf", "txt"}

app.config["UPLOAD_FOLDER"]       = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"]  = 5 * 1024 * 1024   # 5 MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT
    )

def cleanup_upload(filepath: str) -> None:
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass

def format_skills(skills_list: list) -> list:
    return [s.title() for s in skills_list if s]

def get_score_color(score: float) -> str:
    if score >= 75:  return "success"
    if score >= 50:  return "warning"
    if score >= 30:  return "orange"
    return "danger"

def get_progress_color(score: float) -> str:
    if score >= 75:  return "#28a745"
    if score >= 50:  return "#ffc107"
    if score >= 30:  return "#fd7e14"
    return "#dc3545"

def enrich_result(result: dict) -> dict:
    """Adds UI-helper fields to a successful match result."""
    result["matched_skills"] = format_skills(result.get("matched_skills", []))
    result["missing_skills"] = format_skills(result.get("missing_skills", []))
    result["extra_skills"]   = format_skills(result.get("extra_skills",   []))
    result["score_color"]    = get_score_color(result["final_score_pct"])
    result["progress_color"] = get_progress_color(result["final_score_pct"])
    result["analysis_time"]  = datetime.now().strftime("%d %b %Y, %I:%M %p")
    return result


# ============================================================
# ROUTE 1 — Home  GET / POST
# ============================================================
@app.route("/", methods=["GET", "POST"])
def index():
    """Renders the input form; on POST runs match analysis."""
    if request.method == "GET":
        return render_template("index.html")

    resume_text = request.form.get("resume_text", "").strip()
    jd_text     = request.form.get("jd_text",     "").strip()

    # Validate
    errors = []
    if len(resume_text) < 50:
        errors.append("Resume text is too short. Paste your full resume.")
    if len(jd_text)     < 30:
        errors.append("Job description is too short.")
    if errors:
        for e in errors:
            flash(e, "danger")
        return render_template(
            "index.html",
            resume_text=resume_text,
            jd_text=jd_text,
        )

    try:
        result = analyze_match(resume_text, jd_text)
        if result["status"] == "error":
            flash(result["message"], "danger")
            return render_template(
                "index.html",
                resume_text=resume_text,
                jd_text=jd_text,
            )
        return render_template("match.html", result=enrich_result(result))

    except Exception:
        traceback.print_exc()
        flash("Something went wrong. Please try again.", "danger")
        return render_template(
            "index.html",
            resume_text=resume_text,
            jd_text=jd_text,
        )


# ============================================================
# ROUTE 2 — Upload PDF  POST /upload
# ============================================================
@app.route("/upload", methods=["POST"])
def upload_resume():
    """Handles PDF / TXT resume upload then runs analysis."""
    filepath = None
    try:
        if "resume_file" not in request.files:
            flash("No file uploaded.", "danger")
            return redirect(url_for("index"))

        file = request.files["resume_file"]
        if file.filename == "":
            flash("No file selected.", "danger")
            return redirect(url_for("index"))
        if not allowed_file(file.filename):
            flash("Only PDF and TXT files are supported.", "danger")
            return redirect(url_for("index"))

        # Size check
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        validation = validate_upload(file.filename, file_size)
        if not validation["valid"]:
            flash(validation["message"], "danger")
            return redirect(url_for("index"))

        # Save to uploads/
        safe_name = secure_filename(file.filename)
        filename  = f"{str(uuid.uuid4())[:8]}_{safe_name}"
        filepath  = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Extract text
        ext = safe_name.rsplit(".", 1)[1].lower()
        if ext == "pdf":
            resume_text = extract_text_from_pdf(filepath)
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                resume_text = f.read()

        jd_text = request.form.get("jd_text_upload", "").strip()
        if len(jd_text) < 30:
            flash("Please provide a job description.", "warning")
            cleanup_upload(filepath)
            return render_template("index.html", resume_text=resume_text)

        result = analyze_match(resume_text, jd_text)
        cleanup_upload(filepath)

        if result["status"] == "error":
            flash(result["message"], "danger")
            return redirect(url_for("index"))

        result["source"] = f"PDF: {safe_name}"
        return render_template("match.html", result=enrich_result(result))

    except Exception as e:
        cleanup_upload(filepath)
        flash(f"PDF error: {str(e)}. Try pasting text instead.", "danger")
        return redirect(url_for("index"))


# ============================================================
# ROUTE 3 — REST API  POST /api/match
# ============================================================
@app.route("/api/match", methods=["POST"])
def api_match():
    """
    JSON endpoint for resume matching.
    Body: {"resume_text": "...", "jd_text": "..."}
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400

        resume_text = data.get("resume_text", "").strip()
        jd_text     = data.get("jd_text",     "").strip()

        if len(resume_text) < 50:
            return jsonify({"status":"error","message":"resume_text too short"}), 400
        if len(jd_text)     < 30:
            return jsonify({"status":"error","message":"jd_text too short"}), 400

        return jsonify(analyze_match(resume_text, jd_text)), 200

    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500


# ============================================================
# ROUTE 4 — REST API  POST /api/analyze
# ============================================================
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Resume-only analysis (no JD). Returns category + skills."""
    try:
        data        = request.get_json(silent=True)
        resume_text = (data or {}).get("resume_text", "").strip()
        if len(resume_text) < 50:
            return jsonify({"status":"error","message":"Too short"}), 400
        return jsonify(analyze_resume(resume_text)), 200
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500


# ============================================================
# ROUTE 5 — Dashboard  GET /dashboard
# ============================================================
@app.route("/dashboard")
def dashboard():
    """HR analytics dashboard with sample statistics."""
    stats = {
        "total_screened" : 247,
        "strong_matches" : 89,
        "avg_score"      : 62.4,
        "top_category"   : "Data Science",
        "recent_analyses": [
            {"name":"Resume #1","score":85,"category":"Data Science",   "verdict":"Strong Match"},
            {"name":"Resume #2","score":72,"category":"Web Developer",  "verdict":"Moderate Match"},
            {"name":"Resume #3","score":45,"category":"Data Analyst",   "verdict":"Weak Match"},
            {"name":"Resume #4","score":91,"category":"ML Engineer",    "verdict":"Strong Match"},
            {"name":"Resume #5","score":33,"category":"Java Developer", "verdict":"Poor Match"},
        ],
        "category_dist": {
            "Data Science"  : 45,
            "Web Developer" : 38,
            "Data Analyst"  : 29,
            "ML Engineer"   : 22,
            "Java Developer": 18,
            "Others"        : 95,
        },
    }
    return render_template("dashboard.html", stats=stats)


# ============================================================
# ROUTE 6 — Health Check  GET /health
# ============================================================
@app.route("/health")
def health():
    return jsonify({
        "status"   : "healthy",
        "app"      : "AI Resume Screener",
        "version"  : "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }), 200


# ── Error Handlers ─────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    flash("Page not found.", "warning")
    return redirect(url_for("index"))

@app.errorhandler(413)
def file_too_large(e):
    flash("File too large. Maximum size is 5 MB.", "danger")
    return redirect(url_for("index"))

@app.errorhandler(500)
def server_error(e):
    flash("Internal server error. Please try again.", "danger")
    return redirect(url_for("index"))


# ── Context Processor (global template vars) ───────────────
@app.context_processor
def inject_globals():
    return {
        "app_name"    : "AI Resume Screener",
        "app_version" : "1.0.0",
        "current_year": datetime.now().year,
    }


# ── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    print("=" * 50)
    print("  🤖 AI Resume Screener")
    print(f"  URL   : http://127.0.0.1:{port}")
    print(f"  Debug : {debug}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=debug)
