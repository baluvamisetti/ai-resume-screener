# ============================================================
# src/resume_parser.py
# PURPOSE : Parse resumes from PDF files or plain text
#           Validates uploads and builds clean text for NLP
# ============================================================

import os
import re
import pdfplumber

from src.skill_extractor import build_resume_profile
from src.text_cleaner    import normalize_whitespace


# ── Extract Raw Text from PDF ─────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF using pdfplumber.
    Falls back to word-level extraction per page.

    Raises:
        FileNotFoundError : if pdf_path doesn't exist
        ValueError        : if no text could be extracted
        RuntimeError      : for pdfplumber failures
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("File must be a .pdf")

    extracted_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_text.append(page_text)
                else:
                    words = page.extract_words()
                    if words:
                        extracted_text.append(
                            " ".join([w["text"] for w in words])
                        )
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {str(e)}")

    full_text = "\n".join(extracted_text)
    if not full_text.strip():
        raise ValueError(
            "No text extracted. "
            "The PDF may be scanned/image-based. "
            "Please paste text directly."
        )
    return full_text


# ── Clean Raw Extracted Text ──────────────────────────────
def clean_extracted_text(text: str) -> str:
    """
    Removes non-printable characters, normalizes whitespace,
    and filters out symbol-only lines.
    """
    if not isinstance(text, str):
        return ""
    text  = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
    text  = normalize_whitespace(text)
    lines = text.split("\n")
    cleaned = [
        line.strip() for line in lines
        if len(line.strip()) > 2 and
           sum(c.isalpha() for c in line) >= 2
    ]
    return " ".join(cleaned)


# ── Master Resume Parser ──────────────────────────────────
def parse_resume(input_data, input_type: str = "text") -> dict:
    """
    Unified entry point for parsing resumes.

    Args:
        input_data : str path (pdf) or str content (text)
        input_type : "text" | "pdf"

    Returns dict with keys:
        raw_text, clean_text, profile, word_count,
        status ("success"/"error"), message
    """
    raw_text = ""
    try:
        if input_type == "pdf":
            raw_text = extract_text_from_pdf(input_data)
        elif input_type == "text":
            if not isinstance(input_data, str):
                raise ValueError("input_data must be a string")
            raw_text = input_data
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        if len(raw_text.strip()) < 50:
            return {
                "raw_text"  : raw_text,
                "clean_text": "",
                "profile"   : {},
                "word_count": 0,
                "status"    : "error",
                "message"   : "Resume text too short.",
            }

        clean_text = clean_extracted_text(raw_text)
        profile    = build_resume_profile(clean_text)
        word_count = len(clean_text.split())

        return {
            "raw_text"  : raw_text,
            "clean_text": clean_text,
            "profile"   : profile,
            "word_count": word_count,
            "status"    : "success",
            "message"   : f"Parsed successfully ({word_count} words)",
        }

    except Exception as e:
        return {
            "raw_text"  : raw_text,
            "clean_text": "",
            "profile"   : {},
            "word_count": 0,
            "status"    : "error",
            "message"   : str(e),
        }


# ── Parse Job Description ─────────────────────────────────
def parse_job_description(jd_text: str) -> dict:
    """Parses a raw job description string."""
    if not isinstance(jd_text, str) or len(jd_text.strip()) < 20:
        return {
            "clean_text": "",
            "profile"   : {},
            "word_count": 0,
            "status"    : "error",
            "message"   : "Job description too short.",
        }
    clean_text = clean_extracted_text(jd_text)
    profile    = build_resume_profile(clean_text)
    return {
        "clean_text": clean_text,
        "profile"   : profile,
        "word_count": len(clean_text.split()),
        "status"    : "success",
        "message"   : "JD parsed",
    }


# ── Validate File Upload ──────────────────────────────────
def validate_upload(filename: str, file_size_bytes: int) -> dict:
    """
    Validates file extension (.pdf/.txt) and size (max 5 MB).
    Returns {"valid": bool, "message": str}.
    """
    MAX_SIZE = 5 * 1024 * 1024  # 5 MB
    ext      = os.path.splitext(filename.lower())[1]

    if ext not in {".pdf", ".txt"}:
        return {
            "valid"  : False,
            "message": f"'{ext}' not supported. Use PDF or TXT.",
        }
    if file_size_bytes > MAX_SIZE:
        mb = file_size_bytes / (1024 * 1024)
        return {
            "valid"  : False,
            "message": f"File too large ({mb:.1f} MB). Max 5 MB.",
        }
    return {"valid": True, "message": "File is valid."}


# ── Quick Test ─────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    Jane Smith | jane@email.com
    Data Scientist | 3 years experience
    Skills: Python, SQL, Machine Learning, TensorFlow,
            Docker, AWS, Git, Pandas, Flask
    Education: B.Tech Computer Science - 2021
    Projects: NLP chatbot, Image classifier
    """
    result = parse_resume(sample, input_type="text")
    print(f"Status      : {result['status']}")
    print(f"Word count  : {result['word_count']}")
    print(f"Skills found: {result['profile']['summary']['total_skills']}")
    print("✅ resume_parser.py OK")
