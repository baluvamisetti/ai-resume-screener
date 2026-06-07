# ============================================================
# src/text_cleaner.py
# PURPOSE : All text preprocessing functions
#           Cleaning, normalization, tokenization
# ============================================================

import re
import nltk
from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt_tab", quiet=True)

lemmatizer    = WordNetLemmatizer()
STOP_WORDS    = set(stopwords.words("english"))
EXTRA_STOPS   = {
    "resume", "cv", "curriculum", "vitae", "objective",
    "summary", "experience", "education", "skills",
    "responsibilities", "worked", "working", "work",
    "job", "company", "organization", "team",
    "project", "projects", "year", "years", "month", "months",
}
ALL_STOP_WORDS = STOP_WORDS.union(EXTRA_STOPS)


# ── Basic Cleaner ──────────────────────────────────────────
def basic_clean(text: str) -> str:
    """
    Removes URLs, emails, phone numbers,
    special characters and extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+",           " ", text)
    text = re.sub(r"\S+@\S+",                            " ", text)
    text = re.sub(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]"," ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]",                    " ", text)
    text = re.sub(r"\b\d+\b",                            " ", text)
    text = re.sub(r"\s+",                                " ", text).strip()
    return text


# ── Advanced NLP Cleaner ───────────────────────────────────
def advanced_clean(text: str, lemmatize: bool = True) -> str:
    """
    Tokenizes, removes stopwords and lemmatizes text.
    """
    text   = basic_clean(text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in ALL_STOP_WORDS and len(w) > 2]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


# ── Whitespace Normalizer ──────────────────────────────────
def normalize_whitespace(text: str) -> str:
    """Collapses all whitespace into single spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Section Extractor ──────────────────────────────────────
def extract_sections(text: str) -> dict:
    """
    Attempts to extract major resume sections
    (education, experience, skills, projects, summary).
    """
    sections = {
        "education" : "",
        "experience": "",
        "skills"    : "",
        "projects"  : "",
        "summary"   : "",
    }
    patterns = {
        "education" : r"education[:\s]*(.*?)(?=experience|skills|projects|$)",
        "experience": r"experience[:\s]*(.*?)(?=education|skills|projects|$)",
        "skills"    : r"skills[:\s]*(.*?)(?=education|experience|projects|$)",
        "projects"  : r"projects[:\s]*(.*?)(?=education|experience|skills|$)",
        "summary"   : r"(?:summary|objective)[:\s]*(.*?)(?=education|experience|skills|$)",
    }
    text_lower = text.lower()
    for section, pattern in patterns.items():
        match = re.search(pattern, text_lower, re.DOTALL)
        if match:
            sections[section] = match.group(1).strip()[:500]
    return sections


# ── Word Frequency ─────────────────────────────────────────
def get_word_freq(text: str) -> dict:
    """Returns word → frequency dictionary, sorted descending."""
    freq = {}
    for word in text.split():
        freq[word] = freq.get(word, 0) + 1
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))


# ── Quick Test ─────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    John Doe | john.doe@email.com | +1-555-123-4567
    https://linkedin.com/in/johndoe
    Data Scientist with 3 years of Python and Machine Learning experience.
    SKILLS: Python, SQL, TensorFlow, Scikit-learn, Docker
    EDUCATION: B.Tech Computer Science - XYZ University - 2021
    """
    print("basic_clean   :", basic_clean(sample)[:100])
    print("advanced_clean:", advanced_clean(sample)[:100])
    print("sections      :", list(extract_sections(sample).keys()))
    print("✅ text_cleaner.py OK")
