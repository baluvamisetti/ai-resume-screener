# ============================================================
# src/skill_extractor.py
# PURPOSE : Extract skills, education, experience and
#           named entities from resume / JD text
#           Uses spaCy NER + custom skills taxonomy
# ============================================================

import re
import spacy
from collections import defaultdict

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found!\n"
        "Fix: python -m spacy download en_core_web_sm"
    )

# ============================================================
# MASTER SKILLS TAXONOMY  (11 domains, 200+ skills)
# ============================================================
SKILLS_TAXONOMY = {

    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++",
        "c#", "r", "scala", "go", "ruby", "php", "swift",
        "kotlin", "rust", "matlab", "perl", "bash", "shell",
    ],

    "data_science_ml": [
        "machine learning", "deep learning", "neural network",
        "natural language processing", "nlp", "computer vision",
        "reinforcement learning", "transfer learning",
        "feature engineering", "model training", "model evaluation",
        "hyperparameter tuning", "cross validation", "a/b testing",
        "statistical analysis", "data mining", "predictive modeling",
        "classification", "regression", "clustering",
        "dimensionality reduction", "time series", "forecasting",
    ],

    "ml_frameworks": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "xgboost", "lightgbm", "catboost", "hugging face",
        "transformers", "spacy", "nltk", "gensim", "opencv",
        "fastai", "mlflow", "kubeflow", "airflow",
    ],

    "data_tools": [
        "pandas", "numpy", "scipy", "matplotlib", "seaborn",
        "plotly", "bokeh", "tableau", "power bi", "looker",
        "excel", "google sheets", "jupyter", "notebook",
    ],

    "databases": [
        "sql", "mysql", "postgresql", "sqlite", "oracle",
        "mongodb", "cassandra", "redis", "elasticsearch",
        "dynamodb", "firebase", "hive", "snowflake",
        "bigquery", "redshift",
    ],

    "big_data": [
        "hadoop", "spark", "pyspark", "kafka", "hive",
        "hdfs", "mapreduce", "flink", "databricks",
    ],

    "cloud": [
        "aws", "azure", "gcp", "google cloud",
        "amazon web services", "ec2", "s3", "lambda",
        "sagemaker", "azure ml", "heroku", "render", "vercel",
    ],

    "devops_mlops": [
        "docker", "kubernetes", "jenkins", "git", "github",
        "gitlab", "ci/cd", "terraform", "ansible", "linux",
        "mlflow", "dvc", "wandb",
    ],

    "web_development": [
        "html", "css", "react", "angular", "vue", "node.js",
        "flask", "django", "fastapi", "spring", "rest api",
        "graphql", "bootstrap", "tailwind", "jquery",
    ],

    "soft_skills": [
        "communication", "leadership", "teamwork",
        "problem solving", "critical thinking",
        "project management", "agile", "scrum",
        "collaboration", "presentation", "mentoring",
    ],

    "certifications": [
        "aws certified", "google certified", "azure certified",
        "pmp", "scrum master", "tensorflow developer",
        "tableau certified",
    ],
}

# Flat lookup structures
ALL_SKILLS_FLAT = set()
SKILL_TO_DOMAIN = {}
for domain, skills in SKILLS_TAXONOMY.items():
    for skill in skills:
        s = skill.lower()
        ALL_SKILLS_FLAT.add(s)
        SKILL_TO_DOMAIN[s] = domain

# ── Education Keywords ────────────────────────────────────
EDUCATION_KEYWORDS = {
    "degrees": [
        "b.tech", "btech", "b.e", "be", "b.sc", "bsc",
        "m.tech", "mtech", "m.sc", "msc", "mba", "phd",
        "bachelor", "master", "doctorate", "diploma",
        "associate", "undergraduate", "postgraduate",
    ],
    "fields": [
        "computer science", "data science", "information technology",
        "software engineering", "electrical engineering",
        "mathematics", "statistics", "physics",
        "artificial intelligence", "machine learning",
        "information systems",
    ],
}

# ── Experience Regex Patterns ─────────────────────────────
EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*years?\s*(?:of\s*)?experience",
    r"(\d+)\+?\s*yrs?\s*(?:of\s*)?experience",
    r"experience\s*(?:of\s*)?(\d+)\+?\s*years?",
]


# ============================================================
# FUNCTION 1 — Extract Skills
# ============================================================
def extract_skills(text: str) -> dict:
    """
    Scans text against SKILLS_TAXONOMY using regex word
    boundaries for single words and substring match for
    multi-word phrases.

    Returns:
        {
            all_skills  : sorted list of found skills,
            by_domain   : {domain: [skills]},
            skill_count : int,
            top_domain  : str (domain with most skills)
        }
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "all_skills" : [],
            "by_domain"  : {},
            "skill_count": 0,
            "top_domain" : None,
        }

    text_lower      = text.lower()
    found_skills    = []
    found_by_domain = defaultdict(list)

    for skill in ALL_SKILLS_FLAT:
        if " " in skill:
            if skill in text_lower:
                found_skills.append(skill)
                d = SKILL_TO_DOMAIN.get(skill, "other")
                if skill not in found_by_domain[d]:
                    found_by_domain[d].append(skill)
        else:
            if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                found_skills.append(skill)
                d = SKILL_TO_DOMAIN.get(skill, "other")
                if skill not in found_by_domain[d]:
                    found_by_domain[d].append(skill)

    found_skills = list(dict.fromkeys(found_skills))
    top_domain   = (
        max(found_by_domain, key=lambda k: len(found_by_domain[k]))
        if found_by_domain else None
    )

    return {
        "all_skills" : sorted(found_skills),
        "by_domain"  : dict(found_by_domain),
        "skill_count": len(found_skills),
        "top_domain" : top_domain,
    }


# ============================================================
# FUNCTION 2 — Extract Education
# ============================================================
def extract_education(text: str) -> dict:
    """Extracts degree types, fields of study, and years."""
    text_lower = text.lower()

    found_degrees = [
        d for d in EDUCATION_KEYWORDS["degrees"]
        if re.search(r"\b" + re.escape(d) + r"\b", text_lower)
    ]
    found_fields = [
        f for f in EDUCATION_KEYWORDS["fields"]
        if f in text_lower
    ]
    years = re.findall(r"\b(19|20)\d{2}\b", text)

    return {
        "degrees": found_degrees,
        "fields" : found_fields,
        "years"  : years,
    }


# ============================================================
# FUNCTION 3 — Extract Experience Years
# ============================================================
def extract_experience_years(text: str) -> int:
    """Returns the maximum years of experience found in text."""
    text_lower  = text.lower()
    years_found = []
    for pattern in EXPERIENCE_PATTERNS:
        for match in re.findall(pattern, text_lower):
            try:
                years_found.append(int(match))
            except ValueError:
                pass
    return max(years_found) if years_found else 0


# ============================================================
# FUNCTION 4 — Extract Named Entities (spaCy)
# ============================================================
def extract_entities(text: str) -> dict:
    """
    Extracts PERSON, ORG, GPE, DATE entities using spaCy NER.
    Limits to first 5000 chars for performance.
    """
    doc      = nlp(text[:5000])
    entities = defaultdict(list)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]:
            val = ent.text.strip()
            if val and val not in entities[ent.label_]:
                entities[ent.label_].append(val)
    return dict(entities)


# ============================================================
# FUNCTION 5 — Compute Skill Match (Resume vs JD)
# ============================================================
def compute_skill_match(resume_text: str, jd_text: str) -> dict:
    """
    Compares skills found in resume vs job description.
    Returns matched, missing, extra skills and match %.
    """
    resume_skills = set(extract_skills(resume_text)["all_skills"])
    jd_skills     = set(extract_skills(jd_text)["all_skills"])

    matched   = resume_skills.intersection(jd_skills)
    missing   = jd_skills.difference(resume_skills)
    extra     = resume_skills.difference(jd_skills)
    match_pct = (
        round(len(matched) / len(jd_skills) * 100, 2)
        if jd_skills else 0.0
    )

    return {
        "matched_skills"     : sorted(list(matched)),
        "missing_skills"     : sorted(list(missing)),
        "extra_skills"       : sorted(list(extra)),
        "resume_skill_count" : len(resume_skills),
        "jd_skill_count"     : len(jd_skills),
        "matched_count"      : len(matched),
        "skill_match_pct"    : match_pct,
    }


# ============================================================
# FUNCTION 6 — Build Full Resume Profile
# ============================================================
def build_resume_profile(text: str) -> dict:
    """
    Combines all extractors into one unified profile dict.
    Used by resume_parser and predictor.
    """
    skills    = extract_skills(text)
    education = extract_education(text)
    exp_years = extract_experience_years(text)
    entities  = extract_entities(text)

    return {
        "skills"          : skills,
        "education"       : education,
        "experience_years": exp_years,
        "entities"        : entities,
        "summary": {
            "total_skills"    : skills["skill_count"],
            "top_domain"      : skills["top_domain"],
            "degrees_found"   : len(education["degrees"]),
            "experience_years": exp_years,
        },
    }


# ── Quick Test ─────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    Data Scientist | 4 years experience
    Python, TensorFlow, PyTorch, SQL, Machine Learning,
    Deep Learning, NLP, Docker, AWS, Git, Pandas, NumPy,
    Scikit-learn, Tableau, Power BI, Flask
    B.Tech Computer Science - 2020
    """
    profile = build_resume_profile(sample)
    print(f"Skills found : {profile['summary']['total_skills']}")
    print(f"Top domain   : {profile['summary']['top_domain']}")
    print(f"Exp years    : {profile['summary']['experience_years']}")
    print("✅ skill_extractor.py OK")
