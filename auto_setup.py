# ============================================================
# auto_setup.py
# PURPOSE : One script to do everything:
#           1. Generate sample datasets
#           2. Clean and save processed data
#           3. Train and save the model
#           4. Verify everything is ready
# USAGE   : python auto_setup.py
# ============================================================

import os
import sys
import random
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("  AI Resume Screener — Auto Setup")
print("  This will take 2-3 minutes. Please wait...")
print("=" * 55)

# ── Step 0: Create all folders ─────────────────────────────
print("\n📁 Step 1/4 — Creating folders...")
folders = [
    "data/raw",
    "data/processed",
    "model",
    "screenshots",
    "uploads",
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"   ✅ {folder}/")


# ── Step 1: Generate Datasets ──────────────────────────────
print("\n📊 Step 2/4 — Generating sample datasets...")

import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# Resume templates per category
RESUME_DATA = {
    "Data Science": [
        "Data Scientist with {y} years experience. Skills: Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Tableau, Power BI, NLP, Computer Vision, Docker, AWS, Git, Flask, Jupyter. B.Tech Computer Science {g}. Built predictive models and ML pipelines. Strong in statistical analysis, feature engineering and model deployment.",
        "Senior Data Scientist {y} years. Python, R, SQL, Machine Learning, Neural Networks, TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Data Visualization, Tableau, Power BI, AWS, Azure, Docker, Git, NLP, Time Series. M.Tech Data Science {g}. End-to-end ML pipelines and production deployments.",
        "Data Scientist specializing in NLP and Computer Vision. {y} years experience. Python, TensorFlow, PyTorch, Hugging Face, Transformers, spaCy, NLTK, OpenCV, SQL, Docker, AWS, Git, MLflow, Pandas, NumPy. B.Tech CSE {g}.",
    ],
    "Web Developer": [
        "Full Stack Developer {y} years. JavaScript, React, Node.js, HTML, CSS, Bootstrap, Tailwind, Vue, Angular, MongoDB, SQL, PostgreSQL, REST API, GraphQL, Git, Docker, AWS. B.Tech IT {g}. Built scalable web applications.",
        "Frontend Developer {y} years. HTML, CSS, JavaScript, TypeScript, React, Angular, Bootstrap, Tailwind, jQuery, Git, Webpack, REST API, Node.js, MongoDB, Firebase. B.Tech CSE {g}. Responsive UI and user-friendly interfaces.",
        "Backend Web Developer {y} years. Node.js, Express, Python, Django, Flask, REST API, SQL, MySQL, MongoDB, Redis, Docker, AWS, Git, CI/CD. B.Tech IT {g}.",
    ],
    "Data Analyst": [
        "Data Analyst {y} years. SQL, Excel, Power BI, Tableau, Python, Pandas, NumPy, Matplotlib, Seaborn, Statistics, Data Visualization, ETL, Data Cleaning, Google Sheets, R. B.Sc Statistics {g}. Analyzed large datasets and created executive dashboards.",
        "Business Data Analyst {y} years. Excel, SQL, Power BI, Tableau, Python, R, Statistics, Data Mining, Visualization, Pivot Tables, VBA, Google Analytics, SPSS. MBA {g}. Delivered actionable business insights.",
        "Data Analyst with expertise in SQL and visualization. {y} years. SQL, MySQL, Power BI, Tableau, Excel, Python, Pandas, Statistics, ETL, Data Warehousing, Looker. B.Sc Mathematics {g}.",
    ],
    "Machine Learning Engineer": [
        "ML Engineer {y} years. Python, TensorFlow, PyTorch, Keras, Scikit-learn, Machine Learning, Deep Learning, NLP, MLflow, Docker, Kubernetes, AWS SageMaker, Git, SQL, Spark, Airflow. M.Tech AI {g}.",
        "Machine Learning Engineer. Python, Scikit-learn, TensorFlow, PyTorch, NLP, Deep Learning, Feature Engineering, Model Deployment, Docker, AWS, MLflow, DVC, Git, SQL, Pandas, NumPy. {y} years. B.Tech CSE {g}.",
        "Applied ML Engineer {y} years. Python, Machine Learning, Computer Vision, NLP, TensorFlow, OpenCV, Scikit-learn, Docker, Kubernetes, AWS, Git, SQL, Pandas, MLflow, Kubeflow. M.Tech CSE {g}.",
    ],
    "Java Developer": [
        "Java Developer {y} years. Java, Spring Boot, Hibernate, REST API, SQL, MySQL, PostgreSQL, Maven, Git, Docker, Microservices, AWS, Jenkins, JUnit. B.Tech CSE {g}. Enterprise backend systems.",
        "Senior Java Developer. Java, J2EE, Spring, Spring Boot, Hibernate, REST API, MySQL, Oracle, Git, Docker, Kubernetes, Jenkins, Maven, Kafka. {y} years. MCA {g}.",
        "Java Backend Developer {y} years. Core Java, Spring Boot, REST API, Microservices, SQL, PostgreSQL, Redis, Docker, AWS, Git, JUnit, Mockito, Maven. B.Tech IT {g}.",
    ],
    "Python Developer": [
        "Python Developer {y} years. Python, Django, Flask, FastAPI, REST API, SQL, PostgreSQL, MongoDB, Redis, Docker, AWS, Git, Celery, NumPy, Pandas. B.Tech CSE {g}.",
        "Backend Python Developer. Python, Flask, Django, REST API, GraphQL, PostgreSQL, MySQL, MongoDB, Docker, Git, AWS, Linux, Redis, Celery, pytest. {y} years. B.Tech IT {g}.",
        "Python Full Stack Developer {y} years. Python, Django, React, HTML, CSS, JavaScript, REST API, SQL, PostgreSQL, Docker, Git, AWS, Celery, Redis. B.Tech CSE {g}.",
    ],
    "DevOps Engineer": [
        "DevOps Engineer {y} years. Docker, Kubernetes, Jenkins, Git, CI/CD, AWS, Azure, GCP, Terraform, Ansible, Linux, Bash, Python, Prometheus, Grafana. B.Tech CSE {g}.",
        "Cloud DevOps Engineer. AWS, Azure, Docker, Kubernetes, Jenkins, Terraform, Ansible, Git, Linux, Bash, Python, CI/CD, Monitoring, Prometheus, ELK Stack. {y} years. B.Tech {g}.",
        "Site Reliability Engineer {y} years. Linux, Docker, Kubernetes, AWS, Terraform, Git, Python, Bash, Jenkins, CI/CD, Prometheus, Grafana, Ansible. B.Tech CSE {g}.",
    ],
    "Android Developer": [
        "Android Developer {y} years. Java, Kotlin, Android SDK, XML, REST API, Firebase, SQLite, Git, Material Design, MVVM, Retrofit, Coroutines. B.Tech CSE {g}. Multiple Play Store apps.",
        "Mobile Android Developer. Kotlin, Java, Android Studio, Jetpack Compose, REST API, Firebase, Room Database, Git, MVVM, Retrofit, Coroutines, Dagger. {y} years. B.Tech IT {g}.",
    ],
    "SQL Developer": [
        "SQL Developer {y} years. SQL, MySQL, PostgreSQL, Oracle, SQL Server, Stored Procedures, Triggers, Views, Indexing, ETL, SSIS, Power BI, Tableau, Python. B.Sc CS {g}.",
        "Database Developer and DBA. SQL, MySQL, Oracle, PostgreSQL, MongoDB, ETL, Data Warehousing, Stored Procedures, Query Optimization, Power BI, Excel. {y} years. MCA {g}.",
    ],
    "Testing": [
        "QA Engineer {y} years. Manual Testing, Automation Testing, Selenium, Java, Python, TestNG, JUnit, REST API Testing, Postman, JIRA, Git, Agile, Scrum. B.Tech CSE {g}.",
        "Software Test Engineer. Selenium, Python, Java, TestNG, API Testing, Postman, JIRA, Git, Agile, Performance Testing, JMeter, Manual Testing, BDD. {y} years. B.Tech IT {g}.",
    ],
}

# Job Description templates
JD_DATA = [
    ("Data Science",              "Hiring Data Scientist with {y}+ years. Required: Python, SQL, Machine Learning, Deep Learning, TensorFlow, NLP, Docker, AWS, Git, Pandas, Scikit-learn. Nice to have: Spark, Kubernetes, MLflow, PyTorch. B.Tech or M.Tech CS preferred."),
    ("Web Developer",             "Full Stack Web Developer {y}+ years. Required: JavaScript, React, Node.js, HTML, CSS, SQL, Git, REST API, Docker, MongoDB. Nice to have: TypeScript, AWS, GraphQL, Vue. CS degree preferred."),
    ("Data Analyst",              "Data Analyst {y}+ years. Must have: SQL, Excel, Power BI, Tableau, Python, Statistics, Data Visualization, Pandas. Nice to have: R, Machine Learning, ETL, Looker. B.Sc or MBA."),
    ("Machine Learning Engineer", "ML Engineer {y}+ years. Required: Python, TensorFlow, PyTorch, Machine Learning, Deep Learning, Docker, AWS, Git, SQL, Scikit-learn. Nice to have: Kubernetes, MLflow, Spark, Airflow, Kubeflow."),
    ("Java Developer",            "Java Developer {y}+ years. Must have: Java, Spring Boot, REST API, SQL, Git, Docker, Maven, Microservices. Nice to have: Kubernetes, AWS, Kafka, Redis. B.Tech CSE required."),
    ("Python Developer",          "Python Developer {y}+ years. Required: Python, Django or Flask, REST API, SQL, PostgreSQL, Git, Docker, Linux. Nice to have: AWS, Redis, GraphQL, Celery, FastAPI. B.Tech IT or CSE."),
    ("DevOps Engineer",           "DevOps Engineer {y}+ years. Required: Docker, Kubernetes, Jenkins, AWS, Git, Linux, CI/CD, Terraform, Bash. Nice to have: Azure, GCP, Ansible, Prometheus, ELK. B.Tech preferred."),
    ("Android Developer",         "Android Developer {y}+ years. Required: Kotlin, Java, Android SDK, REST API, Git, Firebase, MVVM. Nice to have: Jetpack Compose, Coroutines, Dagger, Room. B.Tech CSE."),
    ("SQL Developer",             "SQL Developer {y}+ years. Must have: SQL, MySQL or PostgreSQL, Stored Procedures, ETL, Indexing, Power BI. Nice to have: Python, Oracle, Data Warehousing, SSIS. B.Sc CS or MCA."),
    ("Testing",                   "QA Engineer {y}+ years. Required: Selenium, Python or Java, Manual Testing, JIRA, Git, Agile, API Testing, Postman. Nice to have: Performance Testing, JMeter, BDD, Cypress. B.Tech."),
]

# Generate resumes
resume_rows = []
for cat, templates in RESUME_DATA.items():
    for _ in range(60):
        tmpl = random.choice(templates)
        y    = random.randint(1, 8)
        g    = random.randint(2015, 2023)
        resume_rows.append({
            "Category": cat,
            "Resume"  : tmpl.format(y=y, g=g)
        })

df_resumes = pd.DataFrame(resume_rows).sample(frac=1, random_state=42).reset_index(drop=True)

# Generate JDs
jd_rows = []
companies  = ["TechCorp","DataInc","WebSols","AI Labs","InnovateTech","CodeWorks","CloudBase","DevHouse","NexGen","ByteForge"]
locations  = ["Bangalore","Hyderabad","Chennai","Pune","Mumbai","Delhi","Noida","Remote"]
for cat, tmpl in JD_DATA:
    for _ in range(50):
        y = random.randint(1, 5)
        jd_rows.append({
            "job_title"       : cat,
            "job_description" : tmpl.format(y=y),
            "company"         : random.choice(companies),
            "location"        : random.choice(locations),
        })

df_jd = pd.DataFrame(jd_rows).sample(frac=1, random_state=42).reset_index(drop=True)

# Save raw datasets
df_resumes.to_csv("data/raw/resumes.csv",          index=False)
df_jd.to_csv(     "data/raw/job_descriptions.csv", index=False)
print(f"   ✅ resumes.csv          — {len(df_resumes)} rows, {df_resumes['Category'].nunique()} categories")
print(f"   ✅ job_descriptions.csv — {len(df_jd)} rows")


# ── Step 2: Clean & Process Data ───────────────────────────
print("\n🧹 Step 3/4 — Cleaning and processing data...")

sys.path.insert(0, ".")
from src.text_cleaner import basic_clean, advanced_clean

print("   Applying text cleaning (60 seconds)...")
df_resumes["resume_clean_basic"]    = df_resumes["Resume"].apply(basic_clean)
df_resumes["resume_clean_advanced"] = df_resumes["Resume"].apply(advanced_clean)
df_resumes["resume_word_count"]     = df_resumes["resume_clean_basic"].apply(
    lambda x: len(str(x).split())
)
# Rename columns to lowercase for consistency
df_resumes = df_resumes.rename(columns={"Category": "category", "Resume": "resume"})
df_resumes.to_csv("data/processed/cleaned_resumes.csv", index=False)
print(f"   ✅ cleaned_resumes.csv saved — {len(df_resumes)} rows")

# Clean JDs too
df_jd["jd_clean"] = df_jd["job_description"].apply(basic_clean)
df_jd.to_csv("data/processed/cleaned_jd.csv", index=False)
print(f"   ✅ cleaned_jd.csv saved — {len(df_jd)} rows")


# ── Step 3: Train Model ────────────────────────────────────
print("\n🤖 Step 4/4 — Training the model...")

from src.model_training import load_and_prepare_data, train_and_evaluate, save_model

X_text, y, le, df = load_and_prepare_data("data/processed/cleaned_resumes.csv")

result = train_and_evaluate(
    X_text, y, le,
    model_type = "logistic",
    test_size  = 0.20,
)

metadata = {
    "model_type"      : result["model_type"],
    "test_accuracy"   : f"{result['test_accuracy']*100:.2f}%",
    "f1_weighted"     : f"{result['f1_weighted']*100:.2f}%",
    "cv_mean"         : f"{result['cv_mean']*100:.2f}%",
    "n_classes"       : len(le.classes_),
    "classes"         : list(le.classes_),
    "n_train_samples" : len(result["X_train"]),
    "n_test_samples"  : len(result["X_test"]),
    "train_time"      : result["train_time"],
}

save_model(result["pipeline"], le, metadata)


# ── Final Summary ──────────────────────────────────────────
print("\n" + "=" * 55)
print("  ✅ AUTO SETUP COMPLETE!")
print("=" * 55)
print(f"""
  📊 Dataset      : {len(df_resumes)} resumes, {df_resumes['category'].nunique()} categories
  🤖 Model        : Logistic Regression + TF-IDF
  📈 Accuracy     : {result['test_accuracy']*100:.2f}%
  🎯 F1 Score     : {result['f1_weighted']*100:.2f}%
  📉 CV Mean      : {result['cv_mean']*100:.2f}%

  ✅ data/raw/resumes.csv
  ✅ data/raw/job_descriptions.csv
  ✅ data/processed/cleaned_resumes.csv
  ✅ model/full_pipeline.pkl
  ✅ model/label_encoder.pkl
  ✅ model/model_metadata.pkl

  🚀 NOW RUN:
     python app.py

  🌐 Then open:
     http://127.0.0.1:5000
""")
print("=" * 55)
