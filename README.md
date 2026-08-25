# AI Career Intelligence Platform

An end-to-end AI Career Intelligence system for the "AI-Based Student
Employability & Placement Prediction System" problem statement — upgraded
from a basic predictor into a full 10-feature platform.

## What's included

| File | Purpose |
|---|---|
| `data/generate_dataset.py` | Generates the training dataset from realistic distributions |
| `data/student_employability_dataset.csv` | The generated dataset (2,500 students) — already included |
| `train_model.py` | Trains the classifier + regressor and saves all model artifacts |
| `models/` | Trained model files (already included — no need to retrain) |
| `job_roles.py` | Job-role target profiles + resume keyword bank |
| `mentor.py` | Rule-based "AI Career Chatbot" engine, grounded in the student's own data |
| `app.py` | The Streamlit application (all 10 features) |
| `requirements.txt` | Python dependencies |

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

That's it — the dataset and trained models are already included, so the
app runs immediately. Your project will open at `http://localhost:8501`.

## Regenerating from scratch (optional)

If you want to regenerate the dataset or retrain the models yourself
(useful if you want to show the "training pipeline" during your demo):

```bash
python data/generate_dataset.py   # rebuilds the CSV
python train_model.py             # retrains + re-saves all artifacts
```

Current model performance (on held-out test data):
- Placement classifier: ~70% accuracy, ~0.77 ROC-AUC
- Package regressor: ~1.1 LPA mean absolute error

## About the dataset

No single public dataset covers every feature this platform needs
(CGPA, internships, projects, certifications, aptitude, coding/communication
skills, backlogs, LinkedIn/GitHub activity, branch, training status) together
with a placement outcome and package. `generate_dataset.py` builds one from
realistic Indian-engineering-college distributions, with the placement label
generated from a transparent, documented weighted formula plus noise — so
the trained models learn genuine, explainable relationships you can describe
confidently to judges (see the formula directly in the script).

## The 10 features (matches your upgrade plan)

1. **Dashboard** — headline score, probability, package, best-fit role
2. **Explainable AI** — feature-impact chart (SHAP if installed, otherwise a documented heuristic) + model metrics
3. **Skill Gap Radar** — your profile vs. a chosen role's requirements, visually
4. **Job Role Matcher** — ranked match % across 5 role archetypes
5. **Resume Analyzer** — upload a PDF/TXT resume, get keyword-coverage scoring
6. **Personalized Roadmap** — prioritized (Critical/High/Medium/Low) action plan
7. **Industry Benchmarking** — percentile rank + branch-wise averages vs. the reference cohort
8. **Placement Simulator** — live what-if sliders, before/after comparison
9. **Progress Tracker** — log snapshots over time, see your trend line
10. **Smart Recommendations** — top-3 quick-glance priorities
11. **AI Career Chatbot** — a data-grounded Q&A mentor (rule-based, offline; swap in a real LLM API later — see the docstring in `mentor.py`)

## Notes for your demo

- The chatbot is intentionally rule-based/offline so it works with zero API
  keys and zero internet dependency during your presentation — every answer
  is generated from the student's actual computed numbers, not canned text.
- If you install `shap` (see `requirements.txt`), the Explainable AI tab
  automatically switches from the heuristic to exact SHAP attribution — no
  code changes needed.
