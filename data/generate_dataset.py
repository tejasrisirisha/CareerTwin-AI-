"""
generate_dataset.py
====================
Creates a synthetic-but-realistic Student Employability & Placement dataset.

Why synthetic: no single public dataset covers every feature this platform
needs (CGPA, internships, projects, certifications, aptitude, coding/comm
skills, backlogs, LinkedIn/GitHub activity, branch, training, etc.) with a
placement outcome and package tied together in a coherent way. This script
builds one from realistic Indian-engineering-college distributions, with the
outcome generated from a transparent weighted formula + noise, so the
resulting ML models learn genuine, explainable relationships.

Run:  python generate_dataset.py
Output: student_employability_dataset.csv  (this folder)
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2500

BRANCHES = ["CSE", "CSM (AI/ML)", "ECE", "EEE", "MECH", "CIVIL", "IT"]
GENDERS = ["Male", "Female"]

# ---- base numeric features -------------------------------------------------
CGPA = np.clip(np.random.normal(7.4, 1.0, N), 4.5, 10.0)
Internships = np.clip(np.random.poisson(1.1, N), 0, 4)
Projects = np.clip(np.random.poisson(2.3, N), 0, 6)
Certifications = np.clip(np.random.poisson(1.5, N), 0, 6)
Aptitude_Score = np.clip(np.random.normal(62, 16, N), 10, 100)
Technical_Skill = np.clip(np.random.normal(6.2, 1.6, N), 1, 10)
Coding_Skill = np.clip(np.random.normal(5.8, 1.9, N), 1, 10)
Communication_Skill = np.clip(np.random.normal(6.4, 1.5, N), 1, 10)
Soft_Skill = np.clip(np.random.normal(6.3, 1.4, N), 1, 10)
Backlogs = np.clip(np.random.poisson(0.6, N), 0, 6)
Extracurricular_Score = np.clip(np.random.normal(5.5, 2.0, N), 0, 10)
Leadership_Score = np.clip(np.random.normal(5.0, 2.1, N), 0, 10)
LinkedIn_GitHub_Activity = np.clip(np.random.normal(5.2, 2.3, N), 0, 10)
Placement_Training = np.random.choice(["Yes", "No"], N, p=[0.62, 0.38])
Gender = np.random.choice(GENDERS, N, p=[0.58, 0.42])
Branch = np.random.choice(BRANCHES, N, p=[0.28, 0.18, 0.16, 0.10, 0.12, 0.08, 0.08])

df = pd.DataFrame({
    "CGPA": CGPA.round(2),
    "Internships": Internships,
    "Projects": Projects,
    "Certifications": Certifications,
    "Aptitude_Score": Aptitude_Score.round(1),
    "Technical_Skill": Technical_Skill.round(1),
    "Coding_Skill": Coding_Skill.round(1),
    "Communication_Skill": Communication_Skill.round(1),
    "Soft_Skill": Soft_Skill.round(1),
    "Backlogs": Backlogs,
    "Extracurricular_Score": Extracurricular_Score.round(1),
    "Leadership_Score": Leadership_Score.round(1),
    "LinkedIn_GitHub_Activity": LinkedIn_GitHub_Activity.round(1),
    "Placement_Training": Placement_Training,
    "Gender": Gender,
    "Branch": Branch,
})

# ---- transparent latent "employability" formula (drives the label) --------
training_bonus = (df["Placement_Training"] == "Yes").astype(int) * 0.35
latent = (
    (df["CGPA"] - 5) * 0.55 +
    df["Internships"] * 0.55 +
    df["Projects"] * 0.30 +
    df["Certifications"] * 0.18 +
    (df["Aptitude_Score"] - 50) * 0.028 +
    (df["Technical_Skill"] - 5) * 0.42 +
    (df["Coding_Skill"] - 5) * 0.48 +
    (df["Communication_Skill"] - 5) * 0.22 +
    (df["Soft_Skill"] - 5) * 0.10 +
    (df["LinkedIn_GitHub_Activity"] - 5) * 0.08 +
    training_bonus -
    df["Backlogs"] * 0.65
)
noise = np.random.normal(0, 0.9, N)
score = latent + noise
prob = 1 / (1 + np.exp(-(score - score.mean()) / (score.std() + 1e-6) * 1.7))
Placed = (np.random.rand(N) < prob).astype(int)

# ---- package (LPA), only meaningful for / correlated with stronger profiles
base_package = 3.5 + (df["CGPA"] - 6) * 0.9 + df["Internships"] * 0.8 + \
    df["Projects"] * 0.35 + (df["Technical_Skill"] - 5) * 0.9 + \
    (df["Coding_Skill"] - 5) * 1.1 + (df["Certifications"]) * 0.25 - \
    df["Backlogs"] * 0.5
base_package = np.clip(base_package + np.random.normal(0, 1.1, N), 2.5, 45)
Package_LPA = np.where(Placed == 1, base_package, 0.0).round(2)

df["Placed"] = Placed
df["Package_LPA"] = Package_LPA

out_path = "student_employability_dataset.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["Placed"].value_counts(normalize=True).round(3))
