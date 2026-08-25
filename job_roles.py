"""
job_roles.py
=============
Static domain knowledge used by the app: target profiles for common job
roles (used for role-matching + skill-gap radar) and a resume keyword bank
(used for resume analysis).
"""

JOB_ROLES = {
    "Software Development Engineer (SDE)": {
        "icon": "💻",
        "description": "Product-company coding roles (Amazon, Google, Microsoft, Flipkart).",
        "core_skills": ["DSA", "System Design", "Coding", "CS Fundamentals"],
        "weights": {
            "CGPA": 7.5, "Coding_Skill": 9, "Technical_Skill": 8.5,
            "Aptitude_Score": 75, "Projects": 4, "Backlogs_Inverse": 9,
        },
    },
    "Data Analyst / ML Engineer": {
        "icon": "📊",
        "description": "Analytics, data science, and applied ML roles.",
        "core_skills": ["Python", "Statistics", "ML", "SQL"],
        "weights": {
            "CGPA": 7.0, "Technical_Skill": 8.5, "Coding_Skill": 7.5,
            "Certifications": 3, "Projects": 4, "Aptitude_Score": 65,
        },
    },
    "IT Services / Consulting": {
        "icon": "🏢",
        "description": "TCS, Infosys, Wipro, Accenture style service roles.",
        "core_skills": ["Aptitude", "Communication", "Core CS", "Adaptability"],
        "weights": {
            "CGPA": 6.5, "Aptitude_Score": 60, "Communication_Skill": 7,
            "Soft_Skill": 6.5, "Backlogs_Inverse": 7,
        },
    },
    "Product Manager (Associate)": {
        "icon": "🧭",
        "description": "APM / associate product roles blending tech + business.",
        "core_skills": ["Communication", "Leadership", "Technical Awareness"],
        "weights": {
            "Communication_Skill": 8.5, "Leadership_Score": 7.5,
            "Technical_Skill": 6, "Extracurricular_Score": 6, "CGPA": 7,
        },
    },
    "Core / Research (Higher Studies track)": {
        "icon": "🔬",
        "description": "Strong academics track for MS/research or core engineering roles.",
        "core_skills": ["Academics", "Projects", "Certifications"],
        "weights": {
            "CGPA": 8.5, "Projects": 5, "Certifications": 4,
            "Technical_Skill": 7.5, "Backlogs_Inverse": 9.5,
        },
    },
}

RESUME_SKILL_KEYWORDS = {
    "Programming Languages": ["python", "java", "c++", "javascript", "sql", "c ", "r ", "go", "typescript"],
    "Web & App Dev": ["react", "node", "django", "flask", "html", "css", "rest api", "next.js", "angular"],
    "Data / ML": ["machine learning", "deep learning", "pandas", "numpy", "tensorflow", "pytorch",
                  "scikit-learn", "nlp", "data analysis", "power bi", "tableau"],
    "Cloud / DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins", "git", "github"],
    "CS Fundamentals": ["data structures", "algorithms", "dbms", "operating systems", "computer networks",
                         "oop", "system design"],
    "Soft / Extra": ["leadership", "internship", "hackathon", "certification", "team", "communication",
                      "project management"],
}
