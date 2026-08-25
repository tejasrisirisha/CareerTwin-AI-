"""
mentor.py
==========
"AI Career Chatbot" engine.

This is a template/rule-based mentor that grounds every answer in the
student's OWN computed data (probability, gaps, role match, weak areas) —
so answers are personalized, not generic canned text. It's intentionally
built with no external API dependency so it works fully offline / without
any API key.

To upgrade it to a true LLM-powered chatbot later: keep `build_context()`
exactly as is (it's already the ideal prompt payload) and swap
`answer_question()`'s body for a call to your LLM of choice, passing
`build_context(...)` as the system/context message. Nothing else in the
app needs to change.
"""

import re


def build_context(profile, probability, emp_score, package, role_matches, gaps):
    """Assemble a compact textual context block describing this student —
    this is exactly what you'd hand to a real LLM as system context."""
    top_role = role_matches[0] if role_matches else None
    lines = [
        f"CGPA: {profile['CGPA']}, Backlogs: {profile['Backlogs']}, "
        f"Internships: {profile['Internships']}, Projects: {profile['Projects']}, "
        f"Certifications: {profile['Certifications']}",
        f"Aptitude: {profile['Aptitude_Score']}, Technical: {profile['Technical_Skill']}, "
        f"Coding: {profile['Coding_Skill']}, Communication: {profile['Communication_Skill']}",
        f"Placement probability: {probability*100:.1f}%, Employability score: {emp_score:.1f}/100, "
        f"Predicted package: {package:.2f} LPA",
    ]
    if top_role:
        lines.append(f"Best-fit role: {top_role['role']} ({top_role['match']}% match)")
    if gaps:
        lines.append("Top skill gaps: " + ", ".join(g["skill"] for g in gaps[:3]))
    return "\n".join(lines)


_INTENTS = [
    (r"amazon|google|product company|faang", "target_company"),
    (r"package|salary|ctc|lpa", "package"),
    (r"chance|probability|placed|likely", "probability"),
    (r"gap|weak|improve|lacking|missing", "gaps"),
    (r"role|job|which company|which role|fit", "role"),
    (r"resume|cv", "resume"),
    (r"backlog", "backlog"),
    (r"aptitude", "aptitude"),
    (r"project", "project"),
    (r"intern", "internship"),
    (r"cgpa|marks|grade", "cgpa"),
]


def _classify(question: str) -> str:
    q = question.lower()
    for pattern, intent in _INTENTS:
        if re.search(pattern, q):
            return intent
    return "general"


def answer_question(question, profile, probability, emp_score, package, role_matches, gaps, plan):
    """Rule-based, data-grounded answer generator. Returns a markdown string."""
    intent = _classify(question)
    top_role = role_matches[0]["role"] if role_matches else "a matching role"
    top_gap = gaps[0] if gaps else None

    if intent == "target_company":
        return (
            f"For product companies like Amazon/Google, your **Coding Skill ({profile['Coding_Skill']}/10)** "
            f"and **Aptitude ({profile['Aptitude_Score']}/100)** matter most — those companies weight DSA and "
            f"problem-solving heavily. Right now your best-fit role is **{top_role}**. "
            + (f"To close the biggest gap, focus on **{top_gap['skill']}** — you're {top_gap['gap']} points "
               f"below what's typically expected." if top_gap else
               "You're in solid shape across the board — keep practicing timed DSA rounds.")
        )
    if intent == "package":
        return (
            f"Based on your current profile, the model estimates a placement package around "
            f"**₹{package:.2f} LPA**. Package predictions rise fastest with Coding Skill, Internships, and "
            f"Projects — each additional strong project or internship tends to move this up."
        )
    if intent == "probability":
        return (
            f"Your current placement probability is **{probability*100:.1f}%**, with an employability score "
            f"of **{emp_score:.1f}/100**. "
            + ("You're in a strong position — keep it up with mock interviews." if probability >= 0.7 else
               "There's real room to move this up — check the Personalized Roadmap tab for your top 3 priorities.")
        )
    if intent == "gaps":
        if not gaps:
            return "You don't have major gaps against your best-fit role right now — nice work. Keep sharpening interview skills."
        gap_list = "\n".join(f"- **{g['skill']}**: {g['current']} vs {g['required']} required (gap of {g['gap']})" for g in gaps[:4])
        return f"Here are your top skill gaps against **{top_role}**:\n\n{gap_list}"
    if intent == "role":
        lines = "\n".join(f"{i+1}. **{r['role']}** — {r['match']}% match" for i, r in enumerate(role_matches[:3]))
        return f"Based on your profile, your best-fit roles are:\n\n{lines}"
    if intent == "resume":
        return (
            "Head to the **Resume Analyzer** tab and upload your resume (PDF or TXT) — I'll scan it for "
            "keyword coverage across programming languages, web/data skills, and CS fundamentals, and tell you "
            "exactly what's missing."
        )
    if intent == "backlog":
        if profile["Backlogs"] > 0:
            return (
                f"You currently have **{profile['Backlogs']} backlog(s)**. Backlogs are one of the heaviest "
                "negative factors in placement prediction — clearing them should be your #1 priority before "
                "any other prep."
            )
        return "You have no backlogs — that's a genuine strength, keep your academics clean."
    if intent == "aptitude":
        return (
            f"Your aptitude score is **{profile['Aptitude_Score']}/100**. "
            + ("This is solid — most core company cutoffs sit around 60-65." if profile["Aptitude_Score"] >= 65
               else "Most core/service company screening rounds cut off around 60-65 — 30 minutes/day on quant + logical reasoning would move this meaningfully.")
        )
    if intent == "project":
        return (
            f"You currently have **{profile['Projects']} project(s)**. Aim for at least 3, including one "
            "end-to-end project (with a live demo/GitHub repo) — recruiters weight demonstrated projects heavily "
            "over course-only knowledge."
        )
    if intent == "internship":
        return (
            f"You have **{profile['Internships']} internship(s)** logged. Even a virtual 6-8 week internship "
            "(Internshala/AICTE) meaningfully improves placement probability, and it's the single fastest lever "
            "if you have zero right now."
        )
    if intent == "cgpa":
        return (
            f"Your CGPA is **{profile['CGPA']}**. "
            + ("This comfortably clears most eligibility cutoffs (typically 6.5-7.5)." if profile["CGPA"] >= 7.0
               else "Many product/service companies set eligibility cutoffs around 6.5-7.5 — worth prioritizing your next two semesters.")
        )

    # general fallback — short personalized summary
    top1 = plan[0] if plan else None
    return (
        f"You're at **{probability*100:.1f}%** placement probability with an employability score of "
        f"**{emp_score:.1f}/100**, best matched to **{top_role}**. "
        + (f"Your top priority right now: **{top1[1]}** — {top1[2]}" if top1 else "")
        + "\n\nAsk me about your package estimate, skill gaps, best-fit role, resume, or specific factors "
          "like CGPA, backlogs, aptitude, projects, or internships."
    )
