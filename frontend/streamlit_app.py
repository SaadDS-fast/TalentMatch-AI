"""
Streamlit frontend for TalentMatch AI.

Supports:
1. Single resume analysis
2. Multiple candidate ranking
"""

import html
import importlib.util
import json
import re
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


from app.config import (  # noqa: E402
    DATA_DIR,
    EDUCATION_MATCH_WEIGHT,
    EXPERIENCE_MATCH_WEIGHT,
    SEMANTIC_SIMILARITY_WEIGHT,
    SKILL_MATCH_WEIGHT,
)
from app.services.ats_scorer import ATSScorer  # noqa: E402
from app.services.candidate_ranker import CandidateRanker  # noqa: E402
from app.services.jd_parser import JobDescriptionParser  # noqa: E402
from app.services.llm_feedback import LLMFeedbackGenerator  # noqa: E402
from app.services.matching_engine import MatchingEngine  # noqa: E402
from app.services.resume_parser import ResumeParser  # noqa: E402
from app.services.scoring import ScoringEngine  # noqa: E402
from app.services.semantic_similarity import SemanticSimilarityEngine  # noqa: E402
from app.services.skill_extractor import SkillExtractor  # noqa: E402
from app.utils.file_loader import load_text_from_file  # noqa: E402


st.set_page_config(
    page_title="TalentMatch AI",
    page_icon="🎯",
    layout="wide",
)


PALETTE = {
    "green": "#1b7f5f",
    "blue": "#1f6feb",
    "amber": "#b7791f",
    "red": "#c93c37",
    "ink": "#202636",
    "muted": "#697386",
    "border": "#dbe3ee",
    "surface": "#f8fafc",
}

SAMPLE_RESUME_PATH = DATA_DIR / "sample_resumes" / "sample_resume.txt"
SAMPLE_JOB_PATH = DATA_DIR / "sample_job_descriptions" / "sample_ai_engineer_jd.txt"

SCORE_TOOLTIPS = {
    "Overall Match": "Weighted fit score combining skills, semantic relevance, experience, and education.",
    "Skill Match": "Measures overlap between required job skills and detected resume skills.",
    "Semantic Match": "Measures contextual similarity using transformer embeddings when cached, with a safe local fallback.",
    "Experience": "Estimates whether resume experience evidence aligns with job requirements.",
    "ATS Score": "Measures resume compatibility with Applicant Tracking Systems using sections, keywords, and evidence checks.",
}


@st.cache_data
def load_skill_taxonomy() -> dict[str, list[str]]:
    skills_file = DATA_DIR / "skills_database" / "skills.json"
    with open(skills_file, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def get_resume_parser() -> ResumeParser:
    return ResumeParser()


@st.cache_resource
def get_job_parser() -> JobDescriptionParser:
    return JobDescriptionParser()


@st.cache_resource
def get_skill_extractor() -> SkillExtractor:
    return SkillExtractor()


@st.cache_resource
def get_semantic_engine() -> SemanticSimilarityEngine:
    return SemanticSimilarityEngine()


@st.cache_resource
def get_scoring_engine() -> ScoringEngine:
    return ScoringEngine()


@st.cache_resource
def get_feedback_generator() -> LLMFeedbackGenerator:
    return LLMFeedbackGenerator()


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name


@st.cache_data(show_spinner=False)
def analyze_resume(resume_path: str, job_path: str) -> dict:
    resume_raw_text = load_text_from_file(resume_path)
    job_raw_text = load_text_from_file(job_path)

    resume = get_resume_parser().parse(resume_path)
    job = get_job_parser().parse(job_path)

    extractor = get_skill_extractor()

    resume.skills = extractor.extract_skills(resume_raw_text)
    job.required_skills = extractor.extract_skills(job_raw_text)

    matching = MatchingEngine().match(resume, job)

    semantic_score = get_semantic_engine().calculate_similarity(
        resume_raw_text,
        job_raw_text,
        resume_data=resume.model_dump(),
        job_data=job.model_dump(),
    )

    scoring_engine = get_scoring_engine()

    experience_score = scoring_engine.estimate_experience_score(
        resume.experience,
        job.experience_required,
    )

    education_score = scoring_engine.estimate_education_score(
        resume.education,
        job.qualifications,
    )

    scores = scoring_engine.calculate_final_score(
        skill_score=matching["skill_match_score"],
        semantic_score=semantic_score,
        experience_score=experience_score,
        education_score=education_score,
    )

    ats_result = ATSScorer().calculate_score(
        resume_text=resume_raw_text,
        matched_skills=matching["matched_skills"],
        missing_skills=matching["missing_skills"],
    )

    feedback = get_feedback_generator().generate_feedback(
        overall_score=scores["overall_score"],
        matched_skills=matching["matched_skills"],
        missing_skills=matching["missing_skills"],
        skill_score=scores["skill_score"],
        ats_score=ats_result["ats_score"],
        experience_score=scores["experience_score"],
    )

    return {
        "resume": resume.model_dump(),
        "job": job.model_dump(),
        "scores": scores,
        "matching": matching,
        "ats": ats_result,
        "feedback": feedback,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "resume_text_stats": {
            "word_count": len(resume_raw_text.split()),
            "character_count": len(resume_raw_text),
        },
    }


def score_contributions(scores: dict) -> pd.DataFrame:
    rows = [
        {
            "Component": "Skill Match",
            "Score": scores["skill_score"],
            "Weight": SKILL_MATCH_WEIGHT,
            "Contribution": round(scores["skill_score"] * SKILL_MATCH_WEIGHT, 2),
        },
        {
            "Component": "Semantic Similarity",
            "Score": scores["semantic_score"],
            "Weight": SEMANTIC_SIMILARITY_WEIGHT,
            "Contribution": round(scores["semantic_score"] * SEMANTIC_SIMILARITY_WEIGHT, 2),
        },
        {
            "Component": "Experience",
            "Score": scores["experience_score"],
            "Weight": EXPERIENCE_MATCH_WEIGHT,
            "Contribution": round(scores["experience_score"] * EXPERIENCE_MATCH_WEIGHT, 2),
        },
        {
            "Component": "Education",
            "Score": scores["education_score"],
            "Weight": EDUCATION_MATCH_WEIGHT,
            "Contribution": round(scores["education_score"] * EDUCATION_MATCH_WEIGHT, 2),
        },
    ]
    return pd.DataFrame(rows)


def grouped_skills(skills: list[str]) -> dict[str, list[str]]:
    taxonomy = load_skill_taxonomy()
    lookup = {
        skill.lower(): format_category(category)
        for category, category_skills in taxonomy.items()
        for skill in category_skills
    }

    grouped: dict[str, list[str]] = {}
    for skill in sorted(set(skills)):
        category = lookup.get(skill.lower(), "Other")
        grouped.setdefault(category, []).append(skill)

    return grouped


def format_category(category: str) -> str:
    labels = {
        "programming_languages": "Programming",
        "data_science": "Data Science",
        "deep_learning_ai": "Deep Learning / AI",
        "databases": "Databases",
        "backend_deployment": "Backend / Deployment",
        "business_intelligence": "Business Intelligence",
        "soft_skills": "Soft Skills",
    }
    return labels.get(category, category.replace("_", " ").title())


def fit_label(score: float) -> str:
    if score >= 80:
        return "Strong Shortlist"
    if score >= 70:
        return "Shortlist"
    if score >= 60:
        return "Conditional Shortlist"
    if score >= 45:
        return "Junior-Level Fit"
    return "Needs More Evidence"


def score_band(score: float) -> tuple[str, str]:
    if score >= 85:
        return "Excellent", PALETTE["green"]
    if score >= 70:
        return "Good", PALETTE["blue"]
    if score >= 50:
        return "Average", PALETTE["amber"]
    return "Weak", PALETTE["red"]


def status_color(status: str) -> str:
    return {
        "Passed": PALETTE["green"],
        "Warning": PALETTE["amber"],
        "Missing": PALETTE["red"],
    }.get(status, "#4b5563")


def render_metric_card(label: str, value: str, caption: str, score: float | None = None) -> None:
    band, color = score_band(score or 0)
    tooltip = SCORE_TOOLTIPS.get(label, "")
    state_html = (
        f'<span class="score-state" style="background:{color};">{band}</span>'
        if score is not None
        else ""
    )

    st.markdown(
        f"""
        <div class="metric-card" title="{html.escape(tooltip)}">
            <div class="metric-label">{html.escape(label)} <span class="info-dot">i</span></div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-caption">{state_html} {html.escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def infer_candidate_level(result: dict) -> str:
    resume = result["resume"]
    experience_text = " ".join(resume.get("experience", [])).lower()
    project_text = " ".join(resume.get("projects", [])).lower()
    education_text = " ".join(resume.get("education", [])).lower()
    full_text = f"{experience_text} {project_text} {education_text}"

    senior_titles = ["senior", "lead", "manager", "architect", "principal", "head of"]
    full_time_terms = ["full-time", "full time", "software engineer", "ml engineer", "ai engineer", "data scientist"]
    teaching_only = "teacher" in experience_text and not any(term in experience_text for term in full_time_terms)
    academic_only = bool(project_text) and not any(term in experience_text for term in full_time_terms)

    years = [
        int(match)
        for match in re.findall(r"\b([1-9]\d?)\+?\s*(?:years|year|yrs|yr)\b", full_text)
    ]

    if any(title in experience_text for title in senior_titles) or any(year >= 5 for year in years):
        return "Senior"

    if any(year >= 3 for year in years) and not teaching_only:
        return "Intermediate"

    if any(term in full_text for term in ["intern", "assistant", "student", "fresh graduate", "ms ", "bs "]):
        return "Entry"

    if teaching_only or academic_only:
        return "Entry / Junior"

    if experience_text:
        return "Junior"

    return "Entry"


def infer_industries(skills: list[str]) -> list[str]:
    lowered = {skill.lower() for skill in skills}
    industries = []

    if lowered.intersection({"machine learning", "pytorch", "tensorflow", "transformers", "llm", "generative ai"}):
        industries.append("AI")
    if lowered.intersection({"pandas", "numpy", "scikit-learn", "statistics", "data visualization"}):
        industries.append("Data Science")
    if lowered.intersection({"fastapi", "django", "flask", "rest api", "docker"}):
        industries.append("Backend")
    if lowered.intersection({"natural language processing", "nlp", "computer vision", "embeddings"}):
        industries.append("Research")

    return industries or ["General Technical"]


def recommend_roles(skills: list[str]) -> list[str]:
    lowered = {skill.lower() for skill in skills}
    roles = []

    if lowered.intersection({"machine learning", "pytorch", "tensorflow", "scikit-learn"}):
        roles.extend(["AI Engineer", "ML Engineer"])
    if lowered.intersection({"natural language processing", "nlp", "transformers", "llm"}):
        roles.append("NLP Engineer")
    if "fastapi" in lowered or "python" in lowered:
        roles.append("Python Developer")
    if lowered.intersection({"pandas", "numpy", "sql", "data visualization"}):
        roles.append("Data Scientist")
    if lowered.intersection({"docker", "aws", "azure", "gcp"}):
        roles.append("ML Platform Intern")

    unique_roles = []
    for role in roles:
        if role not in unique_roles:
            unique_roles.append(role)

    return unique_roles[:5] or ["Technical Intern", "Junior Analyst", "Python Developer"]


def build_recruiter_insights(result: dict) -> dict:
    skills = result["resume"].get("skills", [])
    return {
        "candidate_level": infer_candidate_level(result),
        "industries": infer_industries(skills),
        "best_fit_roles": recommend_roles(skills),
    }


def structured_recommendations(result: dict) -> dict[str, list[str]]:
    matched = result["matching"]["matched_skills"]
    missing = result["matching"]["missing_skills"]
    insights = build_recruiter_insights(result)

    candidate_items = []
    recruiter_items = []

    for skill in missing[:4]:
        if skill.lower() in {"docker", "fastapi", "transformers", "aws", "azure", "gcp"}:
            candidate_items.append(f"Build and document one project that demonstrates {skill}.")
        else:
            candidate_items.append(f"Add stronger resume evidence for {skill}.")

    if "Docker" not in matched and "Docker" not in missing:
        candidate_items.append("Show deployment readiness with a small Dockerized project.")
    if "FastAPI" not in matched and "FastAPI" not in missing:
        candidate_items.append("Add an API-serving project using FastAPI or a similar backend framework.")
    if not candidate_items:
        candidate_items.append("Keep matched skills visible with measurable project outcomes.")

    if result["scores"]["overall_score"] < 60:
        recruiter_items.append("Candidate may be suitable for internship or junior screening if learning velocity is important.")
    else:
        recruiter_items.append("Candidate has enough evidence for a focused recruiter screen.")

    if missing:
        recruiter_items.append("Validate gaps before shortlisting: " + ", ".join(missing[:4]) + ".")
    if matched:
        recruiter_items.append("Use matched skills for interview probes: " + ", ".join(matched[:4]) + ".")

    recruiter_items.append(
        f"Best initial role fit: {', '.join(insights['best_fit_roles'][:3])}."
    )

    return {
        "candidate": candidate_items[:5],
        "recruiter": recruiter_items[:5],
    }


def run_analysis_with_progress(resume_path: str, job_path: str) -> dict:
    progress = st.progress(0)
    status = st.empty()
    steps = [
        "Step 1: Parsing Resume...",
        "Step 2: Extracting Skills...",
        "Step 3: Running Semantic Analysis...",
        "Step 4: Calculating ATS...",
        "Step 5: Generating Recruiter Report...",
    ]

    for index, step in enumerate(steps, start=1):
        status.info(step)
        progress.progress(index / (len(steps) + 1))

    result = analyze_resume(resume_path, job_path)
    progress.empty()
    status.empty()
    return result


def render_skill_groups(title: str, skills: list[str], empty_text: str) -> None:
    st.markdown(f"#### {title}")

    if not skills:
        st.info(empty_text)
        return

    for category, category_skills in grouped_skills(skills).items():
        st.markdown(f"**{category}**")
        st.markdown(
            " ".join(
                f'<span class="skill-pill">{html.escape(skill)}</span>'
                for skill in category_skills
            ),
            unsafe_allow_html=True,
        )


def score_breakdown_chart(scores: dict) -> go.Figure:
    df = score_contributions(scores)
    fig = px.bar(
        df,
        x="Component",
        y="Contribution",
        color="Component",
        text="Contribution",
        hover_data={"Score": ":.2f", "Weight": ":.0%"},
        title="Weighted Score Contribution",
    )
    fig.update_layout(showlegend=False, yaxis_title="Points toward overall score")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    return fig


def skill_coverage_chart(matching: dict) -> go.Figure:
    matched_count = len(matching["matched_skills"])
    missing_count = len(matching["missing_skills"])

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Matched", "Missing"],
                values=[matched_count, missing_count],
                hole=0.62,
                marker_colors=["#1b7f5f", "#d95f59"],
            )
        ]
    )
    fig.update_layout(
        title="Skill Coverage",
        annotations=[
            {
                "text": f"{matched_count}/{matched_count + missing_count or 0}",
                "x": 0.5,
                "y": 0.5,
                "font_size": 22,
                "showarrow": False,
            }
        ],
    )
    return fig


def ats_gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f6feb"},
                "steps": [
                    {"range": [0, 50], "color": "#f8d7da"},
                    {"range": [50, 75], "color": "#fff3cd"},
                    {"range": [75, 100], "color": "#d1e7dd"},
                ],
            },
            title={"text": "ATS Compatibility"},
        )
    )
    fig.update_layout(height=260, margin={"l": 24, "r": 24, "t": 48, "b": 16})
    return fig


def radar_chart(scores: dict, ats_score: float) -> go.Figure:
    labels = ["Skill", "Semantic", "Experience", "Education", "ATS"]
    values = [
        scores["skill_score"],
        scores["semantic_score"],
        scores["experience_score"],
        scores["education_score"],
        ats_score,
    ]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                name="Candidate fit",
            )
        ]
    )
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        showlegend=False,
        title="Candidate-Job Fit Radar",
    )
    return fig


def ranking_chart(ranking_df: pd.DataFrame) -> go.Figure:
    if ranking_df.empty:
        return go.Figure()

    fig = px.bar(
        ranking_df.sort_values("Overall Score"),
        x="Overall Score",
        y="Candidate",
        orientation="h",
        color="Overall Score",
        color_continuous_scale=["#d95f59", "#f2c94c", "#1b7f5f"],
        range_x=[0, 100],
        title="Candidate Ranking",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    return fig


def build_json_report(result: dict) -> str:
    report = {
        "generated_at": result.get("generated_at") or datetime.now(timezone.utc).isoformat(),
        "candidate_profile": result["resume"],
        "job_profile": result["job"],
        "scores": result["scores"],
        "ats_score": result["ats"]["ats_score"],
        "matched_skills": result["matching"]["matched_skills"],
        "missing_skills": result["matching"]["missing_skills"],
        "ats_checklist": result["ats"].get("ats_checklist", []),
        "recommendation": result["feedback"].get("structured_feedback", {}),
    }
    return json.dumps(report, indent=2, ensure_ascii=False)


def build_html_report(result: dict) -> str:
    resume = result["resume"]
    scores = result["scores"]
    matching = result["matching"]
    feedback = result["feedback"]["structured_feedback"]
    generated_at = result.get("generated_at") or datetime.now(timezone.utc).isoformat()

    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>TalentMatch AI Recruiter Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; color: #1f2937; line-height: 1.5; }}
          .wrap {{ max-width: 920px; margin: 32px auto; }}
          .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
          .card {{ border: 1px solid #d9dee8; border-radius: 8px; padding: 14px; }}
          .check {{ border-bottom: 1px solid #edf1f7; padding: 8px 0; }}
          .label {{ color: #687385; font-size: 12px; text-transform: uppercase; }}
          .value {{ font-size: 26px; font-weight: 700; }}
          h1, h2 {{ margin-bottom: 8px; }}
          li {{ margin-bottom: 4px; }}
        </style>
      </head>
      <body>
        <main class="wrap">
          <h1>TalentMatch AI Recruiter Report</h1>
          <p><strong>Generated:</strong> {html.escape(generated_at)}</p>
          <p><strong>Candidate:</strong> {html.escape(resume.get("candidate_name") or "Unknown Candidate")}</p>
          <div class="grid">
            <div class="card"><div class="label">Overall</div><div class="value">{scores["overall_score"]}%</div></div>
            <div class="card"><div class="label">Skill</div><div class="value">{scores["skill_score"]}%</div></div>
            <div class="card"><div class="label">Semantic</div><div class="value">{scores["semantic_score"]}%</div></div>
            <div class="card"><div class="label">ATS</div><div class="value">{result["ats"]["ats_score"]}%</div></div>
          </div>
          <h2>Recommendation</h2>
          <p>{html.escape(feedback["candidate_recommendation"])}</p>
          <h2>Matched Skills</h2>
          <p>{html.escape(", ".join(matching["matched_skills"]) or "None detected")}</p>
          <h2>Missing Skills</h2>
          <p>{html.escape(", ".join(matching["missing_skills"]) or "None detected")}</p>
          <h2>ATS Checklist</h2>
          {"".join(f'<div class="check"><strong>{html.escape(item["status"])}</strong> - {html.escape(item["label"])}: {html.escape(item["message"])}</div>' for item in result["ats"].get("ats_checklist", []))}
          <h2>Next Steps</h2>
          <ul>{"".join(f"<li>{html.escape(step)}</li>" for step in feedback["suggested_next_steps"])}</ul>
        </main>
      </body>
    </html>
    """


def is_pdf_export_available() -> bool:
    return importlib.util.find_spec("reportlab") is not None


def build_pdf_report(result: dict) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("reportlab"):
            raise RuntimeError("PDF export requires the optional reportlab package.") from exc
        raise

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    resume = result["resume"]
    scores = result["scores"]
    matching = result["matching"]
    ats = result["ats"]
    feedback = result["feedback"]["structured_feedback"]
    insights = build_recruiter_insights(result)
    recommendations = structured_recommendations(result)
    generated_at = result.get("generated_at") or datetime.now(timezone.utc).isoformat()

    story.append(Paragraph("TalentMatch AI Recruiter Report", styles["Title"]))
    story.append(Paragraph("AI-powered Resume Screening, ATS Analysis and Candidate Ranking", styles["BodyText"]))
    story.append(Paragraph(f"Generated: {generated_at}", styles["BodyText"]))
    story.append(Spacer(1, 0.2 * inch))

    candidate_rows = [
        ["Candidate", resume.get("candidate_name") or "Unknown Candidate"],
        ["Email", resume.get("email") or "Not detected"],
        ["Phone", resume.get("phone") or "Not detected"],
        ["Candidate Level", insights["candidate_level"]],
        ["Industry", ", ".join(insights["industries"])],
    ]
    story.append(_pdf_table(candidate_rows, [1.8 * inch, 4.8 * inch], colors, Table, TableStyle))
    story.append(Spacer(1, 0.18 * inch))

    score_rows = [
        ["Overall Match", f"{scores['overall_score']}%"],
        ["Skill Match", f"{scores['skill_score']}%"],
        ["Semantic Match", f"{scores['semantic_score']}%"],
        ["Experience", f"{scores['experience_score']}%"],
        ["Education", f"{scores['education_score']}%"],
        ["ATS Score", f"{ats['ats_score']}%"],
    ]
    story.append(Paragraph("Scores", styles["Heading2"]))
    story.append(_pdf_table(score_rows, [2.2 * inch, 1.4 * inch], colors, Table, TableStyle))
    story.append(Spacer(1, 0.18 * inch))

    story.append(Paragraph("Matched Skills", styles["Heading2"]))
    story.append(Paragraph(", ".join(matching["matched_skills"]) or "None detected", styles["BodyText"]))
    story.append(Paragraph("Missing Skills", styles["Heading2"]))
    story.append(Paragraph(", ".join(matching["missing_skills"]) or "None detected", styles["BodyText"]))

    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("ATS Checklist", styles["Heading2"]))
    checklist_rows = [["Item", "Status", "Detail"]]
    checklist_rows.extend(
        [
            [item["label"], item["status"], item["message"]]
            for item in ats.get("ats_checklist", [])
        ]
    )
    story.append(
        _pdf_table(
            checklist_rows,
            [2.0 * inch, 1.0 * inch, 3.2 * inch],
            colors,
            Table,
            TableStyle,
            has_header=True,
        )
    )

    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("Recommendation", styles["Heading2"]))
    story.append(Paragraph(feedback["candidate_recommendation"], styles["BodyText"]))
    story.append(Paragraph(f"Interview Recommendation: {feedback['interview_recommendation']}", styles["BodyText"]))

    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("Best Fit Roles", styles["Heading2"]))
    story.append(Paragraph(", ".join(insights["best_fit_roles"]), styles["BodyText"]))

    story.append(Paragraph("Candidate Recommendations", styles["Heading2"]))
    for item in recommendations["candidate"]:
        story.append(Paragraph(f"- {item}", styles["BodyText"]))

    story.append(Paragraph("Recruiter Notes", styles["Heading2"]))
    for item in recommendations["recruiter"]:
        story.append(Paragraph(f"- {item}", styles["BodyText"]))

    document.build(story)
    return buffer.getvalue()


def _pdf_table(
    rows: list[list[str]],
    col_widths: list[float],
    colors,
    table_cls,
    table_style_cls,
    has_header: bool = False,
):
    table = table_cls(rows, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dbe3ee")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]
    if has_header:
        style.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f6feb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(table_style_cls(style))
    return table


def render_profile(result: dict) -> None:
    resume = result["resume"]
    stats = result["resume_text_stats"]

    st.markdown("### Extracted Resume Profile")

    col1, col2, col3 = st.columns(3)
    col1.write(f"**Candidate:** {resume.get('candidate_name') or 'Not detected'}")
    col2.write(f"**Email:** {resume.get('email') or 'Not detected'}")
    col3.write(f"**Phone:** {resume.get('phone') or 'Not detected'}")

    st.caption(
        f"{stats['word_count']} words and {stats['character_count']} characters extracted from the uploaded resume."
    )

    profile_cols = st.columns(3)
    with profile_cols[0]:
        st.markdown("**Education Evidence**")
        render_evidence_list(resume.get("education", []))
    with profile_cols[1]:
        st.markdown("**Experience Evidence**")
        render_evidence_list(resume.get("experience", []))
    with profile_cols[2]:
        st.markdown("**Project Evidence**")
        render_evidence_list(resume.get("projects", []))

    render_skill_groups(
        "Extracted Skills",
        resume.get("skills", []),
        "No known skills were extracted from the resume text.",
    )


def render_evidence_list(items: list[str], limit: int = 4) -> None:
    if not items:
        st.caption("Not detected")
        return

    for item in items[:limit]:
        st.caption(f"- {item}")


def render_ats_checklist(ats_result: dict) -> None:
    st.markdown("### ATS Compatibility Checklist")

    checklist = ats_result.get("ats_checklist", [])
    if not checklist:
        for issue in ats_result["ats_issues"]:
            st.warning(issue)
        return

    checklist_df = pd.DataFrame(
        [
            {
                "Checklist Item": item["label"],
                "Status": item["status"],
                "Detail": item["message"],
            }
            for item in checklist
        ]
    )

    for item in checklist:
        color = status_color(item["status"])
        st.markdown(
            f"""
            <div class="check-row">
                <span class="status-pill" style="background:{color};">{item["status"]}</span>
                <strong>{html.escape(item["label"])}</strong>
                <span>{html.escape(item["message"])}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.download_button(
        "Download ATS checklist CSV",
        checklist_df.to_csv(index=False).encode("utf-8"),
        "ats_checklist.csv",
        "text/csv",
        use_container_width=True,
    )


def render_feedback(feedback: dict) -> None:
    structured = feedback.get("structured_feedback")

    st.markdown("### Recruiter Recommendation")

    if not structured:
        st.write(feedback["recruiter_summary"])
        for recommendation in feedback["recommendations"]:
            st.write(f"- {recommendation}")
        return

    st.info(structured["candidate_recommendation"])

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Key Strengths**")
        for item in structured["key_strengths"]:
            st.success(item)

        st.markdown("**Suggested Next Steps**")
        for item in structured["suggested_next_steps"]:
            st.write(f"- {item}")

    with cols[1]:
        st.markdown("**Major Skill Gaps**")
        for item in structured["major_skill_gaps"]:
            st.warning(item)

        st.markdown("**Interview Recommendation**")
        st.metric("Decision", structured["interview_recommendation"])


def render_recruiter_insights(result: dict) -> None:
    insights = build_recruiter_insights(result)

    st.markdown("### Recruiter Insights")
    st.markdown(
        f"""
        <div class="insight-panel">
            <div class="metric-label">Candidate Level</div>
            <div class="insight-value">{html.escape(insights["candidate_level"])}</div>
            <div class="metric-caption">Estimated from resume experience, role titles, and project evidence.</div>
            <div class="metric-label insight-gap">Industry Signals</div>
            <div>{''.join(f'<span class="skill-pill info">{html.escape(industry)}</span>' for industry in insights["industries"])}</div>
            <div class="metric-label insight-gap">Best Fit Roles</div>
            <ol class="role-list">
                {''.join(f'<li>{html.escape(role)}</li>' for role in insights["best_fit_roles"])}
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_structured_recommendations(result: dict) -> None:
    recommendations = structured_recommendations(result)

    st.markdown("### Actionable Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**For Candidate**")
        for item in recommendations["candidate"]:
            st.markdown(f'<div class="recommendation-card candidate">{html.escape(item)}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("**For Recruiter**")
        for item in recommendations["recruiter"]:
            st.markdown(f'<div class="recommendation-card recruiter">{html.escape(item)}</div>', unsafe_allow_html=True)


def render_score_explanation(scores: dict) -> None:
    contributions = score_contributions(scores)

    st.markdown("### Explainable Score Breakdown")
    st.dataframe(
        contributions.assign(
            Weight=contributions["Weight"].map(lambda value: f"{value:.0%}"),
            Contribution=contributions["Contribution"].map(lambda value: f"{value:.2f}"),
        ),
        use_container_width=True,
        hide_index=True,
    )

    formula = " + ".join(
        f"{row.Component} {row.Contribution:.2f}"
        for row in contributions.itertuples()
    )
    st.caption(f"Overall Match = {formula} = {scores['overall_score']:.2f}%")


def render_single_result(result: dict) -> None:
    scores = result["scores"]
    matching = result["matching"]
    ats_result = result["ats"]

    st.success("Analysis Complete")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Overall Match", f"{scores['overall_score']}%", fit_label(scores["overall_score"]), scores["overall_score"])
    with col2:
        render_metric_card("Skill Match", f"{scores['skill_score']}%", "Detected skill overlap", scores["skill_score"])
    with col3:
        render_metric_card("Semantic Match", f"{scores['semantic_score']}%", "Resume-to-role relevance", scores["semantic_score"])
    with col4:
        render_metric_card("Experience", f"{scores['experience_score']}%", "Evidence-based estimate", scores["experience_score"])
    with col5:
        render_metric_card("ATS Score", f"{ats_result['ats_score']}%", "Screening compatibility", ats_result["ats_score"])

    insight_cols = st.columns([2.1, 1])
    with insight_cols[0]:
        render_score_explanation(scores)
    with insight_cols[1]:
        render_recruiter_insights(result)

    chart_cols = st.columns([1.2, 1, 1])
    with chart_cols[0]:
        st.plotly_chart(score_breakdown_chart(scores), use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(skill_coverage_chart(matching), use_container_width=True)
    with chart_cols[2]:
        st.plotly_chart(ats_gauge(ats_result["ats_score"]), use_container_width=True)

    st.plotly_chart(
        radar_chart(scores, ats_result["ats_score"]),
        use_container_width=True,
    )

    st.divider()
    render_profile(result)

    st.divider()
    render_ats_checklist(ats_result)

    st.divider()
    left, right = st.columns(2)
    with left:
        render_skill_groups(
            "Matched Skills",
            matching["matched_skills"],
            "No matched skills found.",
        )
    with right:
        render_skill_groups(
            "Missing Skills",
            matching["missing_skills"],
            "No major missing skills found.",
        )

    st.divider()
    render_feedback(result["feedback"])
    render_structured_recommendations(result)

    st.divider()
    st.markdown("### Recruiter Report Export")
    pdf_available = is_pdf_export_available()
    report_cols = st.columns(3 if pdf_available else 2)
    with report_cols[0]:
        st.download_button(
            "Download JSON report",
            build_json_report(result).encode("utf-8"),
            "talentmatch_report.json",
            "application/json",
            use_container_width=True,
        )
    with report_cols[1]:
        st.download_button(
            "Download HTML report",
            build_html_report(result).encode("utf-8"),
            "talentmatch_report.html",
            "text/html",
            use_container_width=True,
        )
    if pdf_available:
        with report_cols[2]:
            try:
                pdf_report = build_pdf_report(result)
            except RuntimeError:
                st.info(
                    "PDF export is disabled in the cloud deployment. JSON and HTML reports are available."
                )
            else:
                st.download_button(
                    "Download PDF report",
                    pdf_report,
                    "talentmatch_report.pdf",
                    "application/pdf",
                    use_container_width=True,
                )
    else:
        st.info(
            "PDF export is disabled in the cloud deployment. JSON and HTML reports are available."
        )


def render_ranking_results(ranked_candidates: list[dict]) -> None:
    ranking_df = pd.DataFrame(
        [
            {
                "Rank": index + 1,
                "Candidate": candidate["candidate_name"],
                "Overall Score": candidate["overall_score"],
                "ATS Score": candidate["ats_score"],
                "Skill Score": candidate["skill_score"],
                "Semantic Score": candidate["semantic_score"],
                "Experience Score": candidate["experience_score"],
                "Education Score": candidate["education_score"],
                "Matched Skills": ", ".join(candidate["matched_skills"]),
                "Missing Skills": ", ".join(candidate["missing_skills"]),
            }
            for index, candidate in enumerate(ranked_candidates)
        ]
    )

    st.success("Candidate ranking completed.")

    top_candidate = ranked_candidates[0]

    hero_cols = st.columns([1.4, 1, 1])
    with hero_cols[0]:
        st.markdown("### Top Candidate")
        st.markdown(
            f"""
            <div class="highlight-card">
                <div class="metric-label">Recommended for review</div>
                <div class="metric-value">{html.escape(top_candidate["candidate_name"])}</div>
                <div class="metric-caption">{top_candidate["overall_score"]}% overall match</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_cols[1]:
        render_metric_card("Top ATS Score", f"{top_candidate['ats_score']}%", "Screening readiness")
    with hero_cols[2]:
        render_metric_card("Top Skill Score", f"{top_candidate['skill_score']}%", "Requirement coverage")

    st.plotly_chart(ranking_chart(ranking_df), use_container_width=True)

    st.markdown("### Ranked Candidate Table")
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Export ranking CSV",
        ranking_df.to_csv(index=False).encode("utf-8"),
        "candidate_ranking.csv",
        "text/csv",
        use_container_width=True,
    )

    st.divider()
    left, right = st.columns(2)
    with left:
        render_skill_groups(
            "Top Candidate Strengths",
            top_candidate["matched_skills"],
            "No matched skills found.",
        )
    with right:
        render_skill_groups(
            "Top Candidate Missing Skills",
            top_candidate["missing_skills"],
            "No major missing skills found.",
        )


def render_landing() -> bool:
    st.markdown(
        """
        <section class="landing">
            <div>
                <div class="app-kicker">Recruiter intelligence dashboard</div>
                <h1>TalentMatch AI</h1>
                <p class="hero-subtitle">AI-powered Resume Screening, ATS Analysis and Candidate Ranking</p>
            </div>
            <div class="feature-grid">
                <div class="feature-card"><span>✓</span><strong>Resume Parsing</strong><p>Extract candidate profile, skills, experience, and project evidence.</p></div>
                <div class="feature-card"><span>✓</span><strong>ATS Compatibility</strong><p>Score resume structure, keyword coverage, and screening readiness.</p></div>
                <div class="feature-card"><span>✓</span><strong>AI Candidate Ranking</strong><p>Compare multiple resumes against one job description with explainable scores.</p></div>
            </div>
            <div class="workflow">
                <span>Upload Resume</span><b>→</b>
                <span>Upload Job Description</span><b>→</b>
                <span>AI Analysis</span><b>→</b>
                <span>Recruiter Report</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    demo_cols = st.columns([1, 2.4])
    with demo_cols[0]:
        return st.button(
            "🚀 Try Demo",
            type="primary",
            use_container_width=True,
            help="Run a complete demo using anonymized sample_resume.txt and sample_ai_engineer_jd.txt.",
        )
    with demo_cols[1]:
        st.caption("Use the demo to preview the recruiter dashboard without uploading files.")
    return False


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3rem !important;
        max-width: 1260px;
    }
    .landing {
        border: 1px solid #dbe3ee;
        border-radius: 8px;
        padding: 28px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        margin-bottom: 18px;
    }
    .landing h1 {
        font-size: 2.6rem;
        line-height: 1.15;
        margin: 0 0 0.35rem 0;
        color: #202636;
    }
    .hero-subtitle {
        color: #526173;
        font-size: 1.1rem;
        margin: 0 0 1.25rem 0;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 18px 0;
    }
    .feature-card {
        border: 1px solid #dbe3ee;
        border-radius: 8px;
        padding: 16px;
        background: #ffffff;
    }
    .feature-card span {
        color: #1b7f5f;
        font-weight: 800;
        margin-right: 6px;
    }
    .feature-card p {
        color: #697386;
        margin: 8px 0 0 0;
        font-size: 0.92rem;
    }
    .workflow {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
        color: #354052;
        font-weight: 700;
    }
    .workflow span {
        border: 1px solid #cfd8e3;
        border-radius: 999px;
        padding: 8px 12px;
        background: #f8fafc;
    }
    .app-kicker {
        color: #526173;
        font-size: 0.92rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 700;
        line-height: 1.5;
        margin: 0 0 0.65rem 0;
        padding-top: 0.25rem;
    }
    .metric-card, .highlight-card {
        border: 1px solid #dbe3ee;
        background: #ffffff;
        border-radius: 8px;
        padding: 16px 18px;
        min-height: 118px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .highlight-card {
        background: #f7fbff;
        border-color: #b9d7ff;
    }
    .metric-label {
        color: #687385;
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 700;
    }
    .metric-value {
        color: #202636;
        font-size: 1.8rem;
        line-height: 1.25;
        font-weight: 800;
        margin-top: 6px;
    }
    .metric-caption {
        color: #697386;
        font-size: 0.88rem;
        margin-top: 8px;
    }
    .score-state {
        color: white;
        border-radius: 999px;
        padding: 3px 8px;
        font-size: 0.72rem;
        font-weight: 800;
        margin-right: 6px;
    }
    .info-dot {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #edf4ff;
        color: #1f6feb;
        font-size: 0.68rem;
        text-transform: none;
        margin-left: 4px;
    }
    .skill-pill {
        display: inline-block;
        border: 1px solid #cfd8e3;
        border-radius: 999px;
        padding: 4px 10px;
        margin: 0 6px 8px 0;
        background: #f8fafc;
        color: #233044;
        font-size: 0.86rem;
    }
    .skill-pill.info {
        background: #edf4ff;
        border-color: #b9d7ff;
    }
    .insight-panel {
        border: 1px solid #dbe3ee;
        border-radius: 8px;
        background: #ffffff;
        padding: 16px;
        min-height: 236px;
    }
    .insight-value {
        color: #202636;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 6px;
    }
    .insight-gap {
        margin-top: 14px;
    }
    .role-list {
        margin: 8px 0 0 20px;
        color: #354052;
    }
    .recommendation-card {
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
        border: 1px solid #dbe3ee;
        background: #ffffff;
    }
    .recommendation-card.candidate {
        border-left: 4px solid #1f6feb;
    }
    .recommendation-card.recruiter {
        border-left: 4px solid #1b7f5f;
    }
    .check-row {
        display: grid;
        grid-template-columns: 92px 220px 1fr;
        gap: 12px;
        align-items: center;
        border-bottom: 1px solid #edf1f7;
        padding: 10px 0;
    }
    .status-pill {
        color: white;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 0.76rem;
        font-weight: 700;
        text-align: center;
    }
    div[data-testid="stFileUploader"] section {
        border-radius: 8px;
        padding: 0.45rem 0.7rem;
        min-height: 76px;
    }
    div[data-testid="stFileUploader"] section > div {
        padding: 0.25rem;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] {
        padding-top: 0;
        padding-bottom: 0;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] div {
        font-size: 0.82rem;
        line-height: 1.2;
    }
    @media (max-width: 900px) {
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


demo_clicked = render_landing()

if demo_clicked:
    if SAMPLE_RESUME_PATH.exists() and SAMPLE_JOB_PATH.exists():
        st.session_state["single_result"] = run_analysis_with_progress(
            str(SAMPLE_RESUME_PATH),
            str(SAMPLE_JOB_PATH),
        )
    else:
        st.error("Demo files are missing. Expected sample_resume.txt and sample_ai_engineer_jd.txt.")


tab1, tab2 = st.tabs(
    [
        "Single Resume Analysis",
        "Candidate Ranking",
    ]
)


with tab1:
    st.header("Single Resume Analysis")

    upload_cols = st.columns(2)
    with upload_cols[0]:
        resume_file = st.file_uploader(
            "Upload Resume",
            type=["pdf", "docx", "txt"],
            key="single_resume",
        )
    with upload_cols[1]:
        job_file = st.file_uploader(
            "Upload Job Description",
            type=["pdf", "docx", "txt"],
            key="single_job",
        )

    analyze_clicked = st.button(
        "Analyze Match",
        key="single_button",
        type="primary",
        use_container_width=True,
    )

    if analyze_clicked:
        if not resume_file or not job_file:
            st.warning("Please upload both resume and job description files.")
        else:
            st.session_state["single_result"] = run_analysis_with_progress(
                save_uploaded_file(resume_file),
                save_uploaded_file(job_file),
            )

    if "single_result" in st.session_state:
        render_single_result(st.session_state["single_result"])


with tab2:
    st.header("Candidate Ranking")

    upload_cols = st.columns(2)
    with upload_cols[0]:
        resumes = st.file_uploader(
            "Upload Multiple Resumes",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="batch_resumes",
        )
    with upload_cols[1]:
        batch_job_file = st.file_uploader(
            "Upload Job Description",
            type=["pdf", "docx", "txt"],
            key="batch_job",
        )

    rank_clicked = st.button(
        "Rank Candidates",
        key="rank_button",
        type="primary",
        use_container_width=True,
    )

    if rank_clicked:
        if not resumes or not batch_job_file:
            st.warning("Please upload multiple resumes and one job description.")
        else:
            with st.spinner("Ranking candidates..."):
                resume_paths = [save_uploaded_file(resume) for resume in resumes]
                job_path = save_uploaded_file(batch_job_file)
                st.session_state["ranked_candidates"] = CandidateRanker().rank_candidates(
                    resume_paths=resume_paths,
                    job_description_path=job_path,
                )

    if "ranked_candidates" in st.session_state:
        render_ranking_results(st.session_state["ranked_candidates"])
