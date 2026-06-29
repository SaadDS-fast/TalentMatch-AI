"""
Unit tests for resume evidence extraction.
"""

from app.services.resume_parser import ResumeParser


def test_resume_parser_detects_teaching_experience_and_dates(tmp_path):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        """
        Demo Candidate
        demo.parser@example.com
        +1 555 010 2001

        Teaching Experience
        Mathematics Teacher
        Oct 2024-Present
        Designed lesson plans and assessed student performance.
        """,
        encoding="utf-8",
    )

    parsed = ResumeParser().parse(str(resume))

    evidence = " ".join(parsed.experience)

    assert "Mathematics Teacher" in evidence
    assert "Oct 2024-Present" in evidence


def test_resume_parser_detects_academic_project_titles(tmp_path):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        """
        Demo Candidate
        demo.parser@example.com
        +1 555 010 2002

        Academic Projects
        Regime-Aware Stock Portfolio Optimization
        Explainable CNN for Brain Tumor Classification
        Multimodal Speech Summarization
        """,
        encoding="utf-8",
    )

    parsed = ResumeParser().parse(str(resume))

    evidence = " ".join(parsed.projects)

    assert "Regime-Aware Stock Portfolio Optimization" in evidence
    assert "Explainable CNN for Brain Tumor Classification" in evidence
    assert "Multimodal Speech Summarization" in evidence
