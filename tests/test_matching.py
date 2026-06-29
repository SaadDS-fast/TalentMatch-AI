"""
Unit tests for the Matching Engine.
"""

from app.models.schemas import ResumeData, JobDescriptionData
from app.services.matching_engine import MatchingEngine


def test_perfect_skill_match():
    """
    Candidate has all required skills.
    """

    resume = ResumeData(
        skills=[
            "Python",
            "SQL",
            "FastAPI",
            "Docker",
        ]
    )

    job = JobDescriptionData(
        required_skills=[
            "Python",
            "SQL",
            "FastAPI",
            "Docker",
        ]
    )

    engine = MatchingEngine()

    result = engine.match(resume, job)

    assert result["skill_match_score"] == 100.0
    assert len(result["missing_skills"]) == 0


def test_partial_skill_match():
    """
    Candidate has some required skills.
    """

    resume = ResumeData(
        skills=[
            "Python",
            "SQL",
        ]
    )

    job = JobDescriptionData(
        required_skills=[
            "Python",
            "SQL",
            "Docker",
            "FastAPI",
        ]
    )

    engine = MatchingEngine()

    result = engine.match(resume, job)

    assert result["skill_match_score"] == 50.0
    assert "Docker" in result["missing_skills"]
    assert "FastAPI" in result["missing_skills"]


def test_no_skill_match():
    """
    Candidate has no matching skills.
    """

    resume = ResumeData(
        skills=[
            "MATLAB",
            "R",
        ]
    )

    job = JobDescriptionData(
        required_skills=[
            "Python",
            "SQL",
        ]
    )

    engine = MatchingEngine()

    result = engine.match(resume, job)

    assert result["skill_match_score"] == 0.0
    assert len(result["matched_skills"]) == 0


def test_empty_resume():
    """
    Empty resume should not crash the system.
    """

    resume = ResumeData()

    job = JobDescriptionData(
        required_skills=[
            "Python",
            "SQL",
        ]
    )

    engine = MatchingEngine()

    result = engine.match(resume, job)

    assert result["skill_match_score"] == 0.0


def test_empty_job_description():
    """
    Empty job description should return 0 missing skills.
    """

    resume = ResumeData(
        skills=[
            "Python",
            "SQL",
        ]
    )

    job = JobDescriptionData()

    engine = MatchingEngine()

    result = engine.match(resume, job)

    assert result["skill_match_score"] == 0.0
    assert len(result["missing_skills"]) == 0


def test_skill_variants_do_not_appear_as_both_matched_and_missing():
    """
    Normalized variants such as FastAPI/Fast API and scikit-learn/scikit learn
    should match once and never appear in both buckets.
    """

    resume = ResumeData(
        skills=[
            "python",
            "Fast API",
            "numpy",
            "scikit learn",
        ]
    )

    job = JobDescriptionData(
        required_skills=[
            "Python",
            "FastAPI",
            "NumPy",
            "scikit-learn",
        ]
    )

    result = MatchingEngine().match(resume, job)

    assert result["skill_match_score"] == 100.0
    assert result["missing_skills"] == []
    assert set(result["matched_skills"]) == {
        "Python",
        "FastAPI",
        "NumPy",
        "scikit-learn",
    }


def test_skill_aliases_match_canonical_names():
    resume = ResumeData(
        skills=[
            "NLP",
            "sklearn",
            "vector db",
        ]
    )

    job = JobDescriptionData(
        required_skills=[
            "Natural Language Processing",
            "scikit-learn",
            "Vector Database",
        ]
    )

    result = MatchingEngine().match(resume, job)

    assert result["skill_match_score"] == 100.0
    assert result["missing_skills"] == []
    assert set(result["matched_skills"]) == {
        "Natural Language Processing",
        "scikit-learn",
        "Vector Database",
    }
