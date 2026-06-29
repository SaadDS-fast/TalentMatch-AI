"""
Unit tests for the Scoring Engine.
"""

from app.services.scoring import ScoringEngine


def test_final_score_calculation():
    scoring = ScoringEngine()

    result = scoring.calculate_final_score(
        skill_score=80.0,
        semantic_score=70.0,
        experience_score=60.0,
        education_score=90.0,
    )

    assert result["overall_score"] == 75.0
    assert result["skill_score"] == 80.0
    assert result["semantic_score"] == 70.0


def test_experience_score_without_requirement():
    scoring = ScoringEngine()

    score = scoring.estimate_experience_score(
        resume_experience=[],
        job_experience_required=None,
    )

    assert score == 70.0


def test_experience_score_with_resume_experience():
    scoring = ScoringEngine()

    score = scoring.estimate_experience_score(
        resume_experience=["AI Intern at XYZ"],
        job_experience_required="1 year",
    )

    assert score == 75.0


def test_education_score_with_resume_education():
    scoring = ScoringEngine()

    score = scoring.estimate_education_score(
        resume_education=["MS Data Science"],
        job_qualifications=["Bachelor or Master degree required"],
    )

    assert score == 80.0