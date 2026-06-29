"""
Scoring module for TalentMatch AI.

This module calculates the final resume-to-job compatibility score
using multiple weighted scoring components.
"""

from app.config import (
    SKILL_MATCH_WEIGHT,
    SEMANTIC_SIMILARITY_WEIGHT,
    EXPERIENCE_MATCH_WEIGHT,
    EDUCATION_MATCH_WEIGHT,
)


class ScoringEngine:
    """
    Calculates final candidate-job match score.
    """

    def calculate_final_score(
        self,
        skill_score: float,
        semantic_score: float,
        experience_score: float = 0.0,
        education_score: float = 0.0,
    ) -> dict:
        """
        Calculate weighted final score.
        """

        overall_score = (
            (skill_score * SKILL_MATCH_WEIGHT)
            + (semantic_score * SEMANTIC_SIMILARITY_WEIGHT)
            + (experience_score * EXPERIENCE_MATCH_WEIGHT)
            + (education_score * EDUCATION_MATCH_WEIGHT)
        )

        return {
            "overall_score": round(overall_score, 2),
            "skill_score": round(skill_score, 2),
            "semantic_score": round(semantic_score, 2),
            "experience_score": round(experience_score, 2),
            "education_score": round(education_score, 2),
        }

    def estimate_experience_score(
        self,
        resume_experience: list[str],
        job_experience_required: str | None,
    ) -> float:
        """
        Estimate experience alignment.
        """

        if not job_experience_required:
            return 70.0

        if resume_experience:
            return 75.0

        return 40.0

    def estimate_education_score(
        self,
        resume_education: list[str],
        job_qualifications: list[str],
    ) -> float:
        """
        Estimate education alignment.
        """

        if not job_qualifications:
            return 70.0

        if resume_education:
            return 80.0

        return 45.0