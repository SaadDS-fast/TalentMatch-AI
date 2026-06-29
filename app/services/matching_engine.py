"""
Matching engine for TalentMatch AI.

This module compares resume skills with job description skills
and computes similarity metrics.
"""

from difflib import SequenceMatcher

from app.models.schemas import ResumeData, JobDescriptionData
from app.services.skill_normalizer import normalize_skill_name


class MatchingEngine:
    """
    Performs resume-to-job matching.
    """

    def __init__(self, similarity_threshold: float = 0.80):
        self.similarity_threshold = similarity_threshold

    def match(
        self,
        resume: ResumeData,
        job: JobDescriptionData,
    ) -> dict:
        """
        Compare resume and job description.

        Returns:
            Dictionary containing matched skills,
            missing skills,
            similarity scores.
        """

        matched_skills = []
        missing_skills = []

        resume_skills = self._normalized_skill_map(resume.skills)
        job_skills = self._normalized_skill_map(
            job.required_skills + job.preferred_skills
        )

        # -----------------------------------------
        # Match skills
        # -----------------------------------------

        matched_normalized_skills = set()

        for job_skill_normalized, job_skill in job_skills.items():

            found = False

            for resume_skill_normalized, resume_skill in resume_skills.items():

                similarity = self._similarity(
                    job_skill_normalized,
                    resume_skill_normalized,
                )

                if similarity >= self.similarity_threshold:

                    matched_skills.append(job_skill)
                    matched_normalized_skills.add(job_skill_normalized)

                    found = True

                    break

            if not found:
                missing_skills.append(job_skill)

        # -----------------------------------------
        # Score
        # -----------------------------------------

        missing_skills = [
            skill
            for normalized_skill, skill in job_skills.items()
            if normalized_skill not in matched_normalized_skills
        ]

        total_required = len(job_skills)

        if total_required == 0:
            skill_match_score = 0.0
        else:
            skill_match_score = (
                len(matched_normalized_skills) / total_required
            ) * 100

        return {
            "matched_skills": sorted(set(matched_skills)),
            "missing_skills": sorted(set(missing_skills)),
            "skill_match_score": round(skill_match_score, 2),
        }

    # ---------------------------------------------------------

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """
        Compute string similarity.

        Current implementation:
        SequenceMatcher

        Future implementation:
        Sentence Transformers embeddings.
        """

        return SequenceMatcher(
            None,
            a,
            b,
        ).ratio()

    @classmethod
    def _normalized_skill_map(cls, skills: list[str]) -> dict[str, str]:
        normalized_skills = {}

        for skill in skills:
            normalized_skill = normalize_skill_name(skill)
            if normalized_skill and normalized_skill not in normalized_skills:
                normalized_skills[normalized_skill] = skill

        return normalized_skills
