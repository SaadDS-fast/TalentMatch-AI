"""
Candidate ranking service for TalentMatch AI.

This module ranks multiple resumes against a single job description.
It is useful for recruiter-style batch screening.
"""

from pathlib import Path

from app.services.resume_parser import ResumeParser
from app.services.jd_parser import JobDescriptionParser
from app.services.skill_extractor import SkillExtractor
from app.services.matching_engine import MatchingEngine
from app.services.scoring import ScoringEngine
from app.services.semantic_similarity import SemanticSimilarityEngine
from app.services.ats_scorer import ATSScorer
from app.utils.file_loader import load_text_from_file


class CandidateRanker:
    """
    Ranks multiple candidates for one job description.
    """

    def rank_candidates(
        self,
        resume_paths: list[str],
        job_description_path: str,
    ) -> list[dict]:
        """
        Rank resumes based on final match score.
        """

        job_raw_text = load_text_from_file(job_description_path)
        job = JobDescriptionParser().parse(job_description_path)

        extractor = SkillExtractor()
        job.required_skills = extractor.extract_skills(job_raw_text)

        results = []

        for resume_path in resume_paths:
            resume_raw_text = load_text_from_file(resume_path)
            resume = ResumeParser().parse(resume_path)

            resume.skills = extractor.extract_skills(resume_raw_text)

            matching = MatchingEngine().match(resume, job)

            semantic_score = SemanticSimilarityEngine().calculate_similarity(
                resume_raw_text,
                job_raw_text,
                resume_data=resume.model_dump(),
                job_data=job.model_dump(),
            )

            scoring_engine = ScoringEngine()

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

            results.append(
                {
                    "candidate_name": resume.candidate_name
                    or Path(resume_path).stem,
                    "resume_path": resume_path,
                    "overall_score": scores["overall_score"],
                    "skill_score": scores["skill_score"],
                    "semantic_score": scores["semantic_score"],
                    "experience_score": scores["experience_score"],
                    "education_score": scores["education_score"],
                    "ats_score": ats_result["ats_score"],
                    "matched_skills": matching["matched_skills"],
                    "missing_skills": matching["missing_skills"],
                    "ats_issues": ats_result["ats_issues"],
                }
            )

        return sorted(
            results,
            key=lambda candidate: candidate["overall_score"],
            reverse=True,
        )
