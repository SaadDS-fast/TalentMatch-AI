"""
API routes for TalentMatch AI.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    MatchRequest,
    MatchResponse,
    MatchResult,
)
from app.services.resume_parser import ResumeParser
from app.services.jd_parser import JobDescriptionParser
from app.services.skill_extractor import SkillExtractor
from app.services.matching_engine import MatchingEngine
from app.services.scoring import ScoringEngine
from app.services.llm_feedback import LLMFeedbackGenerator
from app.services.semantic_similarity import SemanticSimilarityEngine
from app.services.ats_scorer import ATSScorer
from app.utils.file_loader import load_text_from_file


router = APIRouter(prefix="/match", tags=["TalentMatch"])


@router.post("/", response_model=MatchResponse)
def match_resume(request: MatchRequest):
    """
    Match a resume against a job description.
    """

    try:
        resume_raw_text = load_text_from_file(request.resume_path)
        job_raw_text = load_text_from_file(request.job_description_path)

        resume = ResumeParser().parse(request.resume_path)
        job = JobDescriptionParser().parse(request.job_description_path)

        extractor = SkillExtractor()

        resume.skills = extractor.extract_skills(resume_raw_text)
        job.required_skills = extractor.extract_skills(job_raw_text)

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

        feedback = LLMFeedbackGenerator().generate_feedback(
            overall_score=scores["overall_score"],
            matched_skills=matching["matched_skills"],
            missing_skills=matching["missing_skills"],
            skill_score=scores["skill_score"],
            ats_score=ats_result["ats_score"],
            experience_score=scores["experience_score"],
        )

        result = MatchResult(
            overall_score=scores["overall_score"],
            skill_score=scores["skill_score"],
            semantic_score=scores["semantic_score"],
            experience_score=scores["experience_score"],
            education_score=scores["education_score"],
            ats_score=ats_result["ats_score"],
            matched_skills=matching["matched_skills"],
            missing_skills=matching["missing_skills"],
            ats_issues=ats_result["ats_issues"],
            recommendations=feedback["recommendations"],
            recruiter_summary=feedback["recruiter_summary"],
        )

        return MatchResponse(
            success=True,
            result=result,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
