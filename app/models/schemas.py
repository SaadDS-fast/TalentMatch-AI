"""
Pydantic schemas for TalentMatch AI.

These models define the data structures used across the application.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# ==========================================================
# Resume Models
# ==========================================================

class ResumeData(BaseModel):
    """
    Structured information extracted from a candidate's resume.
    """

    candidate_name: Optional[str] = None

    email: Optional[str] = None

    phone: Optional[str] = None

    education: List[str] = Field(default_factory=list)

    experience: List[str] = Field(default_factory=list)

    projects: List[str] = Field(default_factory=list)

    skills: List[str] = Field(default_factory=list)

    certifications: List[str] = Field(default_factory=list)


# ==========================================================
# Job Description Models
# ==========================================================

class JobDescriptionData(BaseModel):
    """
    Structured information extracted from a job description.
    """

    job_title: Optional[str] = None

    company: Optional[str] = None

    required_skills: List[str] = Field(default_factory=list)

    preferred_skills: List[str] = Field(default_factory=list)

    responsibilities: List[str] = Field(default_factory=list)

    qualifications: List[str] = Field(default_factory=list)

    experience_required: Optional[str] = None


# ==========================================================
# Match Result Models
# ==========================================================

class MatchResult(BaseModel):
    """
    Complete matching result returned to the frontend/API.
    """

    overall_score: float

    skill_score: float

    semantic_score: float

    experience_score: float

    education_score: float

    ats_score: float

    matched_skills: List[str]

    missing_skills: List[str]

    ats_issues: List[str]

    recommendations: List[str]

    recruiter_summary: str


# ==========================================================
# API Models
# ==========================================================

class MatchRequest(BaseModel):

    resume_path: str

    job_description_path: str


class MatchResponse(BaseModel):

    success: bool

    result: MatchResult