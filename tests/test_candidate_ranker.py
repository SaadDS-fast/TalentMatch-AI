"""
Unit tests for the Candidate Ranker.

These tests use temporary text files to simulate resumes and job descriptions.
"""

from pathlib import Path

from app.services.candidate_ranker import CandidateRanker


def test_candidate_ranking_orders_best_candidate_first(tmp_path):
    """
    Candidate with stronger skill alignment should rank higher.
    """

    job_description = tmp_path / "job_description.txt"
    resume_strong = tmp_path / "resume_strong.txt"
    resume_weak = tmp_path / "resume_weak.txt"

    job_description.write_text(
        """
        Machine Learning Engineer
        Requirements: Python, SQL, FastAPI, Docker, Machine Learning
        Education: Bachelor or Master degree required
        """,
        encoding="utf-8",
    )

    resume_strong.write_text(
        """
        Demo Candidate One
        Email: demo.one@example.com
        Phone: +1 555 010 1001
        Education: MS Data Science
        Skills: Python, SQL, FastAPI, Docker, Machine Learning
        Projects: Built ML APIs using FastAPI and Docker.
        """,
        encoding="utf-8",
    )

    resume_weak.write_text(
        """
        Candidate Two
        Email: candidate@example.com
        Phone: +1 555 010 1002
        Education: BS Mathematics
        Skills: MATLAB, Excel
        Projects: Academic mathematics projects.
        """,
        encoding="utf-8",
    )

    results = CandidateRanker().rank_candidates(
        resume_paths=[
            str(resume_weak),
            str(resume_strong),
        ],
        job_description_path=str(job_description),
    )

    assert len(results) == 2
    assert results[0]["candidate_name"] == "Demo Candidate One"
    assert results[0]["overall_score"] >= results[1]["overall_score"]
