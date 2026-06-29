"""
ATS scoring service for TalentMatch AI.

This module estimates how ATS-friendly a resume is based on
structure, keyword coverage, contact information, and readability.
"""

import re


class ATSScorer:
    """
    Calculates an ATS compatibility score for a resume.
    """

    def calculate_score(
        self,
        resume_text: str,
        matched_skills: list[str],
        missing_skills: list[str],
    ) -> dict:
        """
        Calculate ATS score out of 100.
        """

        checklist = self.build_checklist(
            resume_text=resume_text,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
        )

        score = 100
        issues = []

        for item in checklist:
            if item["status"] == "Missing":
                score -= item["penalty"]
                issues.append(item["message"])
            elif item["status"] == "Warning":
                score -= item["penalty"]
                issues.append(item["message"])

        score = max(score, 0)

        if not issues:
            issues.append("Resume appears ATS-friendly based on the available checks.")

        return {
            "ats_score": round(score, 2),
            "ats_issues": issues,
            "ats_checklist": checklist,
        }

    def build_checklist(
        self,
        resume_text: str,
        matched_skills: list[str],
        missing_skills: list[str],
    ) -> list[dict]:
        """
        Build recruiter-facing ATS checklist items.
        """

        lower_text = resume_text.lower()
        word_count = len(resume_text.split())
        total_skills = len(matched_skills) + len(missing_skills)
        coverage = len(matched_skills) / total_skills if total_skills else 0.0

        checklist = [
            self._check_item(
                label="Contact information found",
                passed=("@" in resume_text and any(char.isdigit() for char in resume_text)),
                missing_message="Email or phone number is missing or not clearly visible.",
                penalty=10,
            ),
            self._section_item("Skills section found", "skills", lower_text),
            self._section_item("Education section found", "education", lower_text),
            self._section_item("Projects section found", "projects", lower_text),
            self._section_item("Experience section found", "experience", lower_text),
            self._project_evidence_item(lower_text),
            self._experience_evidence_item(resume_text),
            self._length_item(word_count),
            self._coverage_item(coverage, total_skills),
            self._important_keyword_item(missing_skills),
            self._ai_engineer_readiness_item(matched_skills, missing_skills),
        ]

        return checklist

    @staticmethod
    def _check_item(
        label: str,
        passed: bool,
        missing_message: str,
        penalty: int,
    ) -> dict:
        return {
            "label": label,
            "status": "Passed" if passed else "Missing",
            "message": "Requirement satisfied." if passed else missing_message,
            "penalty": 0 if passed else penalty,
        }

    def _section_item(self, label: str, section: str, lower_text: str) -> dict:
        return self._check_item(
            label=label,
            passed=section in lower_text,
            missing_message=f"Resume section missing: {section.title()}.",
            penalty=8,
        )

    @staticmethod
    def _length_item(word_count: int) -> dict:
        if 150 <= word_count <= 1200:
            return {
                "label": "Resume length acceptable",
                "status": "Passed",
                "message": f"{word_count} words detected.",
                "penalty": 0,
            }

        return {
            "label": "Resume length acceptable",
            "status": "Warning",
            "message": (
                f"{word_count} words detected; resumes usually screen best "
                "between 150 and 1200 words."
            ),
            "penalty": 10,
        }

    @staticmethod
    def _coverage_item(coverage: float, total_skills: int) -> dict:
        if total_skills == 0:
            return {
                "label": "Keyword coverage acceptable",
                "status": "Warning",
                "message": "No job skills were detected for keyword coverage analysis.",
                "penalty": 5,
            }

        if coverage >= 0.70:
            status = "Passed"
            penalty = 0
            message = f"{coverage:.0%} of detected job skills are present."
        elif coverage >= 0.40:
            status = "Warning"
            penalty = 10
            message = f"{coverage:.0%} keyword coverage; some important skills are missing."
        else:
            status = "Missing"
            penalty = 20
            message = f"{coverage:.0%} keyword coverage; alignment is low."

        return {
            "label": "Keyword coverage acceptable",
            "status": status,
            "message": message,
            "penalty": penalty,
        }

    @staticmethod
    def _project_evidence_item(lower_text: str) -> dict:
        project_terms = (
            "project",
            "classification",
            "optimization",
            "summarization",
            "prediction",
            "dashboard",
            "portfolio",
        )
        passed = any(term in lower_text for term in project_terms)
        return {
            "label": "Project evidence found",
            "status": "Passed" if passed else "Warning",
            "message": (
                "Project evidence is visible."
                if passed
                else "No clear project evidence was detected in the resume."
            ),
            "penalty": 0 if passed else 8,
        }

    @staticmethod
    def _experience_evidence_item(resume_text: str) -> dict:
        role_pattern = r"\b(teacher|intern|engineer|developer|analyst|assistant|researcher|scientist|consultant|manager|lead)\b"
        date_pattern = r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)?\.?\s*\d{4}\s*[-–—]\s*(?:present|current|now|\d{4})\b"
        passed = bool(
            re.search(role_pattern, resume_text, re.I)
            or re.search(date_pattern, resume_text, re.I)
        )
        return {
            "label": "Experience evidence found",
            "status": "Passed" if passed else "Warning",
            "message": (
                "Experience evidence is visible."
                if passed
                else "No clear role title or date-based experience evidence was detected."
            ),
            "penalty": 0 if passed else 8,
        }

    @staticmethod
    def _important_keyword_item(missing_skills: list[str]) -> dict:
        important_missing = missing_skills[:5]
        if not important_missing:
            return {
                "label": "Important job keywords present",
                "status": "Passed",
                "message": "No high-priority missing job keywords were detected.",
                "penalty": 0,
            }

        return {
            "label": "Important job keywords present",
            "status": "Warning" if len(important_missing) <= 3 else "Missing",
            "message": "Missing important job keywords: " + ", ".join(important_missing),
            "penalty": min(15, len(important_missing) * 3),
        }

    @staticmethod
    def _ai_engineer_readiness_item(
        matched_skills: list[str],
        missing_skills: list[str],
    ) -> dict:
        all_job_skills = {skill.lower() for skill in matched_skills + missing_skills}
        ai_terms = {"machine learning", "tensorflow", "pytorch", "transformers", "nlp", "llm"}
        deployment_terms = {"fastapi", "docker", "aws", "azure", "gcp", "rest api", "api development"}

        if not all_job_skills.intersection(ai_terms):
            return {
                "label": "AI deployment readiness",
                "status": "Passed",
                "message": "AI deployment skills are not central in the detected job keywords.",
                "penalty": 0,
            }

        matched_lower = {skill.lower() for skill in matched_skills}
        missing_deployment = sorted(deployment_terms.intersection(all_job_skills) - matched_lower)

        if not missing_deployment:
            return {
                "label": "AI deployment readiness",
                "status": "Passed",
                "message": "Detected AI role keywords have supporting API/deployment coverage.",
                "penalty": 0,
            }

        return {
            "label": "AI deployment readiness",
            "status": "Warning",
            "message": "Missing deployment/API evidence for AI role: " + ", ".join(missing_deployment),
            "penalty": min(12, len(missing_deployment) * 4),
        }
