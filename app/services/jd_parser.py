"""
Job description parser for TalentMatch AI.

This module extracts structured information from job descriptions.
"""

import re

from app.models.schemas import JobDescriptionData
from app.utils.file_loader import load_text_from_file
from app.utils.text_cleaner import preprocess_text


class JobDescriptionParser:
    """
    Parses job descriptions into structured JobDescriptionData objects.
    """

    def parse(self, file_path: str) -> JobDescriptionData:
        """
        Main entry point.

        Args:
            file_path: Job description file path.

        Returns:
            JobDescriptionData object.
        """

        raw_text = load_text_from_file(file_path)
        text = preprocess_text(raw_text)

        return JobDescriptionData(
            job_title=self._extract_job_title(raw_text),
            company=self._extract_company(raw_text),
            required_skills=[],
            preferred_skills=[],
            responsibilities=self._extract_responsibilities(text),
            qualifications=self._extract_qualifications(text),
            experience_required=self._extract_experience_required(text),
        )

    def _extract_job_title(self, raw_text: str):
        """
        Attempts to extract job title from the first few lines.
        """

        lines = [
            line.strip()
            for line in raw_text.splitlines()
            if line.strip()
        ]

        for line in lines[:5]:
            if any(keyword.lower() in line.lower() for keyword in ["engineer", "analyst", "scientist", "developer", "intern"]):
                return line

        return lines[0] if lines else None

    def _extract_company(self, raw_text: str):
        """
        Basic company extraction placeholder.
        """

        lines = [
            line.strip()
            for line in raw_text.splitlines()
            if line.strip()
        ]

        return lines[1] if len(lines) > 1 else None

    def _extract_responsibilities(self, text: str):

        keywords = [
            "responsibilities",
            "responsible",
            "develop",
            "build",
            "analyze",
            "design",
            "maintain",
        ]

        return self._find_lines(text, keywords)

    def _extract_qualifications(self, text: str):

        keywords = [
            "qualification",
            "requirements",
            "required",
            "degree",
            "bachelor",
            "master",
            "experience",
        ]

        return self._find_lines(text, keywords)

    def _extract_experience_required(self, text: str):

        match = re.search(
            r"(\d+\+?\s*(years|year|yrs|yr))",
            text,
        )

        return match.group(0) if match else None

    @staticmethod
    def _find_lines(text: str, keywords):

        results = []

        for line in text.split("\n"):
            line = line.strip()

            if any(keyword in line for keyword in keywords):
                results.append(line)

        return results