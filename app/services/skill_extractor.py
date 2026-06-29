"""
Skill extraction service for TalentMatch AI.

This module identifies skills from resume and job description text
using an external JSON-based skill database.
"""

import json
import re

from app.config import DATA_DIR
from app.services.skill_normalizer import skill_pattern
from app.utils.text_cleaner import preprocess_text


class SkillExtractor:
    """
    Extracts known skills from text using a structured skill database.
    """

    def __init__(self):
        self.skill_database = self._load_skills_database()

    def extract_skills(self, text: str) -> list[str]:
        """
        Extract skills from input text.
        """

        cleaned_text = preprocess_text(text)
        matched_skills = []

        for skill in self.skill_database:
            pattern = skill_pattern(skill)

            if re.search(pattern, cleaned_text):
                matched_skills.append(skill)

        return sorted(set(matched_skills))

    def _load_skills_database(self) -> list[str]:
        """
        Load skills from data/skills_database/skills.json.
        """

        skills_file = DATA_DIR / "skills_database" / "skills.json"

        if not skills_file.exists():
            raise FileNotFoundError(
                f"Skills database not found at: {skills_file}"
            )

        with open(skills_file, "r", encoding="utf-8") as file:
            skills_data = json.load(file)

        all_skills = []

        for category_skills in skills_data.values():
            all_skills.extend(category_skills)

        return sorted(set(all_skills))
