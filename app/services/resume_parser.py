"""
Resume parser for TalentMatch AI.

Extracts practical structured fields from resume documents while leaving
advanced NLP enrichment to the skill extraction and matching services.
"""

import re

from app.models.schemas import ResumeData
from app.utils.file_loader import load_text_from_file
from app.utils.text_cleaner import clean_text


class ResumeParser:
    """
    Parses resumes into structured ResumeData objects.
    """

    SECTION_KEYWORDS = {
        "education": ("education", "academic", "qualification"),
        "experience": (
            "experience",
            "teaching experience",
            "work experience",
            "professional experience",
            "employment",
            "work history",
            "internship",
        ),
        "projects": (
            "projects",
            "academic projects",
            "personal projects",
            "research projects",
            "portfolio projects",
            "portfolio",
        ),
        "skills": ("skills", "technical skills", "technologies"),
        "certifications": ("certifications", "certificates", "certification"),
    }

    EXPERIENCE_ROLE_PATTERN = re.compile(
        r"\b(teacher|intern|engineer|developer|analyst|assistant|researcher|scientist|consultant|manager|lead)\b",
        re.IGNORECASE,
    )

    DATE_RANGE_PATTERN = re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)?\.?\s*\d{4}\s*[-–—]\s*(?:present|current|now|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)?\.?\s*\d{4})\b|\b\d{4}\s*[-–—]\s*(?:present|current|now|\d{4})\b",
        re.IGNORECASE,
    )

    def parse(self, file_path: str) -> ResumeData:
        """
        Parse a resume file path into ResumeData.
        """

        raw_text = load_text_from_file(file_path)
        text = clean_text(raw_text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        sections = self._extract_sections(lines)

        self._enrich_evidence_from_patterns(sections, lines)

        return ResumeData(
            candidate_name=self._extract_name(lines),
            email=self._extract_email(text),
            phone=self._extract_phone(text),
            education=sections["education"],
            experience=sections["experience"],
            projects=sections["projects"],
            skills=sections["skills"],
            certifications=sections["certifications"],
        )

    def _extract_sections(self, lines: list[str]) -> dict[str, list[str]]:
        sections = {section: [] for section in self.SECTION_KEYWORDS}
        active_section = None

        for line in lines:
            normalized = line.lower().strip(" :")
            matched_section = self._section_for_heading(normalized)

            if matched_section:
                active_section = matched_section
                inline_value = self._inline_section_value(line)
                if inline_value:
                    sections[active_section].append(inline_value)
                continue

            if active_section and self._looks_like_major_heading(line):
                active_section = None
                continue

            if active_section and not self._looks_like_contact_line(line):
                sections[active_section].append(line)

        return sections

    def _looks_like_major_heading(self, line: str) -> bool:
        normalized = line.lower().strip(" :")

        if self._section_for_heading(normalized):
            return True

        known_headings = {
            "awards",
            "achievements",
            "languages",
            "publications",
            "summary",
            "profile",
            "objective",
            "interests",
            "references",
            "volunteer experience",
            "leadership",
        }

        return (
            normalized in known_headings
            or (
                len(line.split()) <= 4
                and line[:1].isupper()
                and not line.endswith(".")
                and not self._is_experience_evidence(line)
                and not self._is_project_title(line, 0, [])
            )
        )

    def _section_for_heading(self, normalized_line: str) -> str | None:
        ordered_sections = [
            "projects",
            "experience",
            "skills",
            "certifications",
            "education",
        ]

        for section in ordered_sections:
            keywords = self.SECTION_KEYWORDS[section]
            if any(self._heading_matches(normalized_line, keyword) for keyword in keywords):
                return section

        return None

    @staticmethod
    def _heading_matches(normalized_line: str, keyword: str) -> bool:
        normalized_keyword = keyword.lower()

        if normalized_line == normalized_keyword:
            return True

        if normalized_line.startswith(f"{normalized_keyword}:"):
            return True

        if (
            " " in normalized_keyword
            and len(normalized_line.split()) <= 4
            and normalized_keyword in normalized_line
        ):
            return True

        return False

    def _enrich_evidence_from_patterns(
        self,
        sections: dict[str, list[str]],
        lines: list[str],
    ) -> None:
        for index, line in enumerate(lines):
            if self._looks_like_contact_line(line):
                continue

            if self._is_experience_evidence(line):
                self._append_unique(sections["experience"], line)

                if index + 1 < len(lines) and self.DATE_RANGE_PATTERN.search(lines[index + 1]):
                    self._append_unique(sections["experience"], lines[index + 1])

            if self._is_project_title(line, index, lines):
                self._append_unique(sections["projects"], line)

    def _is_experience_evidence(self, line: str) -> bool:
        return bool(
            self.EXPERIENCE_ROLE_PATTERN.search(line)
            or self.DATE_RANGE_PATTERN.search(line)
        )

    def _is_project_title(self, line: str, index: int, lines: list[str]) -> bool:
        normalized = line.lower().strip(" :-")

        if self._section_for_heading(normalized) == "projects":
            return False

        if index > 0 and self._section_for_heading(lines[index - 1].lower().strip(" :")) == "projects":
            return True

        project_signal_words = (
            "optimization",
            "classification",
            "summarization",
            "prediction",
            "detection",
            "dashboard",
            "analysis",
            "system",
            "application",
            "model",
        )

        return (
            2 <= len(line.split()) <= 12
            and any(word in normalized for word in project_signal_words)
            and not self._looks_like_contact_line(line)
        )

    @staticmethod
    def _append_unique(items: list[str], item: str) -> None:
        normalized_existing = {existing.lower().strip() for existing in items}
        normalized_item = item.lower().strip()

        if normalized_item and normalized_item not in normalized_existing:
            items.append(item)

    @staticmethod
    def _inline_section_value(line: str) -> str | None:
        if ":" not in line:
            return None

        value = line.split(":", 1)[1].strip()
        return value or None

    @staticmethod
    def _extract_name(lines: list[str]) -> str | None:
        for line in lines[:8]:
            if ResumeParser._looks_like_contact_line(line):
                continue

            lowered = line.lower().strip(" :")
            if lowered in {"resume", "curriculum vitae", "cv"}:
                continue

            if len(line.split()) <= 6 and re.search(r"[A-Za-z]", line):
                return line

        return None

    @staticmethod
    def _extract_email(text: str) -> str | None:
        match = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)
        return match.group(0) if match else None

    @staticmethod
    def _extract_phone(text: str) -> str | None:
        match = re.search(
            r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4})",
            text,
        )
        return match.group(0).strip() if match else None

    @staticmethod
    def _looks_like_contact_line(line: str) -> bool:
        lowered = line.lower()
        return (
            "@" in line
            or "phone" in lowered
            or "email" in lowered
            or "linkedin" in lowered
            or "github" in lowered
        )
