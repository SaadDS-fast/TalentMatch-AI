"""
Text cleaning utilities for TalentMatch AI.

This module standardizes extracted text before it is passed
to the NLP pipeline.
"""

import re


def clean_text(text: str) -> str:
    """
    Perform basic text cleaning.

    Steps:
    - Remove extra spaces
    - Remove tabs
    - Normalize newlines
    - Remove repeated blank lines

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """

    if not text:
        return ""

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove multiple spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_case(text: str) -> str:
    """
    Convert text to lowercase.

    Useful for matching operations.
    """

    return text.lower().strip()


def remove_special_characters(text: str) -> str:
    """
    Remove unwanted special characters while preserving
    common punctuation.
    """

    return re.sub(r"[^a-zA-Z0-9.,@+\-#/()\n ]", "", text)


def tokenize_lines(text: str) -> list[str]:
    """
    Split cleaned text into non-empty lines.
    """

    return [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]


def preprocess_text(text: str) -> str:
    """
    Complete preprocessing pipeline.

    This is the main function other modules should call.
    """

    text = clean_text(text)
    text = normalize_case(text)
    text = remove_special_characters(text)

    return text