"""
Shared skill normalization utilities.

The matching and extraction layers both use these helpers so variants such as
"Fast API", "fastapi", and "FastAPI" resolve to the same canonical skill key.
"""

import re


ALIASES = {
    "fastapi": "fastapi",
    "fastapi": "fastapi",
    "numpy": "numpy",
    "nump y": "numpy",
    "pandas": "pandas",
    "scikitlearn": "scikitlearn",
    "sklearn": "scikitlearn",
    "machinelearning": "machinelearning",
    "naturallanguageprocessing": "naturallanguageprocessing",
    "nlp": "naturallanguageprocessing",
    "largelanguagemodel": "llm",
    "largelanguagemodels": "llm",
    "llm": "llm",
    "vectordb": "vectordatabase",
    "vectordatabase": "vectordatabase",
    "vectorstore": "vectordatabase",
    "restapi": "restapi",
    "openaiapi": "openaiapi",
    "geminiapi": "geminiapi",
    "apidevelopment": "apidevelopment",
}


EXTRACTION_PATTERNS = {
    "fastapi": r"fast[\s_-]*api",
    "numpy": r"num[\s_-]*py|numpy",
    "pandas": r"pandas",
    "scikit-learn": r"scikit[\s_-]*learn|sklearn",
    "machine learning": r"machine[\s_-]+learning",
    "natural language processing": r"natural[\s_-]+language[\s_-]+processing|nlp",
    "llm": r"llm|large[\s_-]+language[\s_-]+models?",
    "vector database": r"vector[\s_-]*(database|db|store)",
    "rest api": r"rest[\s_-]*api",
    "openai api": r"openai[\s_-]*api",
    "gemini api": r"gemini[\s_-]*api",
    "api development": r"api[\s_-]+development",
}


def normalize_skill_name(skill: str) -> str:
    """
    Normalize skill names for comparison.
    """

    normalized = skill.lower().strip()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[\s_-]+", "", normalized)
    normalized = normalized.replace(".", "")

    return ALIASES.get(normalized, normalized)


def skill_pattern(skill: str) -> str:
    """
    Return a boundary-aware extraction pattern for a canonical skill.
    """

    normalized_name = skill.lower()
    pattern = EXTRACTION_PATTERNS.get(normalized_name, re.escape(normalized_name))
    return rf"(?<![a-z0-9+#/]){pattern}(?![a-z0-9+#/])"
