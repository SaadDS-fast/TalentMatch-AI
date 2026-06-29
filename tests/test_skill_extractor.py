"""
Unit tests for the Skill Extractor.
"""

from app.services.skill_extractor import SkillExtractor


def test_extract_single_skill():
    text = "The candidate has strong experience in Python."

    extractor = SkillExtractor()
    skills = extractor.extract_skills(text)

    assert "Python" in skills


def test_extract_multiple_skills():
    text = """
    We need someone with Python, SQL, FastAPI, Docker,
    and Machine Learning experience.
    """

    extractor = SkillExtractor()
    skills = extractor.extract_skills(text)

    assert "Python" in skills
    assert "SQL" in skills
    assert "FastAPI" in skills
    assert "Docker" in skills
    assert "Machine Learning" in skills


def test_no_skill_found():
    text = "The candidate is motivated and hardworking."

    extractor = SkillExtractor()
    skills = extractor.extract_skills(text)

    assert skills == []


def test_duplicate_skills_removed():
    text = "Python Python python SQL sql"

    extractor = SkillExtractor()
    skills = extractor.extract_skills(text)

    assert skills.count("Python") == 1
    assert skills.count("SQL") == 1


def test_extract_skill_spelling_variants():
    text = """
    Built a Fast API service with Num Py arrays, scikit learn models,
    NLP pipelines, large language model evaluation, and a vector db.
    """

    extractor = SkillExtractor()
    skills = extractor.extract_skills(text)

    assert "FastAPI" in skills
    assert "NumPy" in skills
    assert "scikit-learn" in skills
    assert "NLP" in skills
    assert "LLM" in skills
    assert "Vector Database" in skills
