"""
Unit tests for the Semantic Similarity Engine.
"""

from app.services.semantic_similarity import SemanticSimilarityEngine


def test_identical_documents():
    """
    Identical texts should have a very high similarity.
    """

    engine = SemanticSimilarityEngine()

    score = engine.calculate_similarity(
        "Python SQL Machine Learning",
        "Python SQL Machine Learning",
    )

    assert score > 95


def test_related_documents():
    """
    Related texts should have a moderate to high similarity.
    """

    engine = SemanticSimilarityEngine()

    score = engine.calculate_similarity(
        "Python FastAPI Docker Machine Learning",
        "Looking for a Machine Learning Engineer with Python and Docker experience",
    )

    assert score > 60


def test_different_documents():
    """
    Completely unrelated documents should have lower similarity.
    """

    engine = SemanticSimilarityEngine()

    score = engine.calculate_similarity(
        "Civil Engineering AutoCAD Concrete Design",
        "Python NLP Transformers FastAPI",
    )

    assert score < 50


def test_empty_documents():
    """
    Empty input should return 0 similarity.
    """

    engine = SemanticSimilarityEngine()

    score = engine.calculate_similarity("", "")

    assert score == 0.0


def test_one_empty_document():
    """
    One empty document should also return 0 similarity.
    """

    engine = SemanticSimilarityEngine()

    score = engine.calculate_similarity(
        "Python SQL",
        "",
    )

    assert score == 0.0


def test_structured_ai_documents_have_reasonable_similarity():
    engine = SemanticSimilarityEngine()

    score = engine.calculate_similarity(
        "Noisy contact line only",
        "Noisy job intro",
        resume_data={
            "skills": ["Python", "Machine Learning", "FastAPI", "Docker"],
            "projects": ["NLP transformer project for recruiter screening"],
            "experience": ["AI Engineering Intern"],
            "education": ["BS Data Science"],
        },
        job_data={
            "job_title": "AI Engineer Intern",
            "required_skills": ["Python", "Machine Learning", "FastAPI", "Docker"],
            "responsibilities": ["Build AI APIs and evaluate NLP models"],
            "qualifications": ["Data Science or AI background"],
        },
    )

    assert score > 50


def test_ai_ml_resume_vs_ai_ml_job_scores_above_30():
    engine = SemanticSimilarityEngine()

    resume_text = """
    Python Machine Learning NLP PyTorch TensorFlow scikit-learn Pandas.
    Built FastAPI AI pipelines and model evaluation dashboards.
    Projects include transformer classification and data science workflows.
    """
    job_text = """
    AI Data Science role requiring Python, Machine Learning, Deep Learning,
    Natural Language Processing, API Development, model evaluation, and data pipelines.
    """

    score = engine.calculate_similarity(resume_text, job_text)

    assert score > 30


def test_unrelated_resume_scores_lower_than_related_ai_resume():
    engine = SemanticSimilarityEngine()

    ai_job = """
    AI Engineer role requiring Python, Machine Learning, NLP, FastAPI,
    Docker, TensorFlow, PyTorch, and model deployment.
    """
    related_resume = """
    Python Machine Learning NLP PyTorch FastAPI project for AI model deployment.
    """
    unrelated_resume = """
    Civil engineering AutoCAD concrete materials site inspection and structural drawings.
    """

    related_score = engine.calculate_similarity(related_resume, ai_job)
    unrelated_score = engine.calculate_similarity(unrelated_resume, ai_job)

    assert unrelated_score < related_score


def test_nlp_and_natural_language_processing_are_strongly_related():
    engine = SemanticSimilarityEngine()

    details = engine.calculate_similarity_details(
        "Built NLP pipelines and transformer experiments.",
        "Requires Natural Language Processing and text analytics.",
        resume_data={"skills": ["NLP"]},
        job_data={"required_skills": ["Natural Language Processing"]},
    )

    assert details["skill_semantic_score"] >= 80
    assert details["semantic_score"] > 40
    assert set(details) >= {
        "skill_semantic_score",
        "section_semantic_score",
        "chunk_semantic_score",
        "semantic_score",
    }
