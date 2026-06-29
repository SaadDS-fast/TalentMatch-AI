"""
Semantic similarity service for TalentMatch AI.

Hybrid scoring combines:
- normalized skill overlap and skill-relatedness
- focused section similarity
- chunk-level top-k similarity
"""

from functools import lru_cache
import logging
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import EMBEDDING_MODEL_NAME
from app.services.skill_extractor import SkillExtractor
from app.services.skill_normalizer import normalize_skill_name
from app.utils.text_cleaner import clean_text


logger = logging.getLogger(__name__)


class SemanticSimilarityEngine:
    """
    Calculates semantic similarity between a resume and a job description.
    """

    RELATED_SKILL_GROUPS = [
        {
            "nlp",
            "naturallanguageprocessing",
            "transformers",
            "llm",
            "generativeai",
            "rag",
            "embeddings",
        },
        {
            "pytorch",
            "tensorflow",
            "keras",
            "deeplearning",
            "machinelearning",
            "computervision",
        },
        {
            "fastapi",
            "restapi",
            "apidevelopment",
            "flask",
            "django",
        },
        {
            "docker",
            "aws",
            "azure",
            "gcp",
            "ci/cd",
            "git",
            "github",
        },
        {
            "pandas",
            "numpy",
            "scikitlearn",
            "statistics",
            "datacleaning",
            "featureengineering",
            "modelevaluation",
        },
        {
            "sql",
            "mysql",
            "postgresql",
            "sqlite",
            "mongodb",
        },
        {
            "vectordatabase",
            "chromadb",
            "faiss",
            "embeddings",
            "rag",
        },
    ]

    def __init__(self):
        self.model = self._load_model()
        self.skill_extractor = SkillExtractor()

    def calculate_similarity(
        self,
        resume_text: str,
        job_text: str,
        resume_data: dict | None = None,
        job_data: dict | None = None,
    ) -> float:
        """
        Return final semantic score from 0 to 100.
        """

        return self.calculate_similarity_details(
            resume_text=resume_text,
            job_text=job_text,
            resume_data=resume_data,
            job_data=job_data,
        )["semantic_score"]

    def calculate_similarity_details(
        self,
        resume_text: str,
        job_text: str,
        resume_data: dict | None = None,
        job_data: dict | None = None,
    ) -> dict:
        """
        Return component scores for debugging/explainability.
        """

        resume_profile_text = self._build_resume_profile_text(resume_text, resume_data)
        job_requirement_text = self._build_job_requirement_text(job_text, job_data)

        if len(resume_profile_text.split()) < 3 or len(job_requirement_text.split()) < 3:
            return {
                "skill_semantic_score": 0.0,
                "section_semantic_score": 0.0,
                "chunk_semantic_score": 0.0,
                "semantic_score": 0.0,
                "explanation": "Empty or low-quality semantic input.",
            }

        resume_skills = self._skills_from_sources(resume_text, resume_data, "resume")
        job_skills = self._skills_from_sources(job_text, job_data, "job")

        skill_score = self._skill_semantic_score(resume_skills, job_skills)
        section_score = self._section_semantic_score(resume_profile_text, job_requirement_text)
        chunk_score = self._chunk_semantic_score(resume_profile_text, job_requirement_text)

        final_score = (
            (0.50 * skill_score)
            + (0.30 * section_score)
            + (0.20 * chunk_score)
        )

        details = {
            "skill_semantic_score": round(skill_score, 2),
            "section_semantic_score": round(section_score, 2),
            "chunk_semantic_score": round(chunk_score, 2),
            "semantic_score": round(min(max(final_score, 0.0), 100.0), 2),
            "explanation": "Hybrid semantic score from skills, focused sections, and top-k chunks.",
        }
        logger.debug("Semantic similarity details: %s", details)
        return details

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model():
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            logger.info(
                "SentenceTransformer package unavailable; using fallback similarity: %s",
                exc,
            )
            return None

        try:
            return SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                local_files_only=True,
            )
        except TypeError as exc:
            logger.info(
                "Installed SentenceTransformer does not support local-only loading; using fallback similarity: %s",
                exc,
            )
            return None
        except Exception as exc:
            logger.info(
                "SentenceTransformer model unavailable locally; using fallback similarity: %s",
                exc,
            )
            return None

    def _skills_from_sources(
        self,
        raw_text: str,
        structured_data: dict | None,
        document_type: str,
    ) -> list[str]:
        field_names = (
            ["skills"]
            if document_type == "resume"
            else ["required_skills", "preferred_skills"]
        )

        skills = []
        if structured_data:
            for field_name in field_names:
                value = structured_data.get(field_name, [])
                if isinstance(value, list):
                    skills.extend(str(item) for item in value if item)

        if not skills:
            skills = self.skill_extractor.extract_skills(raw_text)

        return sorted(set(skills))

    def _skill_semantic_score(
        self,
        resume_skills: list[str],
        job_skills: list[str],
    ) -> float:
        if not resume_skills or not job_skills:
            return 0.0

        resume_normalized = {
            normalize_skill_name(skill): skill
            for skill in resume_skills
        }
        job_normalized = {
            normalize_skill_name(skill): skill
            for skill in job_skills
        }

        job_scores = []
        for job_key, job_skill in job_normalized.items():
            if job_key in resume_normalized:
                job_scores.append(100.0)
                continue

            related_score = max(
                (
                    self._related_skill_score(resume_key, job_key)
                    for resume_key in resume_normalized
                ),
                default=0.0,
            )

            embedding_score = 0.0
            if related_score < 80:
                embedding_score = max(
                    (
                        self._text_similarity(resume_skill, job_skill)
                        for resume_skill in resume_normalized.values()
                    ),
                    default=0.0,
                )

            job_scores.append(max(related_score, embedding_score))

        return sum(job_scores) / len(job_scores)

    def _section_semantic_score(self, resume_text: str, job_text: str) -> float:
        return self._text_similarity(resume_text, job_text)

    def _chunk_semantic_score(self, resume_text: str, job_text: str) -> float:
        resume_chunks = self._meaningful_chunks(resume_text)
        job_chunks = self._meaningful_chunks(job_text)

        if not resume_chunks or not job_chunks:
            return self._text_similarity(resume_text, job_text)

        if self.model is not None:
            try:
                texts = resume_chunks + job_chunks
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                resume_embeddings = embeddings[: len(resume_chunks)]
                job_embeddings = embeddings[len(resume_chunks):]
                matrix = cosine_similarity(resume_embeddings, job_embeddings) * 100
                return self._top_k_average(matrix.flatten().tolist())
            except Exception as exc:
                logger.warning("Chunk embedding similarity failed; using fallback: %s", exc)

        pair_scores = [
            self._fallback_text_similarity(resume_chunk, job_chunk)
            for resume_chunk in resume_chunks
            for job_chunk in job_chunks
        ]
        return self._top_k_average(pair_scores)

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        text_a = self._prepare_text(text_a)
        text_b = self._prepare_text(text_b)

        if len(text_a.split()) < 1 or len(text_b.split()) < 1:
            return 0.0

        if self.model is not None:
            try:
                embeddings = self.model.encode([text_a, text_b], convert_to_numpy=True)
                return round(float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]) * 100, 2)
            except Exception as exc:
                logger.warning("Embedding text similarity failed; using fallback: %s", exc)

        return self._fallback_text_similarity(text_a, text_b)

    @staticmethod
    def _fallback_text_similarity(text_a: str, text_b: str) -> float:
        if text_a == text_b:
            return 100.0

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 3),
            min_df=1,
        )

        try:
            matrix = vectorizer.fit_transform([text_a, text_b])
            tfidf_score = float(cosine_similarity(matrix[0], matrix[1])[0][0]) * 100
        except ValueError:
            tfidf_score = 0.0

        tokens_a = SemanticSimilarityEngine._tokens(text_a)
        tokens_b = SemanticSimilarityEngine._tokens(text_b)

        if not tokens_a or not tokens_b:
            coverage_score = 0.0
        else:
            overlap = len(tokens_a & tokens_b)
            min_coverage = overlap / min(len(tokens_a), len(tokens_b))
            harmonic_coverage = (2 * overlap) / (len(tokens_a) + len(tokens_b))
            coverage_score = ((min_coverage * 0.65) + (harmonic_coverage * 0.35)) * 100

        score = (tfidf_score * 0.25) + (coverage_score * 0.75)
        return round(min(max(score, 0.0), 100.0), 2)

    @classmethod
    def _related_skill_score(cls, resume_skill: str, job_skill: str) -> float:
        for group in cls.RELATED_SKILL_GROUPS:
            if resume_skill in group and job_skill in group:
                return 85.0
        return 0.0

    @staticmethod
    def _top_k_average(scores: list[float], top_k: int = 5) -> float:
        if not scores:
            return 0.0

        selected_scores = sorted(scores, reverse=True)[: min(top_k, len(scores))]
        return round(sum(selected_scores) / len(selected_scores), 2)

    @staticmethod
    def _build_resume_profile_text(
        raw_text: str,
        resume_data: dict | None = None,
    ) -> str:
        if not resume_data:
            return SemanticSimilarityEngine._prepare_text(raw_text)

        sections = []
        field_labels = [
            ("Professional Summary", ["candidate_name"]),
            ("Technical Skills", ["skills"]),
            ("Projects", ["projects"]),
            ("Experience", ["experience"]),
            ("Education", ["education"]),
            ("Certifications", ["certifications"]),
        ]

        for label, fields in field_labels:
            values = SemanticSimilarityEngine._collect_structured_values(resume_data, fields)
            if values:
                sections.append(f"{label}: " + "; ".join(values))

        focused_text = "\n".join(sections)
        if len(focused_text.split()) < 20:
            focused_text = f"{focused_text}\n{raw_text}"

        return SemanticSimilarityEngine._prepare_text(focused_text)

    @staticmethod
    def _build_job_requirement_text(
        raw_text: str,
        job_data: dict | None = None,
    ) -> str:
        if not job_data:
            return SemanticSimilarityEngine._prepare_text(raw_text)

        sections = []
        field_labels = [
            ("Job Title", ["job_title"]),
            ("Required Skills", ["required_skills"]),
            ("Preferred Skills", ["preferred_skills"]),
            ("Responsibilities", ["responsibilities"]),
            ("Qualifications", ["qualifications"]),
            ("Experience Required", ["experience_required"]),
        ]

        for label, fields in field_labels:
            values = SemanticSimilarityEngine._collect_structured_values(job_data, fields)
            if values:
                sections.append(f"{label}: " + "; ".join(values))

        focused_text = "\n".join(sections)
        if len(focused_text.split()) < 20:
            focused_text = f"{focused_text}\n{raw_text}"

        return SemanticSimilarityEngine._prepare_text(focused_text)

    @staticmethod
    def _collect_structured_values(data: dict, fields: list[str]) -> list[str]:
        values = []

        for field in fields:
            value = data.get(field)
            if isinstance(value, list):
                values.extend(str(item) for item in value if item)
            elif value:
                values.append(str(value))

        return values

    @staticmethod
    def _prepare_text(text: str) -> str:
        text = clean_text(text)
        text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"(?m)^\s*(email|phone|linkedin|github)\s*:\s*.*$", " ", text, flags=re.I)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _meaningful_chunks(text: str) -> list[str]:
        raw_lines = [
            line.strip(" -•\t")
            for line in re.split(r"[\n\r]+|(?<=\.)\s+", text)
            if line.strip(" -•\t")
        ]

        chunks = []
        current_chunk = []

        for line in raw_lines:
            if SemanticSimilarityEngine._line_has_signal(line) or len(line.split()) >= 4:
                current_chunk.append(line)

            if len(" ".join(current_chunk).split()) >= 70:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        if not chunks:
            words = text.split()
            chunks = [
                " ".join(words[index:index + 80])
                for index in range(0, len(words), 60)
                if words[index:index + 80]
            ]

        return [chunk for chunk in chunks if len(chunk.split()) >= 3][:14]

    @staticmethod
    def _line_has_signal(line: str) -> bool:
        signal_terms = (
            "python",
            "machine learning",
            "deep learning",
            "fastapi",
            "api",
            "docker",
            "sql",
            "numpy",
            "pandas",
            "tensorflow",
            "pytorch",
            "transformer",
            "nlp",
            "natural language processing",
            "llm",
            "rag",
            "project",
            "model",
            "classification",
            "optimization",
            "summarization",
            "requirements",
            "responsibilities",
        )
        lowered = line.lower()
        return any(term in lowered for term in signal_terms)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        ignored_tokens = {
            "professional",
            "summary",
            "technical",
            "skills",
            "projects",
            "experience",
            "education",
            "certifications",
            "required",
            "preferred",
            "responsibilities",
            "qualifications",
            "title",
            "job",
        }

        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+#/.-]*", text.lower())
            if len(token) > 1 and token not in ignored_tokens
        }
