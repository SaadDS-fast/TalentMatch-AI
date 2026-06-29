"""
Configuration settings for TalentMatch AI.

This file keeps environment variables, model names, and project paths
centralized so the rest of the codebase remains clean and maintainable.
"""

from pathlib import Path
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()


# ==========================
# Base Directories
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"


# ==========================
# Sample Data Directories
# ==========================

SAMPLE_RESUMES_DIR = DATA_DIR / "sample_resumes"
SAMPLE_JOB_DESCRIPTIONS_DIR = DATA_DIR / "sample_job_descriptions"


# ==========================
# API Keys
# ==========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ==========================
# Model Settings
# ==========================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")


# ==========================
# Scoring Weights
# ==========================

SKILL_MATCH_WEIGHT = 0.45
SEMANTIC_SIMILARITY_WEIGHT = 0.30
EXPERIENCE_MATCH_WEIGHT = 0.15
EDUCATION_MATCH_WEIGHT = 0.10


# ==========================
# Application Settings
# ==========================

APP_NAME = "TalentMatch AI"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"


# ==========================
# Ensure Required Directories Exist
# ==========================

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RESUMES_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_JOB_DESCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)