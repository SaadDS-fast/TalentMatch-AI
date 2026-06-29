"""
Main entry point for TalentMatch AI.
"""

from fastapi import FastAPI

from app.api.routes import router
from app.config import (
    APP_NAME,
    APP_VERSION,
)


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "AI-powered Resume-to-Job Fit Analysis System "
        "using NLP, semantic matching, and recruiter-style scoring."
    ),
)


# -----------------------------------------------------
# Register API Routes
# -----------------------------------------------------

app.include_router(router)


# -----------------------------------------------------
# Health Check
# -----------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """
    Root endpoint.
    """

    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "message": "Welcome to TalentMatch AI 🚀",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
    }