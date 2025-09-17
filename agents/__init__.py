"""
agents package initializer.

This file makes it easy to import agents like:

    from agents import DataCollectionAgent
    from agents import AIAnalysisEngine
    from agents import CompeteAIOrchestrator

It attempts to load environment variables from a `.env` file (if
`python-dotenv` is installed). If not, it logs a warning and continues.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Try to load .env (optional dependency: python-dotenv)
# -------------------------------------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
    logger.info("Loaded environment variables from .env")
except Exception as e:  # ModuleNotFoundError or any load error
    logger.warning(
        "python-dotenv not available or failed to load .env (%s). "
        "Env vars must be provided by your shell/host.",
        e,
    )

# -------------------------------------------------------------------
# Re-export key classes so callers can do `from agents import ...`
# Each import is wrapped to keep the app running even if one module
# fails (the UI can show 'limited mode' and a helpful error).
# -------------------------------------------------------------------

# Data collection
try:
    from .data_collection_agent import DataCollectionAgent
except Exception as e:
    logger.exception("Failed to import DataCollectionAgent: %s", e)
    DataCollectionAgent = None  # type: ignore

# Analysis engine
try:
    from .analysis_engine import (
        AIAnalysisEngine,
        AnalysisConfig,
        AnalysisResult,
        create_analysis_engine,
    )
except Exception as e:
    logger.exception("Failed to import analysis engine: %s", e)
    AIAnalysisEngine = None  # type: ignore
    AnalysisConfig = None  # type: ignore
    AnalysisResult = None  # type: ignore
    create_analysis_engine = None  # type: ignore

# Orchestrator
try:
    from .orchestrator import CompeteAIOrchestrator
except Exception as e:
    logger.exception("Failed to import orchestrator: %s", e)
    CompeteAIOrchestrator = None  # type: ignore

__all__ = [
    "DataCollectionAgent",
    "AIAnalysisEngine",
    "AnalysisConfig",
    "AnalysisResult",
    "create_analysis_engine",
    "CompeteAIOrchestrator",
]
