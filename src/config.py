import os
import json
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (API keys and model names) from .env file
load_dotenv()


class Config:
    """Centralized configuration for the Advanced Enterprise RAG Pipeline."""

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # If the environment variable is not available,
    # try to read the key from Streamlit Secrets.
    if not GROQ_API_KEY:
        try:
            GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        except Exception:
            GROQ_API_KEY = None

    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY not found. API calls will fail.")

    # LLM Settings (Fetched from .env, with fallbacks to active 2026 Groq models)
    FAST_LLM = os.getenv("FAST_LLM", "openai/gpt-oss-20b")
    REASONING_LLM = os.getenv("REASONING_LLM", "openai/gpt-oss-120b")

    # Vector Database Settings
    VECTOR_DB_PATH = "./data/vector_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, local open-source embedding model

    # Graph Database Settings
    GRAPH_DB_PATH = "./data/graph_db"

    # File Paths
    RAW_JSON_PATH = "./data/raw/bastian_knowledge.json"

    # --- DYNAMIC GUARDRAILS & THRESHOLDS ---
    # These are loaded directly from the JSON rulebook
    # to ensure strict adherence.

    @classmethod
    def load_dynamic_thresholds(cls):
        """Loads strict thresholds from the rulebook JSON."""
        try:
            with open(cls.RAW_JSON_PATH, "r", encoding="utf-8") as file:
                rulebook = json.load(file)

            retrieval_config = rulebook.get("retrieval_config", {})
            guardrails = rulebook.get("rag_guardrails", {})

            # Dynamically set class variables based on the JSON
            cls.TOP_K_RETRIEVAL = retrieval_config.get(
                "retrieval", {}
            ).get("top_k", 5)

            cls.SIMILARITY_THRESHOLD = retrieval_config.get(
                "retrieval", {}
            ).get("similarity_threshold", 0.35)

            cls.MIN_SUPPORTING_CHUNKS = guardrails.get(
                "minimum_supporting_chunks", 2
            )

            cls.CONFIDENCE_THRESHOLD = guardrails.get(
                "confidence_threshold", 0.75
            )

            cls.SUPPORTED_INTENTS = rulebook.get(
                "supported_intents", []
            )

            cls.NEGATIVE_KNOWLEDGE = rulebook.get(
                "negative_knowledge", []
            )

            print(
                "Configuration and Guardrails dynamically loaded "
                "from Rulebook JSON."
            )

        except FileNotFoundError:
            print(
                f"Error: Rulebook JSON not found at "
                f"{cls.RAW_JSON_PATH}. Using default thresholds."
            )

            cls.TOP_K_RETRIEVAL = 5
            cls.SIMILARITY_THRESHOLD = 0.30
            cls.MIN_SUPPORTING_CHUNKS = 1
            cls.CONFIDENCE_THRESHOLD = 0.50
            cls.SUPPORTED_INTENTS = []
            cls.NEGATIVE_KNOWLEDGE = []


# Initialize the dynamic thresholds when the module is imported
Config.load_dynamic_thresholds()