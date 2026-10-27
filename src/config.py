"""Configuración común para Jupyter local, Google Colab y Streamlit.

El mismo código funciona en los tres entornos: lo único que cambia es de
dónde sale la API key, y eso se detecta automáticamente.
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "souls_lore"

# Modelos de Gemini (se pueden sobrescribir con variables de entorno).
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash")
# La reescritura de la consulta es una tarea sencilla: un modelo "lite" basta
# y no consume la cuota diaria (muy limitada en el plan gratuito) del modelo principal.
REWRITE_MODEL = os.getenv("GEMINI_REWRITE_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# Juegos indexados; "general" agrupa artículos de toda la saga.
GAMES = ("dark_souls", "bloodborne", "elden_ring", "general")

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def load_api_key() -> str:
    """Carga GOOGLE_API_KEY según el entorno.

    - Colab: panel de Secrets (icono de la llave) con el nombre GOOGLE_API_KEY.
    - Streamlit Cloud: st.secrets["GOOGLE_API_KEY"].
    - Local: archivo .env en la raíz del proyecto.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        if IN_COLAB:
            from google.colab import userdata
            os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
        else:
            from dotenv import load_dotenv
            load_dotenv(ROOT_DIR / ".env")

    if not os.getenv("GOOGLE_API_KEY"):
        try:
            import streamlit as st
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            pass

    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No se encontró GOOGLE_API_KEY. Crea un archivo .env (local), "
            "un Secret en Colab o un secret en Streamlit Cloud."
        )
    return key


def get_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    load_api_key()
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm(temperature: float | None = 0.3, model: str = CHAT_MODEL):
    """temperature=None usa el valor del modelo (los modelos "lite" no admiten cambiarlo)."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    load_api_key()
    kwargs = {} if temperature is None else {"temperature": temperature}
    return ChatGoogleGenerativeAI(model=model, **kwargs)


def get_vectorstore():
    from langchain_chroma import Chroma
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
