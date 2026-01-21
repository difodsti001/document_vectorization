from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv

# Cargamos el .env explícitamente para mayor seguridad
load_dotenv()

class Settings(BaseSettings):
    # ================= API =================
    API_TITLE: str = "Sistema de Vectorización Documental"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8100
    DEBUG: bool = True

    # ================= Storage =================
    UPLOAD_DIR: Path = Path("./uploads")
    MAX_FILE_SIZE: int = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx"]

    # ================= Qdrant =================
    QDRANT_HOST: str = "91.99.108.245"
    QDRANT_PORT: int = 6333

    # ================= API Externa =================
    COLLECTIONS_API_URL: str = ""

    # ================= Embeddings =================
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768

    # ================= Text Processing =================
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

settings = Settings()

def validate_settings():
    """Validaciones al iniciar la app"""
    settings.UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

    valid_models = {
        "BAAI/bge-base-en-v1.5",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/all-mpnet-base-v2",
    }

    # Verifica si el modelo actual contiene alguna de las cadenas válidas
    if not any(m in settings.EMBEDDING_MODEL for m in valid_models):
        print(f"⚠️ Modelo no validado en lista oficial: {settings.EMBEDDING_MODEL}")

    print("--- Configuración Cargada ---")
    print(f"🧠 Modelo: {settings.EMBEDDING_MODEL} (Dim: {settings.EMBEDDING_DIMENSION})")
    print(f"🗄️ Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    print(f"🔗 API Colecciones: {settings.COLLECTIONS_API_URL}")
    print("-----------------------------")
