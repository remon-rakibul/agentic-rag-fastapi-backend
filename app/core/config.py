"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    DATABASE_URL: str
    VECTOR_STORE_TABLE_NAME: str = "data"
    VECTOR_SIZE: int = 1536  # OpenAI embeddings dimension
    
    # OpenAI
    OPENAI_API_KEY: str
    
    # JWT Authentication
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    def __init__(self, **kwargs):
        """Initialize settings with validation."""
        super().__init__(**kwargs)
        # Warn if using default secret key in production
        if self.SECRET_KEY == "your-secret-key-change-in-production" and not self.DEBUG:
            import warnings
            warnings.warn(
                "SECRET_KEY is set to default value. This is insecure for production! "
                "Please set a strong SECRET_KEY in your environment variables.",
                UserWarning
            )
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "/tmp/rag_uploads"
    
    # Application
    APP_NAME: str = "RAG Pipeline API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # User Agent
    USER_AGENT: Optional[str] = None
    
    class Config:
        # Look for .env file: first in current directory, then in parent (refactored/)
        # config.py is at: refactored/app/core/config.py
        # So parent.parent.parent = refactored/
        _config_file = Path(__file__).resolve()
        _project_root = _config_file.parent.parent.parent  # refactored/
        _env_in_project = _project_root / ".env"
        _env_in_cwd = Path(".env")
        
        # Use absolute path to .env file if it exists in project root
        if _env_in_project.exists():
            env_file = str(_env_in_project)
        elif _env_in_cwd.exists():
            env_file = str(_env_in_cwd.resolve())
        else:
            env_file = ".env"  # Fallback - pydantic will look in CWD
        
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()

