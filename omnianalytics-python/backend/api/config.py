from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "OmniAnalytics"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql+asyncpg://postgres:password@localhost:5432/omnianalytics"
    )
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # Allowed Hosts
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # External APIs
    SALESFORCE_CLIENT_ID: Optional[str] = os.getenv("SALESFORCE_CLIENT_ID")
    SALESFORCE_CLIENT_SECRET: Optional[str] = os.getenv("SALESFORCE_CLIENT_SECRET")
    
    # File upload
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "uploads"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()