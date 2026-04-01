import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Supabase Configuration
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str
    
    # Groq Configuration
    GROQ_API_KEY: str
    GROQ_MODEL_NAME: str = "llama3-8b-8192" # Default to Llama 3 8B
    
    # OpenAI (Optional for embeddings if not using a free alternative)
    OPENAI_API_KEY: Optional[str] = None
    
    # App Settings
    APP_NAME: str = "University RAG Chatbot"
    DEBUG: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
