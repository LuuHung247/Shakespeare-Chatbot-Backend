from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Shakespeare Poem Chatbot"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # LLM Settings
    LLM_TYPE: Literal["gemini", "local"] = "gemini"
    

    # Google Gemini
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-flash"  # hoặc "gemini-1.5-pro", "gemini-1.5-flash"
    
    # Local Model
    LOCAL_MODEL_PATH: str = "models/shakespeare-model"
    LOCAL_MODEL_NAME: str = "gpt2"  # hoặc model đã fine-tune
    

    
    # Generation Settings
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.8
    TOP_P: float = 0.9
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()