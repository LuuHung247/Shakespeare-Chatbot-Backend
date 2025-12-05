from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Shakespeare Poem Chatbot"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # LLM Settings
   
   
    LLM_TYPE: Literal["gemini", "shakespeare"] = "gemini"
    # Google Gemini
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-flash"
    

    # Shakespeare Custom Model
    SHAKESPEARE_MODEL_NAME: str = "Hancovirus/shakespear_llama-3.2-3B-Instruct"
    # Alternative: "Hancovirus/shakespear_llama-3.2-3B-Instruct"
    
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