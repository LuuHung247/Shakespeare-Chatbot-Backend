from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from app.config import settings
from typing import Optional
import torch
import os

class LLMManager:
    """Quản lý LLM - hỗ trợ cả API và local model"""
    
    def __init__(self):
        self.llm = None
        self.llm_type = settings.LLM_TYPE
        
    def initialize(self):
        """Khởi tạo LLM dựa trên config"""
        if self.llm_type == "openai":
            self.llm = self._init_openai()
        elif self.llm_type == "anthropic":
            self.llm = self._init_anthropic()
        elif self.llm_type == "gemini":
            self.llm = self._init_gemini()
        elif self.llm_type == "local":
            self.llm = self._init_local()
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")
        
        return self.llm
    
    def _init_openai(self):
        """Khởi tạo OpenAI"""
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            openai_api_key=settings.OPENAI_API_KEY
        )
    
    def _init_anthropic(self):
        """Khởi tạo Anthropic Claude"""
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            anthropic_api_key=settings.ANTHROPIC_API_KEY
        )
    
    def _init_gemini(self):
        """Khởi tạo Google Gemini"""
        # Set API key
        os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
        
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            google_api_key=settings.GOOGLE_API_KEY,
            convert_system_message_to_human=True  # Gemini không support system message
        )
    
    def _init_local(self):
        """Khởi tạo local model (GPT-2 hoặc fine-tuned model)"""
        try:
            # Load tokenizer và model
            tokenizer = AutoTokenizer.from_pretrained(
                settings.LOCAL_MODEL_PATH or settings.LOCAL_MODEL_NAME
            )
            model = AutoModelForCausalLM.from_pretrained(
                settings.LOCAL_MODEL_PATH or settings.LOCAL_MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Tạo pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                do_sample=True
            )
            
            # Wrap với LangChain
            return HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            raise Exception(f"Error loading local model: {str(e)}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text từ prompt"""
        if not self.llm:
            self.initialize()
        
        try:
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            else:
                return self.llm(prompt)
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

# Singleton instance
llm_manager = LLMManager()