from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings
from app.llm.custom_shakespeare_llm import ShakespeareLLM
from typing import Optional, Any, AsyncGenerator, Generator
import os

class LLMManager:
    """Quản lý LLM - hỗ trợ cả API và local model với Streaming"""
    
    def __init__(self):
        self.llm = None
        self.llm_type = settings.LLM_TYPE
        
    def initialize(self):
        """Khởi tạo LLM dựa trên config"""
       
        if self.llm_type == "gemini":
            self.llm = self._init_gemini()
        elif self.llm_type == "shakespeare":
            self.llm = self._init_shakespeare()
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")
        
        return self.llm

    def _init_gemini(self):
        """Khởi tạo Google Gemini"""
        os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
        
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            google_api_key=settings.GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            streaming=True  # Enable streaming
        )
    
    def _init_shakespeare(self):
        """Khởi tạo Shakespeare custom model (Local)"""
        return ShakespeareLLM(
            model_name=settings.SHAKESPEARE_MODEL_NAME,
            max_new_tokens=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P
        )
    
    # =========================================================================
    # NON-STREAMING METHODS (Original)
    # =========================================================================
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Standard generation (dùng cho Gemini/OpenAI) - Non-streaming"""
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
    
    def generate_shakespeare_from_string(
        self, 
        prompt: str, 
        print_stream: bool = True
    ) -> tuple[str, dict]:
        """Generate Shakespeare từ prompt - Non-streaming"""
        if not self.llm:
            self.initialize()
        
        if not isinstance(self.llm, ShakespeareLLM):
            raise ValueError(f"Current LLM is {type(self.llm)}, not ShakespeareLLM")
        
        continuation, stats = self.llm.generate_with_stats(
            prompt, 
            print_stream=print_stream
        )
        
        return continuation, stats
    
    # =========================================================================
    # STREAMING METHODS (New)
    # =========================================================================
    
    async def stream_generate(
        self, 
        prompt: str, 
        temperature: float = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream generation cho Gemini/OpenAI
        Yields: text chunks
        """
        if not self.llm:
            self.initialize()
        
        try:
            # Override temperature if provided
            temp = temperature if temperature is not None else settings.TEMPERATURE
            
            if hasattr(self.llm, 'astream'):
                # LangChain async streaming
                async for chunk in self.llm.astream(prompt):
                    if hasattr(chunk, 'content'):
                        yield chunk.content
                    else:
                        yield str(chunk)
            
            elif hasattr(self.llm, 'stream'):
                # LangChain sync streaming (wrap in async)
                for chunk in self.llm.stream(prompt):
                    if hasattr(chunk, 'content'):
                        yield chunk.content
                    else:
                        yield str(chunk)
            else:
                # Fallback: no streaming support, return full text
                result = self.generate(prompt, temperature=temp, **kwargs)
                yield result
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def stream_shakespeare(
        self, 
        prompt: str
    ) -> Generator[str, None, None]:
        """
        Stream generation cho Shakespeare model (sync generator)
        Yields: text chunks
        """
        if not self.llm:
            self.initialize()
        
        if not isinstance(self.llm, ShakespeareLLM):
            raise ValueError(f"Current LLM is {type(self.llm)}, not ShakespeareLLM")
        
        # Stream từ Shakespeare model
        for chunk in self.llm.stream_generate(prompt):
            yield chunk
    
    async def astream_shakespeare(
        self, 
        prompt: str
    ) -> AsyncGenerator[str, None]:
        """
        Async wrapper cho stream_shakespeare
        """
        if not self.llm:
            self.initialize()
        
        if not isinstance(self.llm, ShakespeareLLM):
            raise ValueError(f"Current LLM is {type(self.llm)}, not ShakespeareLLM")
        
        # Wrap sync generator in async
        for chunk in self.llm.stream_generate(prompt):
            yield chunk

# Singleton instance
llm_manager = LLMManager()