from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.llms import HuggingFacePipeline # Có thể bỏ nếu không dùng pipeline của HF trực tiếp
from app.config import settings
# Chỉ import class ShakespeareLLM, bỏ build_shakespeare_prompt
from app.llm.custom_shakespeare_llm import ShakespeareLLM 
from typing import Optional, Any
import os

class LLMManager:
    """Quản lý LLM - hỗ trợ cả API và local model"""
    
    def __init__(self):
        self.llm = None
        self.llm_type = settings.LLM_TYPE
        
    def initialize(self):
        """Khởi tạo LLM dựa trên config"""
       
        if self.llm_type == "gemini":
            self.llm = self._init_gemini()
        # elif self.llm_type == "local":
        #     self.llm = self._init_local()
        elif self.llm_type == "shakespeare":
            self.llm = self._init_shakespeare()
        else:
            # Fallback hoặc raise error tùy logic
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
            convert_system_message_to_human=True
        )
    
    def _init_shakespeare(self):
        """Khởi tạo Shakespeare custom model (Local)"""
        # Đảm bảo settings có các biến này
        return ShakespeareLLM(
            model_name=settings.SHAKESPEARE_MODEL_NAME, # "Hancovirus/shakespear_Qwen2.5-3B-Instruct"
            max_new_tokens=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Standard generation (dùng cho Gemini/OpenAI).
        Trả về string nội dung.
        """
        if not self.llm:
            self.initialize()
        
        try:
            # LangChain models thường dùng invoke
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
        """
        Generate Shakespeare từ prompt string có sẵn.
        Dành riêng cho luồng Local Model để lấy stats và streaming.
        
        Args:
            prompt: Prompt string hoàn chỉnh (đã build bên ChatService)
            print_stream: In ra console khi streaming (để debug)
            
        Returns:
            tuple: (continuation_text, stats_dict)
        """
        if not self.llm:
            self.initialize()
        
        # Kiểm tra đúng loại model không
        if not isinstance(self.llm, ShakespeareLLM):
            # Nếu lỡ config sai mà gọi hàm này, có thể fallback hoặc raise error
            raise ValueError(f"Current LLM is {type(self.llm)}, not ShakespeareLLM")
        
        # Gọi trực tiếp method của ShakespeareLLM
        continuation, stats = self.llm.generate_with_stats(
            prompt, 
            print_stream=print_stream
        )
        
        return continuation, stats

# Singleton instance
llm_manager = LLMManager()