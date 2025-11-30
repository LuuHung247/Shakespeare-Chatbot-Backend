from langchain.llms.base import LLM
from typing import Optional, List, Any, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
import time
import re

class ShakespeareLLM(LLM):
    """Custom LLM wrapper cho Shakespeare model với streaming và parsing support"""
    
    model_name: str = "Hancovirus/shakespear_Qwen2.5-3B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    tokenizer: Any = None
    model: Any = None
    
    class Config:
        arbitrary_types_allowed = True
        protected_namespaces = ()
    
    def __init__(self, **kwargs):
        # Extract custom params
        model_name = kwargs.pop('model_name', "Hancovirus/shakespear_Qwen2.5-3B-Instruct")
        max_new_tokens = kwargs.pop('max_new_tokens', 512)
        temperature = kwargs.pop('temperature', 0.7)
        top_p = kwargs.pop('top_p', 0.9)
        
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = None
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load model và tokenizer"""
        if self.tokenizer is None or self.model is None:
            print(f"Loading local Shakespeare model: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                )
                print(f"Model loaded successfully on: {self.model.device}")
            except Exception as e:
                print(f"CRITICAL ERROR loading model: {e}")
                raise e
    
    @property
    def _llm_type(self) -> str:
        return "shakespeare_custom"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> str:
        """Standard generation for LangChain compatibility"""
        continuation, _ = self._generate_internal(prompt, print_stream=False, stop_sequences=stop)
        return continuation
    
    def _generate_internal(
        self, 
        prompt: str, 
        print_stream: bool = False,
        stop_sequences: Optional[List[str]] = None
    ) -> tuple[str, dict]:
        """Core generation logic với streaming"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        num_input_tokens = inputs["input_ids"].shape[-1]
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            streamer=streamer,
        )
        
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        generated_parts = []
        start_time = time.time()
        
        for new_text in streamer:
            if print_stream:
                print(new_text, end="", flush=True)
            generated_parts.append(new_text)
            
            # Basic stop sequence checking (optional)
            if stop_sequences and any(s in new_text for s in stop_sequences):
                break
        
        thread.join(timeout=1.0)
        end_time = time.time()
        
        continuation = "".join(generated_parts)
        generation_time = end_time - start_time
        
        # Calculate tokens per second (approx)
        num_new_tokens = 0
        if continuation.strip():
            num_new_tokens = len(self.tokenizer.encode(continuation))
            
        stats = {
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_input_tokens + num_new_tokens,
            "num_new_tokens": num_new_tokens,
            "generation_time": generation_time,
            "tokens_per_second": num_new_tokens / generation_time if generation_time > 0 else 0
        }
        
        return continuation, stats
    
    def generate_with_stats(self, prompt: str, print_stream: bool = True) -> tuple[str, dict]:
        """Public method để gọi generate kèm stats"""
        return self._generate_internal(prompt, print_stream=print_stream)

    @staticmethod
    def parse_generated_text(raw_text: str) -> List[Dict[str, str]]:
        """
        Chuyển đổi raw text output của Shakespeare model thành 
        List[Dict] tương thích với ContinueSceneResponse.
        
        Input:
            [HAMLET] To be or not to be.
            {He sighs.}
            
        Output:
            [
                {"character": "HAMLET", "line": "To be or not to be."},
                {"character": "STAGE_DIRECTION", "line": "He sighs."}
            ]
        """
        dialogue_list = []
        
        # Tách dòng, loại bỏ dòng trống
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        for line in lines:
            # Case 1: Stage Direction {Action}
            # Regex bắt: bắt đầu bằng {, kết thúc bằng }, lấy nội dung bên trong
            stage_match = re.match(r'^\{(.*)\}$', line)
            if stage_match:
                content = stage_match.group(1).strip()
                dialogue_list.append({
                    "character": "STAGE_DIRECTION",
                    "line": content
                })
                continue
            
            # Case 2: Dialogue [CHARACTER] Line
            # Regex bắt: [TÊN] Nội dung
            # Sử dụng non-greedy matching .*? cho tên nhân vật
            dialogue_match = re.match(r'^\[(.*?)\]\s*(.*)$', line)
            if dialogue_match:
                char_name = dialogue_match.group(1).strip().upper()
                speech = dialogue_match.group(2).strip()
                
                # Bỏ qua nếu line rỗng
                if not speech:
                    continue
                    
                dialogue_list.append({
                    "character": char_name,
                    "line": speech
                })
                continue
            
            # Case 3: Dòng không đúng format (hiếm gặp với model instruct tốt)
            # Có thể bỏ qua hoặc gán vào nhân vật của dòng trước (nếu cần xử lý multiline phức tạp)
            # Ở đây ta chọn bỏ qua để đảm bảo clean data
            pass
            
        return dialogue_list