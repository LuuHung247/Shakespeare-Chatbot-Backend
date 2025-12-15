from langchain.llms.base import LLM
from typing import Optional, List, Any, Dict, Generator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
import threading
import time
import re

class ShakespeareLLM(LLM):
    """
    Custom LLM wrapper cho Shakespeare model với streaming và parsing support.
    
    Features:
    - 4-bit quantization cho GPU nhỏ (4GB VRAM)
    - Streaming generation
    - Anti-repetition với repetition_penalty và no_repeat_ngram_size
    - Text parsing cho Shakespeare format
    """
    
    model_name: str = "Hancovirus/shakespear_Qwen2.5-3B-Instruct"
    max_new_tokens: int = 1024
    min_new_tokens: int = 100
    temperature: float = 0.6
    top_p: float = 0.9
    
    # Anti-repetition parameters
    repetition_penalty: float = 1.1  # >1.0 để phạt lặp token
    no_repeat_ngram_size: int = 8    # Cấm lặp n-gram size n
    
    tokenizer: Any = None
    model: Any = None
    
    class Config:
        arbitrary_types_allowed = True
        protected_namespaces = ()
    
    def __init__(self, **kwargs):
        # Extract custom params
        model_name = kwargs.pop('model_name', "Hancovirus/shakespear_Qwen2.5-3B-Instruct")
        max_new_tokens = kwargs.pop('max_new_tokens', 512)
        min_new_tokens = kwargs.pop('min_new_tokens', 100)
        temperature = kwargs.pop('temperature', 0.7)
        top_p = kwargs.pop('top_p', 0.9)
        repetition_penalty = kwargs.pop('repetition_penalty', 1.1)
        no_repeat_ngram_size = kwargs.pop('no_repeat_ngram_size', 4)

        super().__init__(**kwargs)

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.tokenizer = None
        self.model = None

        self._load_model()
    
    def _load_model(self):
        """Load model và tokenizer tối ưu cho GPU 4GB (GTX 3050 Ti)"""
        if self.tokenizer is None or self.model is None:
            print(f"Loading local Shakespeare model: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                )
                
                # Cấu hình lượng tử hóa 4-bit (NF4) để vừa VRAM 4GB
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config, 
                    device_map="auto",              
                    trust_remote_code=True,
                )
                print(f"Model loaded successfully on GPU (4-bit quantized)")
                print(f"  → repetition_penalty: {self.repetition_penalty}")
                print(f"  → no_repeat_ngram_size: {self.no_repeat_ngram_size}")
            
            except Exception as e:
                print(f"Warning: GPU loading failed ({e}). Falling back to CPU.")
                print("Note: CPU inference will be significantly slower.")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
                print(f"Model loaded on CPU")
    
    @property
    def _llm_type(self) -> str:
        return "shakespeare_custom"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
        }
    
    def _build_generation_kwargs(
        self,
        inputs: Dict[str, Any],
        streamer: Optional[TextIteratorStreamer] = None,
        **overrides
    ) -> Dict[str, Any]:
        """
        Build generation kwargs với tất cả parameters.
        Tránh duplicate code giữa các methods.
        """
        gen_kwargs = {
            **inputs,
            "max_new_tokens": overrides.get("max_new_tokens", self.max_new_tokens),
            "min_new_tokens": overrides.get("min_new_tokens", self.min_new_tokens),
            "do_sample": True,
            "temperature": overrides.get("temperature", self.temperature),
            "top_p": overrides.get("top_p", self.top_p),
            "repetition_penalty": overrides.get("repetition_penalty", self.repetition_penalty),
            "no_repeat_ngram_size": overrides.get("no_repeat_ngram_size", self.no_repeat_ngram_size),
        }

        if streamer is not None:
            gen_kwargs["streamer"] = streamer

        return gen_kwargs
    
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
    
    # =========================================================================
    # STREAMING METHOD 
    # =========================================================================
    
    def stream_generate(
        self, 
        prompt: str,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream generation - yields text chunks as they're generated.
        
        Args:
            prompt: Input prompt
            stop_sequences: Optional list of sequences to stop generation
            **kwargs: Override generation params (temperature, repetition_penalty, etc.)
            
        Yields:
            Text chunks as they're generated
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        # Build generation kwargs với anti-repetition
        gen_kwargs = self._build_generation_kwargs(inputs, streamer, **kwargs)
        
        # Run model in separate thread
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        # Yield chunks as they come
        try:
            for new_text in streamer:
                # Check stop sequences
                if stop_sequences and any(s in new_text for s in stop_sequences):
                    break
                yield new_text
        finally:
            # Ensure thread completes
            thread.join(timeout=1.0)
    
    # =========================================================================
    # INTERNAL GENERATION
    # =========================================================================
    
    def _generate_internal(
        self, 
        prompt: str, 
        print_stream: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> tuple[str, dict]:
        """Core generation logic với streaming - collects full output"""
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        num_input_tokens = inputs["input_ids"].shape[-1]
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        # Build generation kwargs với anti-repetition
        gen_kwargs = self._build_generation_kwargs(inputs, streamer, **kwargs)
        
        # Chạy model trong thread riêng để stream output
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        generated_parts = []
        start_time = time.time()
        
        for new_text in streamer:
            if print_stream:
                print(new_text, end="", flush=True)
            generated_parts.append(new_text)
            
            # Stop sequence check
            if stop_sequences and any(s in new_text for s in stop_sequences):
                break
        
        thread.join(timeout=1.0)
        end_time = time.time()
        
        continuation = "".join(generated_parts)
        generation_time = end_time - start_time
        
        # Tính toán thống kê
        num_new_tokens = 0
        if continuation.strip():
            num_new_tokens = len(self.tokenizer.encode(continuation, add_special_tokens=False))
            
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

    # =========================================================================
    # PARSING
    # =========================================================================
    
    @staticmethod
    def parse_generated_text(raw_text: str) -> List[Dict[str, str]]:
        """
        Chuyển đổi raw text output của Shakespeare model thành JSON list.
        
        Format expected:
        - [CHARACTER] Dialogue line
        - {Stage direction}
        
        Returns:
            List of dicts with 'character' and 'line' keys
        """
        dialogue_list = []
        
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        for line in lines:
            # Case 1: Stage Direction {Action}
            stage_match = re.match(r'^\{(.*)\}$', line)
            if stage_match:
                content = stage_match.group(1).strip()
                if content:
                    dialogue_list.append({
                        "character": "STAGE_DIRECTION",
                        "line": content
                    })
                continue
            
            # Case 2: Dialogue [CHARACTER] Line
            dialogue_match = re.match(r'^\[(.*?)\]\s*(.*)$', line)
            if dialogue_match:
                char_name = dialogue_match.group(1).strip().upper()
                speech = dialogue_match.group(2).strip()
                
                if not speech:
                    continue
                    
                dialogue_list.append({
                    "character": char_name,
                    "line": speech
                })
                continue
            
        return dialogue_list