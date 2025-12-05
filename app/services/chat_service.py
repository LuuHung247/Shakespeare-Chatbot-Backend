from typing import List, Dict, Any, AsyncGenerator, Set, Tuple
from io import StringIO
import re

# Sử dụng orjson nếu có (nhanh hơn 3-10x), fallback về json
try:
    import orjson
    def json_dumps(data: dict) -> str:
        return orjson.dumps(data).decode('utf-8')
    def json_loads(data: str) -> Any:
        return orjson.loads(data)
    JSON_LIB = "orjson"
except ImportError:
    import json
    def json_dumps(data: dict) -> str:
        return json.dumps(data, ensure_ascii=False)
    def json_loads(data: str) -> Any:
        return json.loads(data)
    JSON_LIB = "json"

from app.models.requests import ContinueSceneRequest
from app.models.responses import ContinueSceneResponse
from app.services.preprocessing import dialogue_preprocessor
from app.core.llm_manager import llm_manager
from app.config import settings
from app.llm.custom_shakespeare_llm import ShakespeareLLM


class ChatService:
    """
    Optimized Service xử lý scene continuation.
    
    Optimizations:
    - Pre-compiled regex patterns
    - orjson for faster JSON serialization (if available)
    - StringIO buffer thay vì string concatenation
    - Reduced function call overhead
    - Cached character set lookup
    """
    
    # Chỉ giữ delay cho Gemini (backward compatibility)
    DIALOGUE_DELAY = 2.0
    
    # =========================================================================
    # PRE-COMPILED REGEX PATTERNS
    # =========================================================================
    
    # Pattern cho Shakespeare format
    _STAGE_DIRECTION_PATTERN = re.compile(r'^\{(.+)\}$')
    _DIALOGUE_PATTERN = re.compile(r'^\[([A-Z][A-Z\s]*)\]\s*(.+)$')
    
    # Pattern cho JSON cleanup
    _MARKDOWN_PATTERN = re.compile(r'```(?:json)?\s*')
    
    # SSE format template (avoid f-string overhead)
    _SSE_TEMPLATE = "data: {}\n\n"
    
    def __init__(self):
        """Initialize với logging về JSON library đang dùng"""
        print(f"ChatService initialized with {JSON_LIB} for JSON serialization")
    
    # =========================================================================
    # STREAMING ENTRY POINT
    # =========================================================================
    
    async def continue_scene_stream(
        self, 
        request: ContinueSceneRequest
    ) -> AsyncGenerator[str, None]:
        """
        Stream scene continuation - yield từng chunk JSON
        Format: Server-Sent Events (SSE)
        """
        try:
            # 1. Preprocess dialogue
            dialogue_string = self._preprocess_dialogue(request.previous_dialogue)
            
            # 2. Send metadata first
            yield self._format_sse({
                "type": "metadata",
                "data": {
                    "model_type": settings.LLM_TYPE,
                    "style": request.style,
                    "max_lines": request.max_lines,
                    "temperature": request.temperature
                }
            })
            
            # 3. Route to appropriate handler
            if settings.LLM_TYPE == "shakespeare":
                async for chunk in self._stream_shakespeare(request, dialogue_string):
                    yield chunk
                    
            elif settings.LLM_TYPE == "gemini":
                async for chunk in self._stream_gemini(request, dialogue_string):
                    yield chunk
            else:
                async for chunk in self._stream_generic(request, dialogue_string):
                    yield chunk
            
            # 4. Send completion signal
            yield self._format_sse({
                "type": "done",
                "data": {"message": "Generation completed"}
            })
            
        except Exception as e:
            yield self._format_sse({
                "type": "error",
                "data": {"error": str(e)}
            })
    
    # =========================================================================
    # OPTIMIZED SHAKESPEARE STREAMING
    # =========================================================================
    
    async def _stream_shakespeare(
        self, 
        request: ContinueSceneRequest, 
        dialogue_string: str
    ) -> AsyncGenerator[str, None]:
        """
        Optimized streaming cho Shakespeare model.
        
        Uses:
        - StringIO buffer thay vì string += 
        - Pre-compiled regex
        - Set với frozenset keys cho O(1) lookup
        """
        prompt = self._build_shakespeare_prompt_structured(request, dialogue_string)
        
        # StringIO buffer - hiệu quả hơn string concatenation
        buffer = StringIO()
        buffer_pos = 0  # Track read position
        last_parsed_end = 0
        sent_dialogues: Set[Tuple[str, str]] = set()
        
        # Stream từ model
        for chunk in llm_manager.stream_shakespeare(prompt):
            # Write to buffer (O(1) amortized)
            buffer.write(chunk)
            
            # Gửi raw chunk ngay lập tức
            yield self._format_sse({
                "type": "chunk",
                "data": {"text": chunk}
            })
            
            # Get current buffer content for parsing
            current_content = buffer.getvalue()
            
            # Parse incremental
            new_dialogues, new_end = self._parse_shakespeare_incremental(
                current_content, 
                last_parsed_end,
                sent_dialogues
            )
            
            # Gửi các dialogue mới
            for dialogue in new_dialogues:
                sent_dialogues.add((dialogue["character"], dialogue["line"]))
                yield self._format_sse({
                    "type": "dialogue",
                    "data": dialogue
                })
            
            if new_end > last_parsed_end:
                last_parsed_end = new_end
        
        # Final parse
        final_content = buffer.getvalue()
        remaining_dialogues = self._parse_shakespeare_remaining(
            final_content, 
            last_parsed_end, 
            sent_dialogues
        )
        
        for dialogue in remaining_dialogues:
            yield self._format_sse({
                "type": "dialogue",
                "data": dialogue
            })
        
        # Cleanup
        buffer.close()
    
    def _parse_shakespeare_incremental(
        self, 
        buffer: str, 
        start_pos: int,
        already_sent: Set[Tuple[str, str]]
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Optimized incremental parsing với pre-compiled regex.
        """
        last_newline = buffer.rfind('\n')
        if last_newline <= start_pos:
            return [], start_pos
        
        # Parse segment
        safe_start = max(0, start_pos - 100)
        parse_segment = buffer[safe_start:last_newline + 1]
        
        results = []
        
        for line in parse_segment.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Stage direction - dùng pre-compiled pattern
            stage_match = self._STAGE_DIRECTION_PATTERN.match(line)
            if stage_match:
                content = stage_match.group(1).strip()
                if content:
                    key = ("STAGE_DIRECTION", content)
                    if key not in already_sent:
                        results.append({
                            "character": "STAGE_DIRECTION",
                            "line": content
                        })
                continue
            
            # Character dialogue - dùng pre-compiled pattern
            dialogue_match = self._DIALOGUE_PATTERN.match(line)
            if dialogue_match:
                char_name = dialogue_match.group(1).strip().upper()
                speech = dialogue_match.group(2).strip()
                if char_name and speech:
                    key = (char_name, speech)
                    if key not in already_sent:
                        results.append({
                            "character": char_name,
                            "line": speech
                        })
        
        return results, last_newline + 1
    
    def _parse_shakespeare_remaining(
        self, 
        buffer: str, 
        start_pos: int,
        already_sent: Set[Tuple[str, str]]
    ) -> List[Dict[str, str]]:
        """Parse phần còn lại với pre-compiled regex."""
        remaining = buffer[start_pos:].strip()
        if not remaining:
            return []
        
        results = []
        
        # Stage direction
        stage_match = self._STAGE_DIRECTION_PATTERN.match(remaining)
        if stage_match:
            content = stage_match.group(1).strip()
            if content and ("STAGE_DIRECTION", content) not in already_sent:
                results.append({
                    "character": "STAGE_DIRECTION",
                    "line": content
                })
            return results
        
        # Character dialogue
        dialogue_match = self._DIALOGUE_PATTERN.match(remaining)
        if dialogue_match:
            char_name = dialogue_match.group(1).strip().upper()
            speech = dialogue_match.group(2).strip()
            if char_name and speech and (char_name, speech) not in already_sent:
                results.append({
                    "character": char_name,
                    "line": speech
                })
        
        return results
    
    # =========================================================================
    # GEMINI STREAMING 
    # =========================================================================
    
    async def _stream_gemini(
        self, 
        request: ContinueSceneRequest, 
        dialogue_string: str
    ) -> AsyncGenerator[str, None]:
        """Stream output từ Gemini"""
        from asyncio import sleep
        
        prompt = self._build_gemini_prompt(request, dialogue_string)
        
        buffer = StringIO()
        
        async for chunk in llm_manager.stream_generate(prompt, temperature=request.temperature):
            buffer.write(chunk)
            
            yield self._format_sse({
                "type": "chunk",
                "data": {"text": chunk}
            })
        
            parsed = self._try_parse_json_buffer(buffer.getvalue())
            if parsed:
                for item in parsed:
                    yield self._format_sse({
                        "type": "dialogue",
                        "data": item
                    })
                    await sleep(self.DIALOGUE_DELAY)
        
        final_json = self._parse_api_output_to_json(buffer.getvalue())
        if self._validate_json(final_json):
            yield self._format_sse({
                "type": "complete_dialogue",
                "data": final_json
            })
            await sleep(self.DIALOGUE_DELAY)
        
        buffer.close()
    
    async def _stream_generic(
        self, 
        request: ContinueSceneRequest, 
        dialogue_string: str
    ) -> AsyncGenerator[str, None]:
        """Generic streaming cho các model khác"""
        prompt = self._build_gemini_prompt(request, dialogue_string)
        
        async for chunk in llm_manager.stream_generate(prompt):
            yield self._format_sse({
                "type": "chunk",
                "data": {"text": chunk}
            })
    
    # =========================================================================
    # OPTIMIZED HELPERS
    # =========================================================================
    
    def _format_sse(self, data: dict) -> str:
        """Format SSE với orjson (nếu có)"""
        return self._SSE_TEMPLATE.format(json_dumps(data))
    
    def _try_parse_json_buffer(self, buffer: str) -> List[Dict[str, str]]:
        """Parse JSON với pre-compiled regex và orjson"""
        try:
            clean = buffer.strip()
            if clean.startswith("```"):
                clean = self._MARKDOWN_PATTERN.sub('', clean)
            
            if '[' in clean and ']' in clean:
                start = clean.index('[')
                end = clean.rindex(']') + 1
                return json_loads(clean[start:end])
        except:
            pass
        
        return []
    
    # =========================================================================
    # NON-STREAMING METHOD
    # =========================================================================
    
    async def continue_scene(self, request: ContinueSceneRequest) -> ContinueSceneResponse:
        """Non-streaming version"""
        try:
            dialogue_string = self._preprocess_dialogue(request.previous_dialogue)
            continuation_json = []

            if settings.LLM_TYPE == "shakespeare":
                prompt = self._build_shakespeare_prompt_structured(request, dialogue_string)
                raw_text, _ = llm_manager.generate_shakespeare_from_string(prompt, print_stream=False)
                continuation_json = ShakespeareLLM.parse_generated_text(raw_text)
            else:
                prompt = self._build_gemini_prompt(request, dialogue_string)
                llm_output = llm_manager.generate(prompt, temperature=request.temperature)
                continuation_json = self._parse_api_output_to_json(llm_output)
            
            if not self._validate_json(continuation_json):
                if not continuation_json:
                    raise ValueError("LLM generated empty response or invalid format.")
            
            characters_used = self._count_characters(
                continuation_json, 
                request.scene_context.characters
            )
            
            return ContinueSceneResponse(
                success=True,
                continue_dialogue=continuation_json,
                character_count=len(characters_used),
                metadata={
                    "model_type": settings.LLM_TYPE,
                    "style": request.style,
                    "max_lines": request.max_lines,
                    "temperature": request.temperature,
                    "characters_in_continuation": characters_used
                }
            )
        
        except Exception as e:
            print(f"Error in continue_scene: {e}")
            return ContinueSceneResponse(
                success=False,
                continue_dialogue=[
                    {"character": "ERROR", "line": f"Generation failed: {str(e)}"}
                ],
                character_count=0,
                metadata={"error": str(e)}
            )

    # =========================================================================
    # PROMPT BUILDERS 
    # =========================================================================

    def _build_gemini_prompt(self, request: ContinueSceneRequest, dialogue_string: str) -> str:
        characters_info = "\n".join([
            f"- {char.name}: {char.personality} (Emotion: {char.emotion_in_scene})"
            for char in request.scene_context.characters
        ])
        
        return f"""You are a Shakespeare-style playwright assistant.

SCENE INFO:
Summary: {request.scene_context.scene_summary}

CHARACTERS ({request.scene_context.character_count}):
{characters_info}

PLOT STATE: {request.scene_context.plot_state}
NEXT DIRECTION: {request.scene_context.likely_next_direction}

PREVIOUS DIALOGUE:
{dialogue_string}

TASK: 
Continue the scene immediately after the last line.

RULES:
1. Do NOT repeat or modify the given dialogue.
2. Follow the narrative direction strictly.
3. Use ONLY the listed characters.
4. Write only dialogue and stage directions.

OUTPUT FORMAT (CRITICAL):
Return ONLY a valid JSON array. Do not wrap in markdown code blocks.
Example:
[
  {{"character": "HAMLET", "line": "To be, or not to be..."}},
  {{"character": "STAGE_DIRECTION", "line": "He sighs deeply."}}
]

Return JSON array now:"""

    def _build_shakespeare_prompt_structured(self, request: ContinueSceneRequest, dialogue_string: str) -> str:
        characters_info = "\n".join([
            f"- {char.name}: personality: {char.personality}; emotion: {char.emotion_in_scene}"
            for char in request.scene_context.characters
        ])

        header = (
            "You are a creative writing assistant and expert playwright in William Shakespeare's style. "
            "You are given INFO (about the characters, their personalities and emotions, the scene summary, "
            "the plot state, and the likely next narrative direction). Your task is to continue the scene "
            "exactly after the last line of the excerpt. Do not repeat, modify, or reinterpret any part of "
            "the given text. Treat INFO and the excerpt as fixed facts. You must strictly follow the narrative "
            "direction in INFO: the continuation must move only along the path described by the scene_summary, "
            "plot_state, and likely_next_direction. Do NOT introduce new conflicts, wars, crimes, illnesses, "
            "deaths, marriages, miracles, or other dramatic events beyond what INFO clearly supports. Do NOT "
            "introduce any new characters at all, including unnamed roles such as attendants, servants, "
            "messengers, or lords. Use only the characters explicitly listed in INFO as speakers. Do NOT write "
            "narrative prose, summaries, or explanations. Only write dialogue lines and stage directions. "
            "Every spoken line must begin with the fully capitalized speaker name in square brackets, e.g., "
            "[BOTTOM] I shall perform. All stage directions must be enclosed strictly in curly braces, e.g., "
            "{He exits.}. Place each spoken line or stage direction on its own separate line. Do not put more "
            "than one bracketed speaker or stage direction segment on the same line; whenever a new [SPEAKER] "
            "or {STAGE DIRECTION} begins, start a new line. If something is unclear or missing, do not guess: "
            "omit it instead of inventing it. If you are uncertain, keep the continuation very short and "
            "conservative, using only what is safely implied by INFO and the excerpt. Write in authentic Early "
            "Modern English with sharp, focused, Shakespearean dramatic lines."
        )

        return f"""{header}

INFO:
Characters:
{characters_info}

Scene summary:
{request.scene_context.scene_summary}

Plot state:
{request.scene_context.plot_state}

Likely next direction:
{request.scene_context.likely_next_direction}

Continue writing from the following text:
{dialogue_string}
"""

    # =========================================================================
    # PARSING & VALIDATION
    # =========================================================================

    def _parse_api_output_to_json(self, llm_output: str) -> List[Dict[str, str]]:
        llm_output = llm_output.strip()
        
        if llm_output.startswith("```json"):
            llm_output = llm_output.replace("```json", "").replace("```", "").strip()
        elif llm_output.startswith("```"):
            llm_output = llm_output.replace("```", "").strip()
        
        try:
            parsed = json_loads(llm_output)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "continue_dialogue" in parsed:
                return parsed["continue_dialogue"]
            return []
        except (ValueError, TypeError):
            try:
                converted = dialogue_preprocessor.string_to_json_array(llm_output)
                if converted:
                    return converted
            except:
                pass
            return []

    def _validate_json(self, dialogue_json: List[Dict[str, str]]) -> bool:
        if not isinstance(dialogue_json, list):
            return False
        
        valid_items = []
        for item in dialogue_json:
            if isinstance(item, dict) and "character" in item and "line" in item:
                item["character"] = str(item["character"]).upper().strip()
                item["line"] = str(item["line"]).strip()
                valid_items.append(item)
        
        dialogue_json[:] = valid_items
        return len(dialogue_json) > 0

    def _preprocess_dialogue(self, dialogue_input):
        if isinstance(dialogue_input, str):
            return dialogue_preprocessor.clean_dialogue_text(dialogue_input)
        
        if isinstance(dialogue_input, list):
            return dialogue_preprocessor.json_to_dialogue_string(dialogue_input)
        
        return str(dialogue_input)
    
    def _count_characters(self, dialogue_json: List[Dict[str, str]], characters: list) -> list:
        # Pre-compute set for O(1) lookup
        defined_names = {char.name.upper() for char in characters}
        characters_used = []
        seen = set()
        
        for item in dialogue_json:
            char_name = item.get("character", "").strip().upper()
            if char_name == "STAGE_DIRECTION":
                continue
            if char_name in defined_names and char_name not in seen:
                characters_used.append(char_name)
                seen.add(char_name)
        
        return characters_used


# Singleton instance
chat_service = ChatService()