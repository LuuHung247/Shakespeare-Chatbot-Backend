from typing import List, Dict, Any
import json
import re

from app.models.requests import ContinueSceneRequest
from app.models.responses import ContinueSceneResponse
from app.services.preprocessing import dialogue_preprocessor
from app.core.llm_manager import llm_manager
from app.config import settings
from app.llm.custom_shakespeare_llm import ShakespeareLLM

class ChatService:
    """Service xử lý scene continuation hỗ trợ cả API LLM và Custom Local LLM"""
    
    async def continue_scene(self, request: ContinueSceneRequest) -> ContinueSceneResponse:
        """
        Main logic để viết tiếp kịch bản.
        """
        try:
            # 1. Preprocess dialogue (list -> string)
            dialogue_string = self._preprocess_dialogue(request.previous_dialogue)
            
            continuation_json = []

            # 2. PHÂN NHÁNH XỬ LÝ DỰA TRÊN MODEL CONFIG
            if settings.LLM_TYPE == "shakespeare":
                # === NHÁNH LOCAL SHAKESPEARE MODEL ===
                
                # a. Build prompt dành riêng cho model Shakespeare (Dùng header chuẩn của bạn)
                # Lưu ý: Model custom cần input dạng dict để build prompt, nhưng ở đây ta build string luôn
                prompt = self._build_shakespeare_prompt_structured(request, dialogue_string)
                
                # b. Generate Raw Text (Sử dụng hàm generate của manager)
                # LLM Manager cần hỗ trợ nhận prompt string trực tiếp hoặc ta đóng gói lại nếu cần
                # Ở đây giả sử ShakespeareLLM.generate_with_stats nhận vào prompt string
                raw_text, _ = llm_manager.generate_shakespeare_from_string(prompt, print_stream=False)
                
                # c. Parse Raw Text -> JSON chuẩn bằng Regex
                continuation_json = ShakespeareLLM.parse_generated_text(raw_text)

            else:
                # === NHÁNH GEMINI / OPENAI / ANTHROPIC ===
                
                # a. Build prompt dạng Instruction (Ép output JSON)
                prompt = self._build_gemini_prompt(request, dialogue_string)
                
                # b. Generate JSON String
                llm_output = llm_manager.generate(prompt, temperature=request.temperature)
                
                # c. Parse JSON String -> JSON Object
                continuation_json = self._parse_api_output_to_json(llm_output)
            
            # 3. XỬ LÝ CHUNG (VALIDATION & RESPONSE)
            
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
    # 1. PROMPT BUILDERS
    # =========================================================================

    def _build_gemini_prompt(self, request: ContinueSceneRequest, dialogue_string: str) -> str:
        """
        Prompt dành cho Gemini/OpenAI: Yêu cầu output JSON trực tiếp.
        """
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
        """
        Prompt dành cho Custom Model: Sử dụng Header gốc mà model đã được train/test tốt.
        """
        
        # 1. Chuẩn bị thông tin nhân vật
        characters_info = "\n".join([
            f"- {char.name}: personality: {char.personality}; emotion: {char.emotion_in_scene}"
            for char in request.scene_context.characters
        ])

        # 2. HEADER GỐC (Giữ nguyên văn theo yêu cầu)
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

        # 3. Construct Body (Tương ứng với phần INFO được nhắc đến trong Header)
        # Sắp xếp các mục rõ ràng để model dễ tham chiếu
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
    # 2. PARSING & VALIDATION LOGIC
    # =========================================================================

    def _parse_api_output_to_json(self, llm_output: str) -> List[Dict[str, str]]:
        """Parse JSON string từ API (Gemini/OpenAI)"""
        llm_output = llm_output.strip()
        
        if llm_output.startswith("```json"):
            llm_output = llm_output.replace("```json", "").replace("```", "").strip()
        elif llm_output.startswith("```"):
            llm_output = llm_output.replace("```", "").strip()
        
        try:
            parsed = json.loads(llm_output)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "continue_dialogue" in parsed:
                return parsed["continue_dialogue"]
            return []
        except (json.JSONDecodeError, ValueError):
            try:
                converted = dialogue_preprocessor.string_to_json_array(llm_output)
                if converted: return converted
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

    # =========================================================================
    # 3. HELPER METHODS
    # =========================================================================

    def _preprocess_dialogue(self, dialogue_input):
        if isinstance(dialogue_input, str):
            return dialogue_preprocessor.clean_dialogue_text(dialogue_input)
        
        if isinstance(dialogue_input, list):
            return dialogue_preprocessor.json_to_dialogue_string(dialogue_input)
        
        return str(dialogue_input)
    
    def _count_characters(self, dialogue_json: List[Dict[str, str]], characters: list) -> list:
        characters_used = []
        defined_names = {char.name.upper() for char in characters}
        
        for item in dialogue_json:
            char_name = item.get("character", "").strip().upper()
            if char_name == "STAGE_DIRECTION":
                continue
            if char_name in defined_names and char_name not in characters_used:
                characters_used.append(char_name)
        
        return characters_used

# Singleton Instance
chat_service = ChatService()