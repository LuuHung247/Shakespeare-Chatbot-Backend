from app.models.requests import ContinueSceneRequest
from app.models.responses import ContinueSceneResponse
from app.services.preprocessing import dialogue_preprocessor
from typing import List, Dict
import json

class ChatService:
    """Service xử lý scene continuation"""
    
    async def continue_scene(self, request: ContinueSceneRequest) -> ContinueSceneResponse:
        """Continue Shakespeare scene"""
        try:
            # Convert dialogue to string for prompt
            dialogue_string = self._preprocess_dialogue(request.previous_dialogue)
            
            # Build prompt
            prompt = self._build_prompt(request, dialogue_string)
            
            # Generate từ LLM
            from app.core.llm_manager import llm_manager
            llm_output = llm_manager.generate(prompt, temperature=request.temperature)
            
            # Parse to JSON array
            continuation_json = self._parse_to_json(llm_output)
            
            # Validate
            if not self._validate_json(continuation_json):
                raise ValueError("Invalid JSON format from LLM")
            
            # Count characters
            characters_used = self._count_characters(
                continuation_json, 
                request.scene_context.characters
            )
            
            return ContinueSceneResponse(
                success=True,
                continue_dialogue=continuation_json,
                character_count=len(characters_used),
                metadata={
                    "style": request.style,
                    "max_lines": request.max_lines,
                    "temperature": request.temperature,
                    "characters_in_continuation": characters_used
                }
            )
        
        except Exception as e:
            return ContinueSceneResponse(
                success=False,
                continue_dialogue=[
                    {"character": "ERROR", "line": f"Error: {str(e)}"}
                ],
                character_count=0,
                metadata={"error": str(e)}
            )

    def _parse_to_json(self, llm_output: str) -> List[Dict[str, str]]:
        """Parse LLM output to JSON array"""
        llm_output = llm_output.strip()
        
        # Remove markdown code blocks
        if llm_output.startswith("```json"):
            llm_output = llm_output.replace("```json", "").replace("```", "").strip()
        elif llm_output.startswith("```"):
            llm_output = llm_output.replace("```", "").strip()
        
        try:
            # Try parse as JSON
            parsed = json.loads(llm_output)
            
            if isinstance(parsed, list):
                if all(isinstance(item, dict) and "character" in item and "line" in item for item in parsed):
                    return parsed
            
            raise ValueError("Invalid JSON structure")
            
        except (json.JSONDecodeError, ValueError):
            # Try convert from string format
            try:
                converted = dialogue_preprocessor.string_to_json_array(llm_output)
                if converted and len(converted) > 0:
                    return converted
            except Exception:
                pass
            
            raise ValueError(f"Cannot parse to JSON array: {llm_output[:200]}")
    
    def _validate_json(self, dialogue_json: List[Dict[str, str]]) -> bool:
        """Validate JSON format"""
        if not isinstance(dialogue_json, list) or len(dialogue_json) == 0:
            return False
        
        for item in dialogue_json:
            if not isinstance(item, dict):
                return False
            if "character" not in item or "line" not in item:
                return False
            if not isinstance(item["character"], str) or not isinstance(item["line"], str):
                return False
        
        return True
    
    def _preprocess_dialogue(self, dialogue_input):
        """Convert dialogue input to string format"""
        if isinstance(dialogue_input, str):
            return dialogue_preprocessor.clean_dialogue_text(dialogue_input)
        
        if isinstance(dialogue_input, list):
            return dialogue_preprocessor.json_to_dialogue_string(dialogue_input)
        
        return str(dialogue_input)
    
    def _build_prompt(self, request: ContinueSceneRequest, dialogue_string: str) -> str:
        """Build prompt for scene continuation"""
        
        characters_info = "\n".join([
            f"- {char.name}: {char.personality} (Emotion: {char.emotion_in_scene})"
            for char in request.scene_context.characters
        ])
        
        prompt = f"""You are a Shakespeare-style playwright assistant.

SCENE INFO:
Summary: {request.scene_context.scene_summary}

Characters ({request.scene_context.character_count}):
{characters_info}

Plot State: {request.scene_context.plot_state}
Next Direction: {request.scene_context.likely_next_direction}

PREVIOUS DIALOGUE:
{dialogue_string}

TASK: Continue the scene immediately after the last line.

RULES:
1. Do NOT repeat or modify the given dialogue
2. Follow the narrative direction strictly
3. Do NOT introduce new conflicts or characters beyond what's described
4. Use ONLY the listed characters
5. Write only dialogue and stage directions, no prose

OUTPUT FORMAT (CRITICAL):
Return ONLY a valid JSON array with NO markdown, NO code blocks, NO extra text.

[
  {{"character": "CHARACTER_NAME", "line": "dialogue text"}},
  {{"character": "STAGE_DIRECTION", "line": "stage direction"}},
  ...
]

Requirements:
- Character names in UPPERCASE
- Use "STAGE_DIRECTION" for stage directions
- Maximum {request.max_lines} lines
- Authentic Early Modern English
- Sharp, focused dramatic lines

Return only the JSON array now:"""
        
        return prompt
    
    def _count_characters(self, dialogue_json: List[Dict[str, str]], characters: list) -> list:
        """Count unique characters in dialogue"""
        characters_used = []
        character_names = [char.name.upper() for char in characters]
        
        for item in dialogue_json:
            char_name = item.get("character", "").strip().upper()
            
            if char_name == "STAGE_DIRECTION":
                continue
            
            if char_name in character_names and char_name not in characters_used:
                characters_used.append(char_name)
        
        return characters_used

# Singleton
chat_service = ChatService()