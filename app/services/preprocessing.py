from typing import List, Dict

class DialoguePreprocessor:
    """Convert dialogue formats"""
    
    @staticmethod
    def json_to_dialogue_string(dialogue_array: List[Dict[str, str]]) -> str:
        """
        Convert JSON array to formatted string
        
        Input: [{"character": "HOSTESS", "line": "No, good Captain..."}, ...]
        Output: [HOSTESS] No, good Captain...\n{stage direction}
        """
        formatted_lines = []
        
        for item in dialogue_array:
            character = item.get("character", "").strip()
            line = item.get("line", "").strip()
            
            if not line:
                continue
            
            if character.upper() == "STAGE_DIRECTION":
                formatted_lines.append(f"{{{line}}}")
            else:
                formatted_lines.append(f"[{character.upper()}] {line}")
        
        return "\n".join(formatted_lines)
    
    @staticmethod
    def string_to_json_array(dialogue_string: str) -> List[Dict[str, str]]:
        """
        Convert string to JSON array
        
        Input: [HOSTESS] No, good Captain...\n{stage direction}
        Output: [{"character": "HOSTESS", "line": "No, good Captain..."}, ...]
        """
        dialogue_array = []
        
        for line in dialogue_string.split('\n'):
            line = line.strip()
            
            if not line:
                continue
            
            # Stage direction
            if line.startswith('{') and line.endswith('}'):
                stage_text = line[1:-1].strip()
                dialogue_array.append({
                    "character": "STAGE_DIRECTION",
                    "line": stage_text
                })
            
            # Character dialogue
            elif line.startswith('['):
                bracket_end = line.find(']')
                if bracket_end > 0:
                    character = line[1:bracket_end].strip()
                    dialogue_text = line[bracket_end + 1:].strip()
                    dialogue_array.append({
                        "character": character,
                        "line": dialogue_text
                    })
        
        return dialogue_array
    
    @staticmethod
    def clean_dialogue_text(text: str) -> str:
        """Clean dialogue text - remove extra spaces"""
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        return '\n'.join(lines)

# Singleton instance
dialogue_preprocessor = DialoguePreprocessor()