from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union

class CharacterInfo(BaseModel):
    """Thông tin về 1 nhân vật"""
    name: str
    personality: str
    emotion_in_scene: str

class SceneContext(BaseModel):
    """Context của scene từ JSON"""
    character_count: int
    characters: List[CharacterInfo]
    scene_summary: str
    plot_state: str
    likely_next_direction: str

class ContinueSceneRequest(BaseModel):
    """Request để tiếp tục scene Shakespeare với context và dialogue"""
    scene_context: SceneContext = Field(..., description="JSON context về scene, characters, plot")
    previous_dialogue: Union[str, List[Dict[str, str]]] = Field(
        ..., 
        description="Đoạn dialogue trước đó - có thể là string format '[CHAR] line' hoặc JSON array"
    )
    style: Optional[str] = Field(
        default="shakespeare", 
        description="Style của dialogue: shakespeare, tragedy, comedy"
    )
    max_lines: Optional[int] = Field(
        default=20, 
        ge=1, 
        le=50, 
        description="Số dòng dialogue tối đa"
    )
    temperature: Optional[float] = Field(
        default=0.8, 
        ge=0, 
        le=2, 
        description="Temperature cho LLM generation"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "max_lines": 15,
                "previous_dialogue": [
                    {
                        "character": "HOSTESS",
                        "line": "No, good Captain Pistol, not here, sweet captain."
                    },
                    {
                        "character": "PISTOL",
                        "line": "Not I. I tell thee what, Corporal Bardolph, I could tear her."
                    },
                    {
                        "character": "STAGE_DIRECTION",
                        "line": "He draws his sword."
                    },
                    {
                        "character": "BARDOLPH",
                        "line": "Begone, good ancient. This will grow to a brawl anon."
                    }
                ],
                "scene_context": {
                    "character_count": 6,
                    "characters": [
                        {
                            "emotion_in_scene": "Anxious, pleading",
                            "name": "HOSTESS",
                            "personality": "Worried tavern keeper, trying to maintain peace"
                        },
                        {
                            "emotion_in_scene": "Furious, insulted, ready for violence",
                            "name": "PISTOL",
                            "personality": "Boastful, aggressive, prone to bombastic language"
                        },
                        {
                            "emotion_in_scene": "Angry, mocking",
                            "name": "DOLL",
                            "personality": "Sharp-tongued, confrontational"
                        },
                        {
                            "emotion_in_scene": "Concerned, attempting to defuse tension",
                            "name": "BARDOLPH",
                            "personality": "Peacekeeper, trying to prevent conflict"
                        },
                        {
                            "emotion_in_scene": "Distracted, focused on his own pleasures",
                            "name": "FALSTAFF",
                            "personality": "Jovial, self-indulgent"
                        },
                        {
                            "emotion_in_scene": "Nervous, trying to help",
                            "name": "PAGE",
                            "personality": "Young servant"
                        }
                    ],
                    "likely_next_direction": "The confrontation will either escalate or be defused. Pistol may be forcibly removed or calmed down.",
                    "plot_state": "Pistol has drawn his sword in anger after being insulted. Others are attempting to prevent violence.",
                    "scene_summary": "A heated confrontation in a tavern between Pistol and others. Doll mocks his pretensions. The Hostess tries to calm the situation."
                },
                "style": "shakespeare",
                "temperature": 0.3
            }
        }