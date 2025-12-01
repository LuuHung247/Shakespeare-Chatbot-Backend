from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Literal
from datetime import datetime


class ContinueSceneResponse(BaseModel):
    """Response cho scene continuation"""
    success: bool = Field(..., description="Trạng thái thành công")
    continue_dialogue: List[Dict[str, str]] = Field(
        ..., 
        description="Đoạn dialogue tiếp theo - JSON array format"
    )
    character_count: int = Field(
        ..., 
        description="Số nhân vật xuất hiện trong continuation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata về generation"
    )


class HealthResponse(BaseModel):
    """Response cho health check"""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Thời gian check"
    )


# Streaming event models
class StreamMetadata(BaseModel):
    """Metadata event for streaming"""
    type: Literal["metadata"] = "metadata"
    data: Dict[str, Any] = Field(..., description="Generation metadata")


class StreamChunk(BaseModel):
    """Text chunk event for streaming"""
    type: Literal["chunk"] = "chunk"
    data: str = Field(..., description="Raw text chunk from LLM")


class StreamDialogue(BaseModel):
    """Parsed dialogue event for streaming"""
    type: Literal["dialogue"] = "dialogue"
    data: Dict[str, str] = Field(..., description="Parsed dialogue line with character and line")


class StreamCompleteDialogue(BaseModel):
    """Complete dialogue array event"""
    type: Literal["complete_dialogue"] = "complete_dialogue"
    data: List[Dict[str, str]] = Field(..., description="Complete array of dialogue lines")


class StreamDone(BaseModel):
    """Completion event for streaming"""
    type: Literal["done"] = "done"
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Final statistics and metadata"
    )


class StreamError(BaseModel):
    """Error event for streaming"""
    type: Literal["error"] = "error"
    data: Dict[str, str] = Field(..., description="Error information")


# Union type cho tất cả stream events
StreamEventResponse = Union[
    StreamMetadata,
    StreamChunk,
    StreamDialogue,
    StreamCompleteDialogue,
    StreamDone,
    StreamError
]