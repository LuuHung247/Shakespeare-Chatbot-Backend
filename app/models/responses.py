from pydantic import BaseModel, Field
from typing import List, Dict, Any
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