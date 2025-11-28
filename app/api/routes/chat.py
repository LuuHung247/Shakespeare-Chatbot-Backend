"""Chat endpoints (placeholder)."""
from fastapi import APIRouter

router = APIRouter()

@router.post("/chat")
def chat_endpoint(payload: dict):
    return {"reply": "This is a placeholder."}
