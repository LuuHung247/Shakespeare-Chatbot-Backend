from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.models.requests import ContinueSceneRequest
from app.models.responses import ContinueSceneResponse, HealthResponse
from app.services.chat_service import chat_service
import uvicorn
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup và shutdown events"""
    print("🚀 Initializing Shakespeare Scene Continuation API...")
    print("✅ Backend ready!")
    yield
    print("👋 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Shakespeare Scene Continuation API",
    version=settings.VERSION,
    description="API for continuing Shakespeare-style scenes",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        timestamp=""
    )

# Main endpoint - Continue scene
@app.post("/api/continue-scene", response_model=ContinueSceneResponse)
async def continue_scene(request: ContinueSceneRequest):
    """
    Continue Shakespeare scene với context và previous dialogue
    
    Returns JSON format:
    {
        "success": true,
        "continue_dialogue": [
            {"character": "NAME", "line": "dialogue"},
            ...
        ],
        "character_count": 2,
        "metadata": {...}
    }
    """
    try:
        response = await chat_service.continue_scene(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )