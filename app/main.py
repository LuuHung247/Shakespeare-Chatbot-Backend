from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.models.requests import ContinueSceneRequest
from app.models.responses import ContinueSceneResponse, HealthResponse, StreamEventResponse
from app.services.chat_service import chat_service
import uvicorn
from contextlib import asynccontextmanager
from typing import AsyncGenerator

router = APIRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup và shutdown events"""
    print("🚀 Initializing Shakespeare Scene Continuation API...")
    print("✅ Backend ready!")
    
    # Debug: Print all registered routes
    print("\n📋 Registered routes:")
    for route in app.routes:
        if hasattr(route, 'methods'):
            print(f"  {list(route.methods)[0]:6} {route.path}")
    print()
    
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
@router.post("/api/continue-scene", response_model=ContinueSceneResponse)
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


@router.post(
    "/api/continue-scene/stream",
    responses={
        200: {
            "description": "Server-Sent Events stream of dialogue generation",
            "content": {
                "text/event-stream": {
                    "example": "data: {\"type\": \"metadata\", \"data\": {...}}\n\ndata: {\"type\": \"chunk\", \"data\": \"HAMLET: To be...\"}\n\n"
                }
            }
        }
    },
    summary="Stream Shakespeare scene continuation",
    description="""
    Streaming endpoint - Returns Server-Sent Events (SSE)
    
    **Event types:**
    - `metadata`: Initial metadata about generation
    - `chunk`: Raw text chunk from LLM
    - `dialogue`: Parsed dialogue line (character + line)
    - `complete_dialogue`: Final complete dialogue array
    - `done`: Generation completed
    - `error`: Error occurred
    
    **Example usage with JavaScript:**
    ```javascript
    const eventSource = new EventSource('/api/continue-scene/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    });
    
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data.type, data.data);
    };
    
    eventSource.addEventListener('error', () => {
        eventSource.close();
    });
    ```
    
    **Example usage with Python:**
    ```python
    import requests
    
    response = requests.post(
        'http://localhost:8000/api/continue-scene/stream',
        json=request_data,
        stream=True
    )
    
    for line in response.iter_lines():
        if line.startswith(b'data: '):
            data = json.loads(line[6:])
            print(data)
    ```
    
    **Note:** Swagger UI cannot test SSE endpoints directly. Use curl, Postman, or custom client code.
    """,
    tags=["Streaming"]
)
async def continue_scene_stream(request: ContinueSceneRequest):
    """
    Streaming endpoint for real-time dialogue generation
    """
    try:
        return StreamingResponse(
            chat_service.continue_scene_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Include router AFTER defining all routes
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )