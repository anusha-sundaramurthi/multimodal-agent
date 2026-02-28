"""
backend/main.py
Lightweight proxy server — runs locally in VS Code.
Forwards requests to the Colab model server.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

COLAB_API_URL = os.getenv("COLAB_API_URL", "")  # Set this in .env

app = FastAPI(title="Multimodal Agent - Backend Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend static files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


class PromptRequest(BaseModel):
    prompt: str


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "colab_url_configured": bool(COLAB_API_URL)}


@app.post("/api/generate")
async def generate(request: PromptRequest):
    if not COLAB_API_URL:
        raise HTTPException(
            status_code=503,
            detail="COLAB_API_URL not set. Please add it to your .env file."
        )

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{COLAB_API_URL}/generate",
                json={"prompt": request.prompt}
            )
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Cannot connect to Colab model server. Make sure the Colab notebook is running."
            )
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail="Model server timed out. Generation can take 1-3 minutes, please try again."
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))


@app.get("/api/colab-status")
async def colab_status():
    """Check if the Colab model server is reachable."""
    if not COLAB_API_URL:
        return {"online": False, "reason": "COLAB_API_URL not configured"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{COLAB_API_URL}/health")
            return {"online": True, "detail": response.json()}
        except Exception as e:
            return {"online": False, "reason": str(e)}
