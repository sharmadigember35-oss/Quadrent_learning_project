
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Import your existing RAG functions
from quadrent import (
    EMBEDDING_MODEL_NAME,
    TOP_K,
    ask_llm,
    check_refusal,
    ensure_collection,
    format_timestamp,
    search,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
state = {}


# Load model once when server starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model...")

    try:
        state["model"] = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device="cpu",
    model_kwargs={"low_cpu_mem_usage": True}
)
      

        # Check/create collection without deleting existing data
        ensure_collection()

        print("Server ready.")

        yield

    finally:
        state.clear()


app = FastAPI(
    title="YouTube Lecture RAG API",
    description="Search lecture transcripts and get AI-generated answers with video timestamps.",
    version="1.0.0",
    lifespan=lifespan,
)


# CORS configuration
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        frontend_url,
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# Request model
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=TOP_K, ge=1, le=10)


# Source response model
class Source(BaseModel):
    title: str
    video_id: str
    start_seconds: float
    end_seconds: float
    timestamp: str
    youtube_url: str
    score: float
    text: str


# API response model
class AskResponse(BaseModel):
    question: str
    answer: str
    answered: bool
    sources: List[Source]


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Ask question
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):

    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        # Search relevant transcript segments
        results = search(
            question,
            state["model"],
            top_k=req.top_k
        )

        # Generate answer using Groq
        answer = ask_llm(question, results)

        # Prepare video sources
        sources = []

        for r in results:

            p = r.payload or {}

            start = float(p.get("start", 0))
            end = float(p.get("end", start))

            video_id = p.get("video_id", "")

            youtube_url = (
                f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s"
                if video_id
                else ""
            )

            sources.append(
                Source(
                    title=p.get("title", "Unknown"),
                    video_id=video_id,
                    start_seconds=start,
                    end_seconds=end,
                    timestamp=format_timestamp(start),
                    youtube_url=youtube_url,
                    score=round(float(r.score), 4),
                    text=p.get("text", ""),
                )
            )

        return AskResponse(
            question=question,
            answer=answer,
            answered=not check_refusal(answer),
            sources=sources,
        )

    except HTTPException:
        raise

    except Exception as error:

        print("RAG Error:", error)

        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing your question."
        )


# Serve frontend HTML
@app.get("/")
def index():

    file_path = BASE_DIR / "index.html"

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend index.html not found."
        )

    return FileResponse(file_path)
