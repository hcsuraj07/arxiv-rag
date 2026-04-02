from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# get absolute path to src/ and add it to Python's search path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

# now import directly — no relative path confusion
import pipeline

app = FastAPI(
    title="arXiv RAG API",
    description="Ask questions about research papers using RAG + GPT-4o",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

class SourceModel(BaseModel):
    title: str
    score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]
    usage: dict

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "arXiv RAG API is running"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        result = pipeline.run_query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {
        "message": "arXiv RAG API",
        "docs": "http://localhost:8000/docs",
        "health": "http://localhost:8000/health"
    }