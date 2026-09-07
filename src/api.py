from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from src.pipeline import RAGPipeline

app = FastAPI(
    title="Academic Lecture RAG API",
    description="Local RAG API with metadata filtering for multi-lecture materials.",
    version="2.0.0"
)

pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is spatial filtering?")
    course_id: Optional[str] = Field(None, example="IMAGEPROCESSING")
    lecture_num: Optional[int] = Field(None, example=1)

class QueryResponse(BaseModel):
    query: str
    response: str
    vector_distance: Optional[float] = None
    reasoning_plan: Optional[str] = None
    is_relevant: bool
    sources: List[Dict[str, Any]] = []

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": "llama3.2"}

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        result = pipeline.run(
            query=payload.query,
            course_id=payload.course_id,
            lecture_num=payload.lecture_num
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal pipeline error: {str(e)}")