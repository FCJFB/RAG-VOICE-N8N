from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from src.config import MAX_COSINE_DISTANCE
from src.pipeline import RAGPipeline
from src.reasoning import generate_reasoning_plan, synthesize_final_answer

app = FastAPI(
    title="CachyOS Local RAG API",
    description="Privacy-focused, local RAG pipeline with CoT reasoning and distance thresholding.",
    version="1.0.0"
)

pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    query: str = Field(..., example="what scheduler does cachyos use?")

class QueryResponse(BaseModel):
    query: str
    response: str
    vector_distance: Optional[float] = None
    reasoning_plan: Optional[str] = None
    is_relevant: bool

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": "llama3.2"}

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        results = pipeline.db.similarity_search_with_score(payload.query, k=4)
        
        # Safely unpack mock or real search results
        if results and len(results) > 0:
            best_doc, top_score = results[0]
            top_score = float(top_score)
        else:
            best_doc, top_score = None, float('inf')
        
        is_relevant = top_score <= MAX_COSINE_DISTANCE
        
        if not is_relevant:
            fallback_prompt = f"User Input: {payload.query}\nResponse:"
            raw_response = pipeline.llm.invoke(fallback_prompt)
            
            # Extract plain text string safely
            if hasattr(raw_response, 'content'):
                response_text = str(raw_response.content)
            else:
                response_text = str(raw_response)
            
            return QueryResponse(
                query=payload.query,
                response=response_text.strip(),
                vector_distance=round(top_score, 2) if top_score != float('inf') else None,
                reasoning_plan=None,
                is_relevant=False
            )

        context_text = "\n---\n".join([str(getattr(doc, 'page_content', '')) for doc, _ in results])
        reasoning_plan = str(generate_reasoning_plan(payload.query, context_text, pipeline.llm))
        final_answer = str(synthesize_final_answer(payload.query, context_text, reasoning_plan, pipeline.llm))

        return QueryResponse(
            query=payload.query,
            response=final_answer,
            vector_distance=round(top_score, 2),
            reasoning_plan=reasoning_plan,
            is_relevant=True
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal pipeline error: {str(e)}")