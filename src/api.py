from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from src.pipeline import RAGPipeline

# Name the instance 'app'
app = FastAPI(
    title="CachyOS Local RAG API",
    description="Privacy-focused, local RAG pipeline with CoT reasoning and distance thresholding.",
    version="1.0.0"
)

# Pipeline initialization
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
        # Run vector similarity search
        results = pipeline.db.similarity_search_with_score(payload.query, k=4)
        best_doc, top_score = (results[0][0], results[0][1]) if results else (None, float('inf'))
        
        is_relevant = top_score <= pipeline.db._collection.count() and top_score <= 350.0  # MAX_COSINE_DISTANCE check
        
        if not is_relevant:
            # Fallback path for greetings / out-of-scope queries
            fallback_prompt = f"""You are a specialized CachyOS Linux documentation assistant.

INSTRUCTIONS:
- For greetings, farewells, or polite conversational remarks, respond warmly and concisely.
- For non-technical or out-of-scope questions (e.g., trivia, entertainment, cooking), politely state that you only assist with CachyOS Linux documentation.

User Input: {payload.query}
Response:"""
            response = pipeline.llm.invoke(fallback_prompt).strip()
            return QueryResponse(
                query=payload.query,
                response=response,
                vector_distance=round(top_score, 2) if top_score != float('inf') else None,
                reasoning_plan=None,
                is_relevant=False
            )

        # Relevant technical query path -> Two-Pass Reasoning
        context_text = "\n---\n".join([doc.page_content for doc, _ in results])
        from src.reasoning import generate_reasoning_plan, synthesize_final_answer
        
        reasoning_plan = generate_reasoning_plan(payload.query, context_text, pipeline.llm)
        final_answer = synthesize_final_answer(payload.query, context_text, reasoning_plan, pipeline.llm)

        return QueryResponse(
            query=payload.query,
            response=final_answer,
            vector_distance=round(top_score, 2),
            reasoning_plan=reasoning_plan,
            is_relevant=True
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal pipeline error: {str(e)}")