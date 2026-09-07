from typing import Optional, Dict, Any
from src.vector_store import query_vector_store
from src.reasoning import generate_reasoning_plan, synthesize_final_answer
from src.config import MAX_COSINE_DISTANCE, LLM_MODEL, OLLAMA_HOST
from langchain_community.llms import Ollama

class RAGPipeline:
    def __init__(self):
        self.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_HOST)

    def run(
        self,
        query: str,
        course_id: Optional[str] = None,
        lecture_num: Optional[int] = None
    ) -> Dict[str, Any]:
        results = query_vector_store(
            query=query,
            k=4,
            course_id=course_id,
            lecture_num=lecture_num
        )

        if results and len(results) > 0:
            best_doc, top_score = results[0]
            top_score = float(top_score)
        else:
            best_doc, top_score = None, float('inf')

        is_relevant = top_score <= MAX_COSINE_DISTANCE

        if not is_relevant:
            fallback_prompt = f"User Query: {query}\nProvide a concise general response:"
            raw = self.llm.invoke(fallback_prompt)
            res_text = raw.content if hasattr(raw, 'content') else str(raw)
            return {
                "query": query,
                "response": res_text.strip(),
                "vector_distance": round(top_score, 2) if top_score != float('inf') else None,
                "reasoning_plan": None,
                "is_relevant": False,
                "sources": []
            }

        # Extract context and source metadata
        context_text = "\n---\n".join([doc.page_content for doc, _ in results])
        sources = [
            {
                "file": doc.metadata.get("source_file", "Unknown"),
                "slide": doc.metadata.get("slide_num", "N/A"),
                "course": doc.metadata.get("course_id", "GENERAL")
            }
            for doc, _ in results
        ]

        reasoning_plan = str(generate_reasoning_plan(query, context_text, self.llm))
        final_answer = str(synthesize_final_answer(query, context_text, reasoning_plan, self.llm))

        return {
            "query": query,
            "response": final_answer,
            "vector_distance": round(top_score, 2),
            "reasoning_plan": reasoning_plan,
            "is_relevant": True,
            "sources": sources
        }