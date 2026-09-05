from langchain_ollama import OllamaLLM
from src.config import MAX_COSINE_DISTANCE, LLM_MODEL
from src.vector_store import get_or_create_vector_db
from src.reasoning import generate_reasoning_plan, synthesize_final_answer

class RAGPipeline:
    def __init__(self):
        self.db = get_or_create_vector_db()
        self.llm = OllamaLLM(model=LLM_MODEL)

    def process_query(self, query: str):
        # 1. Always run similarity search first
        results = self.db.similarity_search_with_score(query, k=4)
        
        best_doc, top_score = (results[0][0], results[0][1]) if results else (None, float('inf'))
        print(f"\n[Vector Search Distance Score: {top_score:.2f}]")

        # 2. Check mathematical relevance against threshold
        is_relevant = top_score <= MAX_COSINE_DISTANCE

        # 3. IF IRRELEVANT (Greetings, Small Talk, Out-of-Scope Questions)
        if not is_relevant:
            unified_fallback_prompt = f"""You are a helpful, specialized CachyOS Linux documentation assistant.

INSTRUCTIONS:
- For greetings, farewells, or polite conversational remarks, respond warmly and concisely.
- For non-technical or out-of-scope questions (e.g., trivia, entertainment, cooking), politely state that you only assist with CachyOS Linux documentation.

User Input: {query}
Response:"""

            response = self.llm.invoke(unified_fallback_prompt).strip()
            print("\n=== RESPONSE ===")
            print(response)
            return

        # 4. IF RELEVANT (Technical CachyOS Queries) -> Run Two-Pass Reasoning
        context_text = "\n---\n".join([doc.page_content for doc, _ in results])

        reasoning_plan = generate_reasoning_plan(query, context_text, self.llm)
        print("\n┌─── LLM REASONING STEPS (PASS 1) ──────────────────────────────")
        print(reasoning_plan)
        print("└───────────────────────────────────────────────────────────────")

        final_response = synthesize_final_answer(query, context_text, reasoning_plan, self.llm)
        print("\n=== FINAL RAG RESPONSE (PASS 2) ===")
        print(final_response)


if __name__ == "__main__":
    pipeline = RAGPipeline()

    print("\n--- Unified RAG CLI (type 'exit' or 'q' to quit) ---")
    while True:
        user_query = input("\nAsk a question: ").strip()
        if user_query.lower() in ["exit", "q"]:
            print("Goodbye!")
            break

        if user_query:
            pipeline.process_query(user_query)