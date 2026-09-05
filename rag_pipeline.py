import os
import re
from typing import Literal
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM, ChatOllama

# --- CONFIGURATION ---
DB_DIR = "./chroma_db"
DOC_PATH = "knowledge.txt"
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2"

# Distance threshold for mxbai-embed-large in ChromaDB
# Lower = Stricter, Higher = More lenient
MAX_COSINE_DISTANCE = 350.0


# --- LAYER 1: INTENT ROUTER SCHEMA ---
class RouteQuery(BaseModel):
    destination: Literal["GREETING", "KNOWLEDGE_QUERY", "OUT_OF_SCOPE"] = Field(
        description="Route input: GREETING for small talk, KNOWLEDGE_QUERY for anything related to CachyOS or Linux, OUT_OF_SCOPE for completely unrelated topics (cooking, movies, etc.)."
    )

def classify_intent(query: str) -> str:
    """Layer 1: Classifies query intent using structured Pydantic output."""
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    router_llm = llm.with_structured_output(RouteQuery)

    prompt = f"""You are an intent classifier for a CachyOS documentation assistant.
Classify the user input into exactly one route:
- GREETING: Small talk, hellos, goodbyes, or polite remarks.
- KNOWLEDGE_QUERY: Questions specifically seeking technical or feature information about CachyOS, Linux kernels, schedulers, or desktop environments.
- OUT_OF_SCOPE: Any question completely unrelated to system documentation (e.g., cooking, baking, general trivia, entertainment).

User Input: {query}"""

    try:
        decision = router_llm.invoke(prompt)
        return decision.destination
    except Exception:
        return "KNOWLEDGE_QUERY"


# --- VECTOR DATABASE MANAGEMENT ---
def get_or_create_vector_db():
    embedding_function = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("✓ Existing vector database found. Loading from disk...")
        return Chroma(
            persist_directory=DB_DIR, 
            embedding_function=embedding_function
        )

    print("! No database found. Building new database...")
    if not os.path.exists(DOC_PATH):
        sample_data = """
        CachyOS is an Arch Linux-based distribution optimized for performance and ease of use.
        Key features of CachyOS:
        - Uses custom kernels compiled with x86-64-v3 and x86-64-v4 CPU instruction sets.
        - Features the BORE (Burst-Oriented Response Enhancer) CPU scheduler by default.
        - Includes a custom package management tool named cachyos-rate-mirrors for mirror selection.
        - Supports KDE Plasma, GNOME, Hyprland, and XFCE desktop environments.
        """
        with open(DOC_PATH, "w", encoding="utf-8") as f:
            f.write(sample_data.strip())

    loader = TextLoader(DOC_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=DB_DIR
    )
    print("✓ Vector database built and saved to disk.")
    return vector_db


# --- TWO-PASS REASONING (CHAIN-OF-THOUGHT) ---
def generate_reasoning_plan(query: str, context_text: str, llm: OllamaLLM) -> str:
    """Pass 1: Forces Llama 3.2 to generate explicit reasoning steps ONLY."""
    plan_prompt = f"""
    You are a logical reasoning engine. Analyze the context and question below.

    Tasks:
    1. Verify if the context contains facts relevant to the question.
    2. List the relevant key facts from the context.
    3. Outline a brief logic plan for answering the question.
    4. If facts are missing, write EXACTLY: "STATUS: INSUFFICIENT CONTEXT". Otherwise write "STATUS: SUFFICIENT CONTEXT".

    Do NOT write the final user answer yet. Only provide the reasoning plan.

    Context:
    {context_text}

    Question: {query}

    Reasoning Plan:
    """
    return llm.invoke(plan_prompt).strip()


def synthesize_final_answer(query: str, context_text: str, plan: str, llm: OllamaLLM) -> str:
    """Pass 2: Uses the Reasoning Plan from Pass 1 to generate the final response."""
    
    # Check if Pass 1 flagged missing context directly in Python
    if "STATUS: INSUFFICIENT CONTEXT" in plan:
        return "I do not have enough information in my knowledge base to answer that."

    answer_prompt = f"""
    You are a helpful assistant. Use the Context and Reasoning Plan below to answer the user's question.

    Rules:
    - Base your answer strictly on the facts confirmed in the Context and Reasoning Plan.
    - Keep the answer direct and concise.

    Context:
    {context_text}

    Reasoning Plan:
    {plan}

    Question: {query}
    Answer:
    """
    return llm.invoke(answer_prompt).strip()

# --- MAIN RAG EXECUTION ---
def ask_rag_with_threshold(query: str, vector_db: Chroma):
    """Layer 2 (Vector Thresholding) + Two-Pass CoT Execution."""
    results = vector_db.similarity_search_with_score(query, k=4)

    if not results:
        print("\n=== RAG RESPONSE ===")
        print("I do not have enough information in my knowledge base.")
        return

    best_doc, top_score = results[0]
    print(f"\n[Vector Distance Score: {top_score:.2f}]")

    # Layer 2 Guardrail: Distance check
    if top_score > MAX_COSINE_DISTANCE:
        print("\n=== RAG RESPONSE ===")
        print("I do not have enough relevant information in my knowledge base to answer that.")
        return

    context_text = "\n---\n".join([doc.page_content for doc, _ in results])
    llm = OllamaLLM(model=LLM_MODEL)

    # --- PASS 1: REASONING & PLANNING ---
    reasoning_plan = generate_reasoning_plan(query, context_text, llm)
    
    print("\n┌─── LLM REASONING STEPS (PASS 1) ──────────────────────────────")
    print(reasoning_plan)
    print("└───────────────────────────────────────────────────────────────")

    # --- PASS 2: FINAL SYNTHESIS ---
    final_response = synthesize_final_answer(query, context_text, reasoning_plan, llm)

    print("\n=== FINAL RAG RESPONSE (PASS 2) ===")
    print(final_response)


# --- INTERACTIVE CLI LOOP ---
if __name__ == "__main__":
    db = get_or_create_vector_db()

    print("\n--- Reasoning-Enabled Local RAG CLI (type 'exit' or 'q' to quit) ---")
    while True:
        user_query = input("\nAsk a question: ").strip()
        if user_query.lower() in ["exit", "q"]:
            print("Goodbye!")
            break

        if user_query:
            # Layer 1: Intent Routing
            route = classify_intent(user_query)
            print(f"[Router Classification: {route}]")

            if route == "GREETING":
                llm = OllamaLLM(model=LLM_MODEL)
                print("\n=== RESPONSE ===")
                print(llm.invoke(f"Respond warmly and concisely to this greeting: {user_query}"))

            elif route == "OUT_OF_SCOPE":
                print("\n=== RESPONSE ===")
                print("I am a CachyOS assistant. I cannot answer questions outside of my documentation context.")

            else:  # KNOWLEDGE_QUERY
                ask_rag_with_threshold(user_query, db)