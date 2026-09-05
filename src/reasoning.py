from langchain_ollama import OllamaLLM
from src.config import LLM_MODEL

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