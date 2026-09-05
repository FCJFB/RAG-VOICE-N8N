from typing import Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from src.config import LLM_MODEL

class RouteQuery(BaseModel):
    destination: Literal["GREETING", "KNOWLEDGE_QUERY", "OUT_OF_SCOPE"] = Field(
        description="Classification decision for user input."
    )

def classify_intent(query: str) -> str:
    """Classifies user intent using a structured Pydantic LLM call."""
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    router_llm = llm.with_structured_output(RouteQuery)
    
    prompt = f"""You are an intent router for a local Linux assistant.
Classify the user input into exactly ONE of these categories:

1. GREETING: Conversational openings, hellos, hi, hey, good morning, good evening, polite small talk, or farewells.
   Examples: "hello", "hi", "hey there", "how are you", "bye"

2. KNOWLEDGE_QUERY: Questions asking for technical details, definitions, commands, features, or help regarding CachyOS, Arch Linux, kernels, schedulers, or desktop environments.
   Examples: "what is cachyos", "what scheduler does cachyos use", "kernel options"

3. OUT_OF_SCOPE: Requests completely unrelated to Linux or conversation (e.g., recipes, cooking, general trivia, movies, math problems).
   Examples: "how to bake a cake", "who directed inception", "2+2"

User Input: {query}"""

    try:
        decision = router_llm.invoke(prompt)
        return decision.destination
    except Exception:
        return "KNOWLEDGE_QUERY"