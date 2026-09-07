def generate_reasoning_plan(query: str, context: str, llm) -> str:
    """Single-pass: ask the model to reason step-by-step over the retrieved slides."""
    plan_prompt = f"""
You are an academic tutor analyzing lecture material.

Query: {query}
Retrieved Slides:
{context}

Task: Think through the problem step-by-step using ONLY the slides.
List the key facts from the slides that are relevant to the query, then outline
a short logic plan for answering. Do NOT write the final answer yet.
"""
    raw = llm.invoke(plan_prompt)
    return raw.content if hasattr(raw, 'content') else str(raw)


def synthesize_final_answer(query: str, context_text: str, plan: str, llm) -> str:
    """Uses the reasoning plan to produce the final concise answer."""
    answer_prompt = f"""
You are a helpful academic assistant answering a student's question using retrieved lecture slides.

Rules:
- Base your answer strictly on the facts confirmed in the Reasoning Plan and Slides Context.
- Keep the answer direct and concise.

Slides Context:
{context_text}

Reasoning Plan:
{plan}

Question: {query}
Answer:
"""
    raw = llm.invoke(answer_prompt)
    return raw.content if hasattr(raw, 'content') else str(raw)
