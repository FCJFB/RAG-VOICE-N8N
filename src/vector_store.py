from typing import List, Tuple, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from src.config import DB_DIR, OLLAMA_HOST, EMBED_MODEL

def get_vector_store() -> Chroma:
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_HOST
    )
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings
    )

def query_vector_store(
    query: str,
    k: int = 4,
    course_id: Optional[str] = None,
    lecture_num: Optional[int] = None
) -> List[Tuple[Document, float]]:
    """
    Queries ChromaDB with optional metadata filtering for course_id and lecture_num.
    """
    db = get_vector_store()
    
    # Build Chroma metadata filter
    where_filter: Dict[str, Any] = {}
    if course_id:
        where_filter["course_id"] = course_id.upper()
    if lecture_num is not None:
        where_filter["lecture_num"] = lecture_num

    # Chroma expects None if no filter conditions are applied
    filter_dict = where_filter if where_filter else None

    results = db.similarity_search_with_score(
        query,
        k=k,
        filter=filter_dict
    )
    return results