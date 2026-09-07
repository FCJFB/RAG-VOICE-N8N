from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.config import DB_DIR, OLLAMA_HOST, EMBED_MODEL

def inspect_vector_store():
    print(f"Connecting to ChromaDB at: {DB_DIR}")
    
    if not Path(DB_DIR).exists():
        print("Error: Vector DB directory does not exist yet. Run 'make ingest' first.")
        return

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_HOST
    )

    db = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings
    )

    # Access the underlying raw collection
    collection = db._collection
    total_chunks = collection.count()

    print(f"\n================ VECTOR DB STATUS ================")
    print(f"Total Document Chunks Ingested: {total_chunks}")
    
    if total_chunks == 0:
        print("The vector store is currently empty.")
        return

    # Fetch all document metadata
    raw_data = collection.get(include=["metadatas", "documents"])
    metadatas = raw_data.get("metadatas", [])
    
    # Extract unique source files, course IDs, and lectures
    sources = sorted(list(set(meta.get("source_file", "Unknown") for meta in metadatas)))
    courses = sorted(list(set(meta.get("course_id", "Unknown") for meta in metadatas)))
    
    print("\n--- Ingested Courses ---")
    for course in courses:
        print(f"  • {course}")

    print("\n--- Ingested Source Files ---")
    for src in sources:
        # Count chunks per source file
        chunk_count = sum(1 for meta in metadatas if meta.get("source_file") == src)
        print(f"  • {src} ({chunk_count} chunks)")

    print("\n--- Sample Document Entry ---")
    if metadatas:
        sample_doc = raw_data["documents"][0]
        sample_meta = metadatas[0]
        print(f"Metadata: {sample_meta}")
        print(f"Content Preview:\n  \"{sample_doc[:200]}...\"")
        
    print("==================================================")

if __name__ == "__main__":
    inspect_vector_store()