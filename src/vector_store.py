import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.config import DB_DIR, DOC_PATH, EMBED_MODEL

def get_or_create_vector_db() -> Chroma:
    """Loads existing ChromaDB or initializes a new one from source documents."""
    embedding_function = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("✓ Existing vector database found. Loading from disk...")
        return Chroma(
            persist_directory=str(DB_DIR), 
            embedding_function=embedding_function
        )

    print("! No database found. Building new vector database...")
    
    # Create default knowledge file if missing
    if not os.path.exists(DOC_PATH):
        os.makedirs(os.path.dirname(DOC_PATH), exist_ok=True)
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

    loader = TextLoader(str(DOC_PATH))
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=str(DB_DIR)
    )
    print("✓ Vector database initialized and saved to disk.")
    return vector_db