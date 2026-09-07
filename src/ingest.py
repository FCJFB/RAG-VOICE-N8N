import re
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config import DB_DIR, OLLAMA_HOST, EMBED_MODEL

def extract_lecture_metadata(filename: str) -> Dict[str, Any]:
    """
    Extracts course_id and lecture_num from flexible filename patterns.
    Examples: 'CS101_L03.pdf', 'ImageProcessing01.pdf', 'Lecture_02_Vision.pdf'
    """
    filename_clean = Path(filename).stem
    
    # Check for standard course codes (e.g. CS101, MATH200)
    course_match = re.search(r'([A-Za-z]{2,4}\d{3})', filename_clean)
    
    # Check for numbers associated with lecture/module
    lecture_match = re.search(r'(?:Lecture|L|Module|Processing)[_\-\s]*(\d+)', filename_clean, re.IGNORECASE)
    
    if course_match:
        course_id = course_match.group(1).upper()
    else:
        # Fallback: extract main word prefix (e.g., ImageProcessing01 -> IMAGEPROCESSING)
        alpha_prefix = re.sub(r'[\d_\-\s]+$', '', filename_clean)
        course_id = alpha_prefix.upper() if alpha_prefix else "GENERAL"

    return {
        "course_id": course_id,
        "lecture_num": int(lecture_match.group(1)) if lecture_match else 1,
        "source_file": Path(filename).name
    }

def load_and_parse_pdf(file_path: str) -> List[Document]:
    """
    Parses a PDF slide deck page-by-page, attaching slide/page metadata.
    """
    documents = []
    base_meta = extract_lecture_metadata(file_path)
    
    with pdfplumber.open(file_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                continue
            
            slide_meta = base_meta.copy()
            slide_meta["slide_num"] = idx
            
            doc = Document(
                page_content=text.strip(),
                metadata=slide_meta
            )
            documents.append(doc)
            
    return documents

def process_and_ingest_slides(data_dir: Path = None):
    """
    Processes all PDF files in data directory, chunks them, and stores in ChromaDB.
    """
    from src.config import DATA_DIR
    target_dir = data_dir or DATA_DIR
    pdf_files = list(target_dir.glob("*.pdf")) + list(target_dir.parent.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{target_dir}'. Place your lecture PDFs there.")
        return

    all_slide_docs = []
    for pdf_path in pdf_files:
        print(f"Parsing: {pdf_path.name}...")
        slide_docs = load_and_parse_pdf(str(pdf_path))
        all_slide_docs.extend(slide_docs)
        print(f"  -> Extracted {len(slide_docs)} slides/pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunked_docs = text_splitter.split_documents(all_slide_docs)
    print(f"Total chunked documents generated: {len(chunked_docs)}")

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_HOST
    )

    print(f"Persisting vector store to '{DB_DIR}'...")
    db = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=str(DB_DIR)
    )
    print("Ingestion complete!")

if __name__ == "__main__":
    process_and_ingest_slides()