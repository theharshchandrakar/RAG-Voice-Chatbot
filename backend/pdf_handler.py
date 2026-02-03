"""
PDF Handler Module
Handles PDF upload, text extraction, OCR detection, and ChromaDB storage.
"""

import os
import tempfile
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import chunk_and_store
from ocr_utils import is_scanned_pdf, extract_text_with_ocr


async def process_pdf(
    file_bytes: bytes,
    filename: str,
    pdf_collection,
    groq_vision_client=None,
    max_pdf_size_mb: int = 10
):
    """
    Process uploaded PDF file: extract text, perform OCR if needed, chunk, and store.
    
    Args:
        file_bytes: Raw PDF file bytes
        filename: Original filename
        pdf_collection: ChromaDB collection for PDF storage
        groq_vision_client: Optional Groq Vision client for OCR
        max_pdf_size_mb: Maximum allowed PDF size in MB
        
    Returns:
        dict: Status with message, chunk count, and OCR usage flag
        
    Raises:
        ValueError: If PDF is too large or no text extracted
    """
    # Enforce upload size limit
    if len(file_bytes) > max_pdf_size_mb * 1024 * 1024:
        raise ValueError(f"PDF too large (> {max_pdf_size_mb} MB)")
    
    # Save to temp file (PyPDFLoader requires file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    try:
        print(f"📄 Processing PDF: {filename}")
        
        # First, try simple text extraction with PyPDFLoader
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        pdf_text = "\n\n".join([doc.page_content for doc in documents])
        page_count = len(documents)
        
        # Check if OCR is needed with smarter heuristic
        needs_ocr = is_scanned_pdf(pdf_text, page_count=page_count)
        
        if needs_ocr:
            print(f"⚠️  Detected scanned/image-based PDF (only {len(pdf_text.strip())} chars extracted)")
            
            # Try OCR with Groq Vision if client available
            if groq_vision_client:
                ocr_text = extract_text_with_ocr(tmp_path, groq_vision_client)
                if ocr_text and len(ocr_text.strip()) > len(pdf_text.strip()):
                    print(f"✅ OCR extracted {len(ocr_text)} characters (better than {len(pdf_text)})")
                    pdf_text = ocr_text
                else:
                    print(f"⚠️  OCR did not improve extraction, using original")
            else:
                print(f"⚠️  Groq Vision client not available for OCR, using basic extraction")
        else:
            print(f"✅ Text-based PDF detected ({len(pdf_text)} characters)")
        
        if not pdf_text.strip():
            raise ValueError("No text extracted from PDF. Please ensure the file is readable.")
        
        # Chunk and store using reusable utility
        chunk_count, already_exists = chunk_and_store(pdf_text, pdf_collection, source=filename, chunk_size=1000, chunk_overlap=200)
        
        if already_exists:
            return {
                "status": "success",
                "message": f"PDF '{filename}' already exists in database. Skipped processing.",
                "duplicate": True
            }
        
        ocr_status = " (OCR used)" if needs_ocr and groq_vision_client else ""
        return {
            "status": "success",
            "message": f"Processed {chunk_count} chunks from PDF{ocr_status}.",
            "ocr_used": bool(needs_ocr and groq_vision_client),
            "duplicate": False
        }
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)