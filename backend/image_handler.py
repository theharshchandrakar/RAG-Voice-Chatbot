"""
Image Handler Module
Handles image upload, Gemini Vision analysis, and ChromaDB storage.
"""

import io
from PIL import Image

from embeddings import chunk_and_store
from ocr_utils import analyze_image_with_retry


async def process_image(
    file_bytes: bytes,
    filename: str,
    image_collection,
    gemini_model
):
    """
    Process uploaded image: analyze with Gemini Vision, chunk, and store.
    
    Args:
        file_bytes: Raw image file bytes
        filename: Original filename
        image_collection: ChromaDB collection for image storage
        gemini_model: Gemini Vision model instance
        
    Returns:
        dict: Status with message and chunk count
        
    Raises:
        ValueError: If no text extracted from image
    """
    print(f"📷 Uploading image: {filename}")
    print(f"File size: {len(file_bytes)} bytes")
    
    # Open image from bytes directly without temp file
    img = Image.open(io.BytesIO(file_bytes))
    print(f"Image opened successfully: {img.format} {img.size}")
    
    print("Calling analyze_image_with_retry...")
    extracted_text = analyze_image_with_retry(img, gemini_model)
    print(f"Extracted text length: {len(extracted_text)}")
    
    if not extracted_text.strip():
        raise ValueError("No text extracted from image")
    
    # Chunk and store using reusable utility
    chunk_count, already_exists = chunk_and_store(extracted_text, image_collection, source=filename, chunk_size=1000, chunk_overlap=200)
    
    if already_exists:
        return {
            "status": "success",
            "message": f"Image '{filename}' already exists in database. Skipped processing.",
            "duplicate": True
        }
    print(f"Successfully stored image analysis in ChromaDB")
    
    return {
        "status": "success",
        "message": f"Processed {chunk_count} chunks from image.",
        "duplicate": False
    }
