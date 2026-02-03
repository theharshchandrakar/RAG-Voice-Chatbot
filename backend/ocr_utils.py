"""
OCR Utilities Module
Handles OCR operations for scanned PDFs and images
"""

import io
import base64
from PIL import Image


def encode_image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string for Groq Vision API.
    
    Args:
        image: PIL Image object
    
    Returns:
        Base64 encoded image string with data URI prefix
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def is_scanned_pdf(pdf_text: str, page_count: int | None = None, 
                   min_total_chars: int = 300, 
                   min_chars_per_page: int = 120, 
                   min_alpha_ratio: float = 0.2) -> bool:
    """
    Heuristic to decide if a PDF likely needs OCR.
    
    Args:
        pdf_text: Extracted text from PDF
        page_count: Number of pages in PDF
        min_total_chars: Minimum total characters threshold
        min_chars_per_page: Minimum characters per page threshold
        min_alpha_ratio: Minimum alphabetic character ratio
    
    Returns:
        True if PDF likely needs OCR, False otherwise
    """
    text = pdf_text.strip()
    total_chars = len(text)

    if total_chars == 0:
        return True

    # Average chars per page if page_count known
    if page_count and page_count > 0:
        chars_per_page = total_chars / page_count
    else:
        chars_per_page = total_chars

    # Alphabetic character ratio
    alpha_chars = sum(ch.isalpha() for ch in text)
    alpha_ratio = alpha_chars / total_chars if total_chars else 0

    return (
        total_chars < min_total_chars
        or chars_per_page < min_chars_per_page
        or alpha_ratio < min_alpha_ratio
    )


def extract_text_with_ocr(pdf_path: str, groq_vision_client, 
                         model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> str:
    """
    Extract text from scanned PDF using Groq Vision OCR.
    
    Args:
        pdf_path: Path to PDF file
        groq_vision_client: Groq Vision client instance
        model: Groq Vision model name
    
    Returns:
        Extracted text from all pages
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("⚠️  PyMuPDF not installed. Install with: pip install PyMuPDF")
        return ""

    try:
        print("🔍 Using OCR (Groq Vision) for scanned PDF...")
        doc = fitz.open(pdf_path)
        full_text = []
        total_pages = doc.page_count
        
        # Render at ~300 DPI by applying a zoom factor of 300/72
        zoom = 300 / 72
        mat = fitz.Matrix(zoom, zoom)

        for page_num in range(total_pages):
            print(f"   Processing page {page_num+1}/{total_pages} with OCR...")
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            base64_img = encode_image_to_base64(img)

            response = groq_vision_client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Extract all text from this document page accurately, including any handwriting. Return only the extracted text."
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": base64_img}
                        }
                    ]
                }],
                temperature=0.1,
                max_tokens=4096,
                stream=False
            )
            page_text = response.choices[0].message.content.strip()
            full_text.append(f"--- Page {page_num+1} ---\n{page_text}")

        doc.close()
        print(f"✅ OCR completed for {total_pages} pages")
        return "\n\n".join(full_text)

    except Exception as e:
        print(f"⚠️  OCR failed: {str(e)}")
        return ""


def analyze_image_with_retry(image: Image.Image, gemini_model, retries: int = 3) -> str:
    """
    Analyze image via Gemini Vision and return text; retry on 429/503.
    
    Args:
        image: PIL Image object
        gemini_model: Gemini model instance
        retries: Number of retry attempts
    
    Returns:
        Extracted/analyzed text from image
    """
    if not gemini_model:
        # Fallback: Return basic image info without Gemini
        print("⚠️  Gemini not available, using fallback image description")
        return f"Image uploaded: {image.format} format, size: {image.size[0]}x{image.size[1]} pixels. Note: Google Gemini Vision is not configured. To enable AI-powered image analysis, add a valid GOOGLE_API_KEY to your .env file."
    
    prompt = "Analyze this image in detail. Extract text, layout, objects, and data. Also describe the image based on your understanding."
    last_err = None
    
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1}: Calling Gemini API...")
            resp = gemini_model.generate_content([prompt, image])
            text = getattr(resp, "text", "") or ""
            print(f"Gemini response: {text[:100]}...")
            return text
        except Exception as e:
            last_err = e
            print(f"Gemini error on attempt {attempt + 1}: {str(e)}")
            
            # Handle rate limiting or temporary errors
            if "503" in str(e) or "429" in str(e):
                import time
                time.sleep(2)
                continue
            
            # For other errors (404, invalid model, etc.), use fallback
            elif "404" in str(e) or "not found" in str(e).lower() or attempt == retries - 1:
                print(f"⚠️  Gemini API error, using fallback description")
                return f"Image uploaded: {image.format} format, size: {image.size[0]}x{image.size[1]} pixels. Note: Gemini Vision API error - {str(e)[:100]}. Image stored but AI analysis unavailable."
            
    # Final fallback if all retries exhausted
    return f"Image uploaded: {image.format} format, size: {image.size[0]}x{image.size[1]} pixels. Note: Gemini API unavailable after {retries} attempts."
