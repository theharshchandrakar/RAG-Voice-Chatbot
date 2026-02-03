"""
Video Processing Module
Handles video transcription and storage
"""

from embeddings import chunk_and_store


async def process_video(file_bytes: bytes, filename: str, groq_client, video_collection):
    """
    Transcribe video using Groq Whisper and store chunks in ChromaDB.
    
    Args:
        file_bytes: Video file bytes
        filename: Name of the video file
        groq_client: Groq client instance
        video_collection: ChromaDB video collection
    
    Returns:
        Dict with status and message
    """
    # Transcribe with Groq Whisper
    transcription = groq_client.audio.transcriptions.create(
        file=(filename, file_bytes),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
    )
    transcript = transcription.text
    
    # Chunk and store in ChromaDB
    chunk_count, already_exists = chunk_and_store(transcript, video_collection, source=filename)
    
    if already_exists:
        return {
            "status": "success",
            "message": f"Video '{filename}' already exists in database. Skipped processing.",
            "duplicate": True
        }
    
    return {
        "status": "success",
        "message": f"Processed {chunk_count} chunks from video.",
        "duplicate": False
    }
