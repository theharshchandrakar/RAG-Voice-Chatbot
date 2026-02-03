"""
Embedding and Context Retrieval Module
Handles text chunking and context retrieval from ChromaDB collections
"""

import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_and_store(text: str, collection, source: str = None, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Chunk text and store in ChromaDB collection.
    Checks for duplicate sources before storing.
    
    Args:
        text: Text to chunk
        collection: ChromaDB collection to store in
        source: Source identifier (e.g., filename) to check for duplicates
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        Tuple of (chunks_created, already_exists)
        - chunks_created: Number of chunks created (0 if duplicate)
        - already_exists: Boolean indicating if content already exists
    """
    # Check if this source already exists in the collection
    if source:
        try:
            existing = collection.get(where={"source": source})
            if existing and len(existing.get('ids', [])) > 0:
                return 0, True  # Already exists, return 0 chunks and True flag
        except Exception as e:
            # If collection is empty or error occurs, proceed with storing
            print(f"Note: Could not check for duplicates: {e}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    
    # Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    # Prepare metadata with source information
    metadatas = [{"source": source} if source else {} for _ in chunks]
    
    # Store in ChromaDB
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    
    return len(chunks), False


def retrieve_context(query: str, collection, n_results: int = 5):
    """
    Retrieve relevant context from ChromaDB collection.
    
    Args:
        query: Query text
        collection: ChromaDB collection to search
        n_results: Number of results to retrieve
    
    Returns:
        Concatenated context string
    """
    try:
        # Check if collection has any documents
        count = collection.count()
        print(f"📊 Collection has {count} documents")
        
        if count == 0:
            print("⚠️  Collection is empty - no documents to search")
            return ""
        
        results = collection.query(query_texts=[query], n_results=min(n_results, count))
        documents = results.get("documents", [[]])[0]
        
        # Log retrieval results for debugging
        print(f"🔍 Retrieved {len(documents)} documents for query: '{query[:50]}...'")
        if documents:
            print(f"📝 First document preview: {documents[0][:200]}...")
        
        context = "\n\n".join(documents)
        return context
    except Exception as e:
        print(f"❌ Error retrieving context: {str(e)}")
        return ""
