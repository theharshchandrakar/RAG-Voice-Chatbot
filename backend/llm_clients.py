"""
LLM Client Interaction Module
Handles communication with Groq, Ollama, and Gemini APIs
"""


def send_groq_chat(groq_client, groq_model: str, messages: list[dict], temperature: float = 0.7, max_tokens: int = 600) -> str:
    """
    Send chat request to Groq API.
    
    Args:
        groq_client: Groq client instance
        groq_model: Model name
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
    
    Returns:
        Response text
    """
    response = groq_client.chat.completions.create(
        model=groq_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def send_ollama_chat(ollama_client, ollama_model: str, messages: list[dict]) -> str:
    """
    Send chat request to Ollama API.
    
    Args:
        ollama_client: Ollama client instance
        ollama_model: Model name
        messages: List of message dictionaries
    
    Returns:
        Response text
    """
    try:
        if not ollama_client:
            return "Ollama Fallback Error: Client unavailable"
        response = ollama_client.chat(model=ollama_model, messages=messages, stream=False)
        return response['message']['content']
    except Exception as e:
        return f"Ollama Fallback Error: {str(e)}"
