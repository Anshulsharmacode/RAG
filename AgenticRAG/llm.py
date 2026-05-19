from typing import Any


def get_chat_model(model: str = "llama3.2:1b", temperature: float = 0.0) -> Any:
    """
    Return an Ollama chat model with compatibility fallback.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama

    return ChatOllama(model=model, temperature=temperature)


def get_embedding_model(model: str = "nomic-embed-text:latest") -> Any:
    """
    Return an Ollama embedding model with compatibility fallback.
    """
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        from langchain_community.embeddings import OllamaEmbeddings

    return OllamaEmbeddings(model=model)
