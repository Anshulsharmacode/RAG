from typing import List
import numpy as np
import requests


class NomicEmbeddingModel:
    def __init__(self, host: str = "http://localhost:11434", model: str = "nomic-embed-text:latest"):
        self.host = host.rstrip("/")
        self.model = model
        self.endpoint = f"{self.host}/api/embeddings"

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")

        embeddings = []

        for text in texts:
            payload = {
                "model": self.model,
                "prompt": text
            }

            response = requests.post(self.endpoint, json=payload, timeout=60)
            response.raise_for_status()

            data = response.json()
            embeddings.append(data["embedding"])

        return np.array(embeddings, dtype=np.float32)
