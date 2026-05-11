from typing import List
import numpy as np
import requests
from tqdm.auto import tqdm
import time


class NomicEmbeddingModel:
    def __init__(self, host: str = "http://localhost:11434", model: str = "nomic-embed-text:latest"):
        self.host = host.rstrip("/")
        self.model = model
        self.endpoint = f"{self.host}/api/embeddings"

    def generate_embeddings(self, texts: List[str], show_progress: bool = False, timeout: int = 20, retries: int = 2, pause: float = 0.1) -> np.ndarray:
        """
        Generate embeddings for a list of texts with progress, retries and timeouts.

        Args:
            texts: List of text strings to embed
            show_progress: show tqdm progress bar
            timeout: per-request timeout in seconds
            retries: number of retries per text on failure
            pause: seconds to sleep between requests

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")

        embeddings = []
        iterator = tqdm(texts, desc="Embedding texts", disable=not show_progress, unit="txt")

        for text in iterator:
            last_err = None
            for attempt in range(retries + 1):
                try:
                    payload = {
                        "model": self.model,
                        "prompt": text
                    }
                    response = requests.post(self.endpoint, json=payload, timeout=timeout)
                    response.raise_for_status()
                    data = response.json()
                    # support either `embedding` or `data[0].embedding` style responses
                    emb = data.get("embedding") or (data.get("data") and data["data"][0].get("embedding"))
                    if emb is None:
                        raise RuntimeError("No embedding in response: %s" % data)
                    embeddings.append(emb)
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(0.5)  # short wait before retry
            else:
                # all retries failed
                raise RuntimeError(f"Embedding failed for a text after {retries} retries: {last_err}")

            time.sleep(pause)  # avoid hammering the server

        return np.array(embeddings, dtype=np.float32)
