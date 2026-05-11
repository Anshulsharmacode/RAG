from typing import Any, Dict, List



class RAG:
    def __init__(self, db, emb):
        self.db = db
        self.emb = emb

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:

        print(f"Retrieving top {top_k} documents for query: {query}")

        query_emb = self.emb.generate_embeddings([query])

        try:
            results = self.db.collection.query(
                query_embeddings=query_emb,
                n_results=top_k,
                include=["documents", "show metadatas", "distances"]
            )

            retrieved_docs = []

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    # Chroma uses cosine distance → similarity
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })

            return retrieved_docs

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
