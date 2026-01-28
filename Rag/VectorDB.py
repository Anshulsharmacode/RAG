import os
from typing import List

import chromadb
import numpy as np


class VectorDB:
    def __init__(self, collection_name:str = "text", dir:str ="./Db"):
        self.collection_name = collection_name
        self.dir = dir
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.dir, exist_ok=True)
            self.client = chromadb.dir(path=self.dir)

            self.collection = self.client.get_or_create_collection(name=self.collection_name,
                                                                   metadata={"description": "Text embeddings"})
            print("ChromaDB initialized successfully.")

        except Exception as e:
            print(f"Error initializing store: {e}")

    def add_Document(self , document:List[any] , embeddings:np.ndarray):
        
        if len(document) != embeddings.shape[0]:
            raise ValueError("Length of document and embeddings must be the same")
        
        ids= []
        metadatas= []
        document_text = []
        embeddings_list = []

        for i,(doc , emb) in enumerate(zip(document, embeddings)):
            ids.append(str(i))
            metadatas.append({"source": "text"})
            document_text.append(doc)
            embeddings_list.append(emb.tolist())

        try:
            self.collection.add(
                documents=document_text,
                embeddings=embeddings_list,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"Error adding document: {e}")
