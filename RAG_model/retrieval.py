import numpy as np
from RAG_model.embbedings import get_embedder

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.embedder = get_embedder()

    def retrieve(self, query: str, top_k: int = 5):
        """Поиск релевантных чанков"""
    
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        

        distances, indices = self.vector_store.index.search(
            query_embedding.astype(np.float32), top_k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.vector_store.chunks[idx]["text"],
                "score": float(distances[0][i]),
                "metadata": self.vector_store.chunks[idx]["metadata"]
            })
        
        return results