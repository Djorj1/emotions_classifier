import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from RAG_model.save_embedings import load_embeddings, save_embeddings


_embedder_model = None



def get_embedder():
    """Возвращает модель эмбеддингов (singleton)"""
    global _embedder_model
    if _embedder_model is None:
        print("Загрузка модели эмбеддингов (intfloat/multilingual-e5-small)...")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'  


        _embedder_model = SentenceTransformer('intfloat/multilingual-e5-small', device=device)
        print("Модель эмбеддингов загружена")
    return _embedder_model


def get_embeddings(chunks: list, batch_size: int = 64, use_cache: bool = True, cache_dir: str = "embeddings_cache"):
    """Превращает чанки в вектора"""
    if use_cache:
        try:  
            embeddings, cached_chunks = load_embeddings(cache_dir)
            if embeddings is not None and len(cached_chunks) == len(chunks):
                return embeddings, get_embedder()
        except Exception:  
                pass 
        

    embedder = get_embedder()
    texts = [chunk["text"] for chunk in chunks]
    

    if torch.cuda.is_available():  
        batch_size = 128


    print("Вычисляем эмбеддинги...")
    embeddings = embedder.encode(
        texts, 
        show_progress_bar=True,
        normalize_embeddings=True,   
        batch_size=batch_size
    )
    
    if use_cache: 
        try:
            save_embeddings(embeddings, chunks, cache_dir)
        except Exception:
            pass
    return np.array(embeddings), embedder