import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


_embedder_model = None

def get_embedder():
    """Возвращает модель эмбеддингов (singleton)"""
    global _embedder_model
    if _embedder_model is None:
        print("Загрузка модели эмбеддингов (intfloat/multilingual-e5-small)...")
        _embedder_model = SentenceTransformer('intfloat/multilingual-e5-small')
        print("Модель эмбеддингов загружена")
    return _embedder_model


def get_embeddings(chunks: list):
    """Превращает чанки в вектора"""
    embedder = get_embedder()
    texts = [chunk["text"] for chunk in chunks]
    
    print("Вычисляем эмбеддинги...")
    embeddings = embedder.encode(
        texts, 
        show_progress_bar=True,
        normalize_embeddings=True,   
        batch_size=32
    )
    
    return np.array(embeddings), embedder