import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer


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


def get_embeddings(chunks: list, batch_size: int = 64):
    """Превращает чанки в вектора"""
    embedder = get_embedder()
    texts = [chunk["text"] for chunk in chunks]
    

    if torch.cuda.is_available():  # ✅ НОВАЯ СТРОКА - проверка GPU
        batch_size = 128


    print("Вычисляем эмбеддинги...")
    embeddings = embedder.encode(
        texts, 
        show_progress_bar=True,
        normalize_embeddings=True,   
        batch_size=batch_size
    )
    
    return np.array(embeddings), embedder