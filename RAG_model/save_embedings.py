import pickle
import numpy as np 
from pathlib import Path

def save_embeddings(embeddings: np.ndarray, chunks: list, save_dir: str = "embeddings_cache"):
    """Сохраняет эмбеддинги и чанки на диск"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    np.save(save_path / "embeddings.npy", embeddings)
    
    with open(save_path / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_embeddings(load_dir: str = "embeddings_cache"): 
    """Загружает эмбеддинги и чанки с диска"""
    load_path = Path(load_dir)
    
    embeddings = np.load(load_path / "embeddings.npy")
    
    with open(load_path / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    
    return embeddings, chunks