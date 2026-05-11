import faiss
import pickle
import numpy as np
from pathlib import Path
from RAG_model.embbedings import get_embeddings

class VectorStore:
    def __init__(self, index_path: str = "index"):
        self.index_path = Path(index_path)
        self.index = None
        self.chunks = None

    def create(self, chunks: list):
        """Создаёт новую векторную базу"""
        embeddings, _ = get_embeddings(chunks)
        
        print("🗄️ Создаём FAISS индекс...")
        dimension = embeddings.shape[1]
        
        # IndexFlatIP + нормализованные вектора = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks
        
        self.save()
        print(f"✅ Векторная база создана ({len(chunks)} чанков)")
        return self

    def save(self):
        """Сохраняет индекс и чанки на диск"""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(self.index_path / "faiss.index"))
        
        with open(self.index_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        print(f"💾 Индекс сохранён в папку: {self.index_path}")

    def load(self):
        """Загружает существующую векторную базу"""
        try:
            self.index = faiss.read_index(str(self.index_path / "faiss.index"))
            with open(self.index_path / "chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
            print(f"✅ Векторная база загружена ({len(self.chunks)} чанков)")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"Ошибка загрузки индекса: {e}")
            return False