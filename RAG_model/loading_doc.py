import os
import pickle 
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

def load_document(directory: str = 'data'):
    documents = []
    data_path = Path(directory)


    if not data_path.exists():
        print(f"❌ Папка '{directory}' не найдена!")
        return documents
     
    html_files = list(data_path.rglob("*.html")) + list(data_path.rglob("*.htm"))


    if not html_files:
        print("File not find!!!!!")
        return documents
    

    print(f"Find {len(html_files)}")


    for file_path in tqdm(html_files, desc="Загрузка HTML"):
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(content,"html.parser")

            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
                       
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)

            if len(clean_text) < 50:
                print(f"⚠️ Файл слишком маленький: {file_path.name}")
                continue

            metadata = {
                "file_name": file_path.name,
                "source": str(file_path),
                "file_path": str(file_path),
                "title": soup.title.string.strip() if soup.title else file_path.name,
                "text_length": len(clean_text)
            }

            documents.append({
                "text": clean_text,
                "metadata": metadata
            })

        except Exception as e:
            print(f"Ошибка при обработке {file_path.name}: {e}")
    
    
    with open("cache/documents.pkl", "wb") as f:
        pickle.dump(documents, f)
    print(f"\n Успешно загружено документов: {len(documents)}")
    return documents


