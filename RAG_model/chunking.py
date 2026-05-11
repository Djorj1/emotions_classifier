from tqdm import tqdm


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size


        if end < text_len:
            for sep in ['.\n', '.\n\n', '\n\n', '. ', '！', '？', '。']:
                pos = text.rfind(sep, start, end)
                if pos > start + chunk_size // 2:
                    end = pos + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap
    return chunks


def create_chuncks(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200):
    all_chunks = []


    print("Разбиваем документы на чанки")

    for doc in tqdm(documents, desc = "Chunking"):
        chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)

        for i, chunks_text in enumerate(chunks):
            chunk = {
                "text": chunks_text,
                "metadata": {
                    **doc["metadata"],           
                    "chunk_id": i,
                    "chunk_size": len(chunks_text)
                }
            }
            all_chunks.append(chunk)

    print(f" Создано чанков: {len(all_chunks)}")
    return all_chunks