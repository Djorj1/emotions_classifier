# generator.py
import ollama

def generate_answer(query: str, retrieved_chunks: list, model: str = "llama3.2"):
    """Генерация ответа с помощью LLM"""
    
    context = "\n\n".join([
        f"Источник: {r['metadata'].get('file_name', 'unknown')}\n{r['text']}" 
        for r in retrieved_chunks
    ])

    prompt = f"""Ты — точный и полезный ассистент.
Используй только предоставленный контекст для ответа.
Если информации недостаточно — честно скажи об этом.

Контекст:
{context}

Вопрос: {query}

Ответь на русском языке, подробно и понятно:"""

    print("Генерация ответа...")
    
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    answer = response['message']['content']
    
    print("\n" + "="*60)
    print(answer)
    print("="*60)
    
    return answer