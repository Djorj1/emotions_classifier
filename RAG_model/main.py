from RAG_model.embbedings import get_embeddings
from RAG_model.chunking import create_chuncks
from RAG_model.loading_doc import load_document
from RAG_model.promt import generate_answer
from RAG_model.retrieval import Retriever
from RAG_model.vector_store import VectorStore


def main():
    folder_path = "data"                  
    documents = load_document(folder_path)
    print(f"\nВсего загружено документов: {len(documents)}")


    chunks = create_chuncks(documents = documents)


        # embeddings, model_embeddings = get_embeddings(chunks = chunks)
        # print(f"\nЭмбэдинги сохранены")


    vector_store = VectorStore(index_path="my_vector_index")
    vector_store.create(chunks)
    print("\n Векторная база успешно создана и сохранена!")

    retriever = Retriever(vector_store)


    print("\nСистема готова. Введите 'выход' для завершения\n")
    
    while True:
        query = input("Ваш вопрос: ").strip()
        
        if query.lower() in ['выход', 'quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        relevant_chunks = retriever.retrieve(query=query, top_k=5)
        
        if not relevant_chunks:
            print("Ничего не найдено\n")
            continue
        
        answer = generate_answer(
            query=query,
            retrieved_chunks=relevant_chunks,
            model="llama3.2"
        )

        
if __name__ == "__main__":
    main()