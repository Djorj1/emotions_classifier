from RAG_model.embbedings import get_embeddings
from RAG_model.chunking import create_chuncks
from RAG_model.loading_doc import load_document
# from RAG_model.promt import
# from RAG_model.retrieval import
# from RAG_model.vector_store import

if __name__ == "__main__":

    folder_path = "data"                  
    documents = load_document(folder_path)
    print(f"\nВсего загружено документов: {len(documents)}")


    chunks = create_chuncks(documents = documents)


    embeddings, model_embeddings = get_embeddings(chunks = chunks)
    print(f"\nЭмбэдинги сохранены")