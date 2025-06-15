from langchain_core.vectorstores import InMemoryVectorStore

class IntegratedVDB:
    def __init__(self, embedding, db_type: str = "in_memory"):
        if db_type not in ["in_memory", "astradb", "chroma", "faiss"]:
            raise ValueError("Unsupported vector store type. Currently only 'in_memory', 'astradb', 'chroma', and 'faiss' are supported.")
        if db_type == "in_memory":
            self.vector_store = InMemoryVectorStore(embedding)
        elif db_type == "astradb":
            # pip install -qU langchain-astradb
            from langchain_astradb import AstraDBVectorStore
            self.vector_store = AstraDBVectorStore(
                embedding=embeddings,
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                collection_name="astra_vector_langchain",
                token=ASTRA_DB_APPLICATION_TOKEN,
                namespace=ASTRA_DB_NAMESPACE,
            )
        elif db_type == "chroma":
            # pip install -qU langchain-chroma
            from langchain_chroma import Chroma

            self.vector_store = Chroma(
                collection_name="example_collection",
                embedding_function=embeddings,
                persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
            )
        elif db_type == "faiss":
            # pip install -qU langchain-community
            import faiss
            from langchain_community.docstore.in_memory import InMemoryDocstore
            from langchain_community.vectorstores import FAISS

            embedding_dim = len(embeddings.embed_query("hello world"))
            index = faiss.IndexFlatL2(embedding_dim)

            vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
        
  

    def add_texts(self, texts: list, metadatas: list = None):
