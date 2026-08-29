import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch

class KauflandRAG:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vector_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "chroma_db"))
        
        if not os.path.exists(vector_dir):
            raise FileNotFoundError(f"ChromaDB not found at: {vector_dir}. Please run utility/ingest_data.py first.")

        # 1. Initialize Free Local Embeddings (Must match the ingest script exactly)
        print("[RAG Engine] Initializing HuggingFace Embeddings...")
        cuda_isavailable = torch.cuda.is_available()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cuda" if cuda_isavailable else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # 2. Load the existing Vector Database (Read-Only)
        print("[RAG Engine] Connecting to existing ChromaDB...")
        self.vector_store = Chroma(
            persist_directory=vector_dir, 
            embedding_function=self.embeddings, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        print("[RAG Engine] Ready!")

    def retrieve(self, query: str, k: int = 4, score_threshold: float = 0.5) -> str:
        """Searches the vector database for the top 'k' most relevant FAQs."""
        results_with_scores = self.vector_store.similarity_search_with_relevance_scores(query, k=k)

        filtered_results = [
            (doc, score) for doc, score in results_with_scores
            if score >= score_threshold
        ]

        if not filtered_results:
            print(f"[RAG] No results above relevance threshold ({score_threshold}) for query: '{query}'")
            return ""

        formatted_context = ""
        for doc, score in filtered_results:
            formatted_context += f"{doc.page_content}\n---\n"

        return formatted_context.strip()
    
    def get_all_documents(self) -> tuple[list[str], list[dict]]:
        """
        Returns all document texts + metadatas directly from the Chroma collection.
        Useful for building a BM25 index and spell-correction vocabulary.
        """
        # We use .get() to pull everything directly from the database instead of a CSV
        all_data = self.vector_store.get(include=['documents', 'metadatas'])
        return all_data['documents'], all_data['metadatas']

    def retrieve_scored(self, query: str, k: int = 5, score_threshold: float = 0.0) -> list[dict]:
        """Like retrieve(), but returns individual scored docs for RRF fusion."""
        results_with_scores = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        return [
            {"content": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in results_with_scores
            if score >= score_threshold
        ]

# --- Test Block ---
if __name__ == "__main__":
    rag = KauflandRAG()

    print("--- Relevant query ---")
    results = rag.vector_store.similarity_search_with_relevance_scores("Wie funktioniert Kaufland Pay?", k=4)
    for doc, score in results:
        print(f"{score:.3f} | {doc.page_content[:60]}...")

    print("\n--- Irrelevant query (should score much lower) ---")
    results = rag.vector_store.similarity_search_with_relevance_scores("Wie ist das Wetter in Berlin?", k=4)
    for doc, score in results:
        print(f"{score:.3f} | {doc.page_content[:60]}...")