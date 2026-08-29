import os
import glob
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch


def build_vector_database(force_rebuild: bool = False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.abspath(os.path.join(script_dir, "..", "data"))
    vector_dir = os.path.join(data_folder, "chroma_db")

    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_folder}. Please run the faq_crawl.py first.")
        return

    if not force_rebuild and os.path.exists(vector_dir):
        newest_csv_mtime = max(os.path.getmtime(f) for f in csv_files)
        store_mtime = os.path.getmtime(vector_dir)
        if newest_csv_mtime <= store_mtime:
            print("Vector store already up to date with all CSV files. Skipping rebuild.")
            print("Pass force_rebuild=True to rebuild anyway.")
            return

    print(f"Found {len(csv_files)} CSV files. Loading data...")
    all_docs = []

    # Loop through every CSV and combine them (if multiple exist)
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        for i, row in df.iterrows():
            # Skip empty rows if any exist
            if pd.isna(row.get('Question')) or pd.isna(row.get('Answer')):
                continue

            doc = Document(
                page_content=f"Frage: {row['Question']}\nAntwort: {row['Answer']}",
                metadata={
                    "file_source": os.path.basename(csv_path),
                    "url_source": row.get('Source', 'Unknown'),
                    "row": int(i),
                }
            )
            all_docs.append(doc)

    print(f"Total Q&A pairs loaded (before dedup): {len(all_docs)}")

    # Dedup by full page content, in case of overlapping/stale CSV files
    seen = set()
    deduped_docs = []
    for doc in all_docs:
        key = doc.page_content
        if key not in seen:
            seen.add(key)
            deduped_docs.append(doc)

    if len(deduped_docs) < len(all_docs):
        print(f"Removed {len(all_docs) - len(deduped_docs)} duplicate entries.")
    all_docs = deduped_docs

    print(f"Total Q&A pairs after dedup: {len(all_docs)}")

    if not all_docs:
        print("No valid Q&A pairs found after filtering. Aborting.")
        return

    # Initialize Embeddings
    print("Initializing HuggingFace Embeddings (BAAI/bge-m3)...")
    cuda_isavailable = torch.cuda.is_available()
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",  # multilingual, handles German text
        model_kwargs={"device": "cuda" if cuda_isavailable else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Build and save the ChromaDB
    print(f"Building ChromaDB at: {vector_dir}...")
    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=vector_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("Database successfully created/updated! You can now run your chatbot.")


if __name__ == "__main__":
    build_vector_database()