#!/usr/bin/env python3
"""
store_and_print_chroma.py
Reads a .txt file, stores its content in ChromaDB persistent DB, and prints stored data.
"""

import os
import sys

# Disable Chroma telemetry to avoid warnings
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

import chromadb

def main():
    # --- Input Validation ---
    if len(sys.argv) != 3:
        print("Usage: python store_and_print_chroma.py <txt_file_path> <persist_dir>")
        sys.exit(1)

    txt_file_path = sys.argv[1]
    persist_dir   = sys.argv[2]

    # Ensure file exists
    if not os.path.isfile(txt_file_path):
        print(f"Error: File '{txt_file_path}' not found.")
        sys.exit(1)

    # Ensure persist_dir is absolute
    persist_dir = os.path.abspath(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    # --- Read file content ---
    with open(txt_file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        print("Error: The .txt file is empty.")
        sys.exit(1)

    # --- Connect to ChromaDB PersistentClient ---
    client = chromadb.PersistentClient(path=persist_dir)
    collection_name = "podcast_rag"
    collection = client.get_or_create_collection(name=collection_name)

    # --- Add document ---
    doc_id = os.path.basename(txt_file_path)  # Use filename as ID
    collection.add(documents=[content], ids=[doc_id])

    # Persist changes
    # client.persist()

    print(f"✅ Data stored in ChromaDB persist dir: {persist_dir}")

    # --- Retrieve and print all documents ---
    all_docs = collection.get()
    print("\n📂 Stored Documents:")
    for idx, doc in enumerate(all_docs["documents"], start=1):
        print(f"{idx}. ID: {all_docs['ids'][idx-1]}")
        print(f"   Content: {doc}\n")

if __name__ == "__main__":
    main()


# pip install chromadb
# python store_and_print_chroma.py sample.txt ./chroma_db
