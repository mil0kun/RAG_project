#!/usr/bin/env python3
"""
RAG Indexing Script
Loads JSON files from ./data/, processes them, and stores in ChromaDB with embeddings.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer


def load_json_files(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load all JSON files from the data directory.
    Each record in each file becomes a separate document.
    
    Returns:
        List of dictionaries containing document data and metadata
    """
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Data directory '{data_dir}' does not exist!")
        return documents
    
    json_files = list(data_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{data_dir}'")
        return documents
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for json_file in json_files:
        try:
            print(f"Processing: {json_file.name}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both single objects and arrays
            if isinstance(data, list):
                for i, record in enumerate(data):
                    documents.append({
                        'content': record,
                        'filename': json_file.name,
                        'record_id': i
                    })
            else:
                # Single object
                documents.append({
                    'content': data,
                    'filename': json_file.name,
                    'record_id': 0
                })
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    return documents


def dict_to_text(data: Dict[str, Any]) -> str:
    """
    Convert a dictionary to a flat text string for embedding.
    
    Args:
        data: Dictionary to convert
        
    Returns:
        Flattened text representation
    """
    text_parts = []
    
    def flatten_dict(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}.{key}" if prefix else key
                flatten_dict(value, new_key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
                flatten_dict(item, new_key)
        else:
            # Convert value to string
            text_parts.append(f"{prefix}: {str(obj)}")
    
    flatten_dict(data)
    return " | ".join(text_parts)


def main():
    """Main indexing function"""
    
    # Configuration
    DATA_DIR = "./content"
    CHROMA_STORE_DIR = "./chroma_store"
    COLLECTION_NAME = "company_docs"
    MODEL_NAME = "all-mpnet-base-v2"  # Upgraded from all-MiniLM-L6-v2 for better quality
    
    print("=== RAG Indexing Pipeline ===")
    print(f"Data directory: {DATA_DIR}")
    print(f"Chroma store: {CHROMA_STORE_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {MODEL_NAME}")
    print()
    
    # Load sentence transformer model
    print("Loading sentence transformer model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully!")
    print()
    
    # Load JSON documents
    print("Loading JSON documents...")
    documents = load_json_files(DATA_DIR)
    
    if not documents:
        print("No documents found to index. Exiting.")
        return
    
    print(f"Loaded {len(documents)} documents total.")
    print()
    
    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_STORE_DIR)
    
    # Get or create collection
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        print(f"Found existing collection '{COLLECTION_NAME}'. Deleting to rebuild...")
        chroma_client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Company documents with embeddings"}
    )
    print(f"Created collection '{COLLECTION_NAME}'")
    print()
    
    # Process documents in batches
    print("Processing documents and generating embeddings...")
    batch_size = 50
    total_indexed = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        # Convert documents to text
        texts = []
        metadatas = []
        ids = []
        
        for j, doc in enumerate(batch):
            doc_id = f"doc_{total_indexed + j}"
            text = dict_to_text(doc['content'])
            
            texts.append(text)
            metadatas.append({
                'filename': doc['filename'],
                'record_id': doc['record_id']
            })
            ids.append(doc_id)
        
        # Generate embeddings
        embeddings = model.encode(texts, convert_to_tensor=False)
        embeddings_list = embeddings.tolist()
        
        # Add to collection
        collection.add(
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        total_indexed += len(batch)
        print(f"Indexed {total_indexed}/{len(documents)} documents...")
    
    print()
    print("=== Indexing Complete ===")
    print(f"Successfully indexed {total_indexed} documents")
    print(f"ChromaDB collection '{COLLECTION_NAME}' saved to '{CHROMA_STORE_DIR}'")
    
    # Verify the collection
    collection_info = collection.count()
    print(f"Collection contains {collection_info} documents")
    print()
    print("You can now run 'rag_query.py' to search the indexed documents!")


if __name__ == "__main__":
    main()