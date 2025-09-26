#!/usr/bin/env python3
"""
RAG Querying Script
Loads the ChromaDB collection and performs similarity search based on user queries.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


def format_results(results: Dict[str, Any], query: str) -> None:
    """
    Format and display search results in a readable way.
    
    Args:
        results: Results from ChromaDB query
        query: Original user query
    """
    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)
    
    if not results['documents'] or not results['documents'][0]:
        print("No results found.")
        return
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0] if 'distances' in results else [None] * len(documents)
    
    for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
        print(f"\n--- RESULT {i+1} ---")
        print(f"Source File: {metadata.get('filename', 'Unknown')}")
        print(f"Record ID: {metadata.get('record_id', 'N/A')}")
        if distance is not None:
            print(f"Similarity Score: {1 - distance:.4f}")
        print(f"Content:")
        
        # Truncate very long documents for display
        if len(doc) > 500:
            print(f"{doc[:500]}...")
            print(f"[Document truncated - showing first 500 chars of {len(doc)} total]")
        else:
            print(doc)
        print("-" * 40)


def main():
    """Main querying function"""
    
    # Configuration
    CHROMA_STORE_DIR = "./chroma_store"
    COLLECTION_NAME = "company_docs"
    MODEL_NAME = "all-mpnet-base-v2"  # Upgraded from all-MiniLM-L6-v2 for better quality
    TOP_K_RESULTS = 3
    
    print("=== RAG Query Interface ===")
    print(f"Chroma store: {CHROMA_STORE_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {MODEL_NAME}")
    print(f"Returning top {TOP_K_RESULTS} results per query")
    print()
    
    # Load sentence transformer model
    print("Loading sentence transformer model...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    print()
    
    # Initialize ChromaDB
    print("Connecting to ChromaDB...")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_STORE_DIR)
        collection = chroma_client.get_collection(COLLECTION_NAME)
        
        # Get collection info
        doc_count = collection.count()
        print(f"Connected to collection '{COLLECTION_NAME}' with {doc_count} documents")
        
        if doc_count == 0:
            print("The collection is empty. Please run 'rag_index.py' first to index your documents.")
            return
            
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        print("Make sure you have run 'rag_index.py' first to create the collection.")
        return
    
    print()
    print("Ready for queries! (Type 'quit' or 'exit' to stop)")
    print("-" * 50)
    
    # Query loop
    while True:
        try:
            # Get user input
            query = input("\nEnter your query: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            print(f"Searching for: '{query}'...")
            
            # Generate query embedding
            query_embedding = model.encode([query], convert_to_tensor=False)
            
            # Search the collection
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=TOP_K_RESULTS,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Display results
            format_results(results, query)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"Error processing query: {e}")
            continue


if __name__ == "__main__":
    main()