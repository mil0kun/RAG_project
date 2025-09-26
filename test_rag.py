#!/usr/bin/env python3
"""
RAG Test Script
Demonstrates the RAG pipeline with sample queries.
"""

import chromadb
from sentence_transformers import SentenceTransformer


def test_query(collection, model, query_text):
    """Test a single query and display results"""
    print(f"\n{'='*60}")
    print(f"QUERY: {query_text}")
    print('='*60)
    
    # Generate query embedding
    query_embedding = model.encode([query_text], convert_to_tensor=False)
    
    # Search the collection
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3,
        include=['documents', 'metadatas', 'distances']
    )
    
    if not results['documents'] or not results['documents'][0]:
        print("No results found.")
        return
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
        print(f"\n--- RESULT {i+1} ---")
        print(f"Source File: {metadata.get('filename', 'Unknown')}")
        print(f"Record ID: {metadata.get('record_id', 'N/A')}")
        print(f"Similarity Score: {1 - distance:.4f}")
        print(f"Content Preview:")
        
        # Show first 300 characters
        if len(doc) > 300:
            print(f"{doc[:300]}...")
        else:
            print(doc)
        print("-" * 40)


def main():
    """Run test queries"""
    
    # Configuration
    CHROMA_STORE_DIR = "./chroma_store"
    COLLECTION_NAME = "company_docs"
    MODEL_NAME = "all-MiniLM-L6-v2"
    
    print("=== RAG Pipeline Test ===")
    print(f"Loading model and connecting to database...")
    
    # Load model and connect to ChromaDB
    model = SentenceTransformer(MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_STORE_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    
    print(f"Connected to collection with {collection.count()} documents")
    
    # Test queries based on what might be in your content
    test_queries = [
        "investment portfolio and services",
        "mergers and acquisitions advisory",
        "who we are and our team",
        "privacy policy and terms",
        "successful cases and portfolio"
    ]
    
    for query in test_queries:
        test_query(collection, model, query)
    
    print(f"\n{'='*60}")
    print("Test completed! You can now use rag_query.py for interactive queries.")
    print('='*60)


if __name__ == "__main__":
    main()