#!/usr/bin/env python3
"""
Business-Specific RAG Test Script
Demonstrates the RAG pipeline with business-relevant queries for FPG Berhad content.
"""

import chromadb
from sentence_transformers import SentenceTransformer


def test_business_queries():
    """Run business-specific test queries"""
    
    # Configuration
    CHROMA_STORE_DIR = "./chroma_store"
    COLLECTION_NAME = "company_docs"
    MODEL_NAME = "all-MiniLM-L6-v2"
    
    print("=== FPG Berhad RAG Business Demo ===")
    print("Loading model and connecting to database...")
    
    # Load model and connect to ChromaDB
    model = SentenceTransformer(MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_STORE_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    
    print(f"Connected to collection with {collection.count()} documents")
    
    # Business-specific queries for FPG Berhad content
    business_queries = [
        "What is FPG Berhad's vision and mission?",
        "Tell me about FPG's M&A advisory services",
        "What are FPG's investment portfolio sectors?",
        "Where does FPG have global presence?",
        "What are FPG's successful M&A transactions?",
        "Tell me about FPG's team and leadership",
        "What are FPG's fees for M&A services?",
        "What is FPG's privacy policy?",
        "Tell me about FPG's career opportunities"
    ]
    
    for i, query in enumerate(business_queries, 1):
        print(f"\n{'='*70}")
        print(f"BUSINESS QUERY {i}: {query}")
        print('='*70)
        
        # Generate query embedding
        query_embedding = model.encode([query], convert_to_tensor=False)
        
        # Search the collection
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=2,  # Show top 2 results for each query
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['documents'] or not results['documents'][0]:
            print("No results found.")
            continue
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for j, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            print(f"\n--- RESULT {j+1} ---")
            print(f"Source: {metadata.get('filename', 'Unknown').replace('.json', '').replace('_', ' ').title()}")
            print(f"Similarity: {1 - distance:.4f}")
            print("Content:")
            
            # Show first 400 characters with better formatting
            content = doc.replace(" | ", "\n• ").replace(": ", ": ")
            if len(content) > 400:
                print(f"{content[:400]}...")
            else:
                print(content)
            print("-" * 50)
    
    print(f"\n{'='*70}")
    print("Business demo completed!")
    print("Your RAG pipeline is ready for interactive use with rag_query.py")
    print('='*70)


if __name__ == "__main__":
    test_business_queries()