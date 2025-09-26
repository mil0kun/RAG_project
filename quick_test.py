#!/usr/bin/env python3
"""
Quick test of the improved RAG system
"""

from rag_gemini import RAGWithGemini

def quick_test():
    try:
        print("🚀 Testing improved RAG system...")
        rag = RAGWithGemini()
        
        # Test with a single question
        question = "What is FPG Berhad and what do they do?"
        result = rag.query(question)
        
        print(f"\n🤖 ANSWER:")
        print("-" * 50)
        print(result['answer'])
        
        print(f"\n📚 SOURCES:")
        print("-" * 50)
        for i, source in enumerate(result['sources'], 1):
            similarity = result['context_used'][i-1]['similarity_score']
            print(f"{i}. {source} (Similarity: {similarity:.3f})")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    quick_test()