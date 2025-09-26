#!/usr/bin/env python3
"""
Demo script for RAG + Gemini integration
Tests the enhanced system with sample queries
"""

import os
from dotenv import load_dotenv
from rag_gemini import RAGWithGemini


def demo_gemini_rag():
    """Demonstrate RAG + Gemini with business queries"""
    
    print("="*70)
    print("🚀 RAG + Gemini Demo - FPG Berhad")
    print("="*70)
    
    # Load environment
    load_dotenv()
    
    # Check API key
    if not os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY') == 'your_gemini_api_key_here':
        print("❌ Please set your GEMINI_API_KEY in the .env file")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    try:
        # Initialize RAG system
        print("Initializing RAG + Gemini system...")
        rag = RAGWithGemini()
        print("✅ System ready!")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Demo queries
    demo_questions = [
        "What is FPG Berhad and what do they do?",
        "Tell me about FPG's M&A advisory services",
        "What is FPG's global presence?"
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{'='*70}")
        print(f"DEMO QUERY {i}/3: {question}")
        print(f"{'='*70}")
        
        # Process query
        result = rag.query(question)
        
        # Display formatted result
        print(f"\n🤖 GEMINI ANSWER:")
        print("-" * 50)
        print(result['answer'])
        
        print(f"\n📚 SOURCES:")
        for j, source in enumerate(result['sources'], 1):
            similarity = result['context_used'][j-1]['similarity_score']
            print(f"{j}. {source} (Similarity: {similarity:.3f})")
        
        # Wait for user to continue (except last question)
        if i < len(demo_questions):
            input(f"\n⏭️  Press Enter to continue to demo query {i+1}...")
    
    print(f"\n{'='*70}")
    print("✅ Demo completed!")
    print("Use 'python rag_gemini.py' for interactive queries")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo_gemini_rag()