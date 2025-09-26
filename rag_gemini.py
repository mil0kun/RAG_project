#!/usr/bin/env python3
"""
Enhanced RAG with Gemini Integration
Combines ChromaDB similarity search with Google Gemini for intelligent answers.
"""

import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List, Dict, Any


class RAGWithGemini:
    """RAG system enhanced with Google Gemini for answer generation"""
    
    def __init__(self, chroma_store_dir: str = "./chroma_store", 
                 collection_name: str = "company_docs",
                 model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG system"""
        
        # Load environment variables
        load_dotenv()
        
        self.chroma_store_dir = chroma_store_dir
        self.collection_name = collection_name
        self.model_name = model_name
        self.top_k = 3
        
        # Initialize components
        self._setup_gemini()
        self._load_sentence_transformer()
        self._connect_chromadb()
    
    def _setup_gemini(self):
        """Setup Google Gemini API"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key or api_key == 'your_gemini_api_key_here':
            raise ValueError(
                "Please set your GEMINI_API_KEY in the .env file. "
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )
        
        genai.configure(api_key=api_key)
        
        # Configure generation settings
        self.gemini_model = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        self.generation_config = {
            'temperature': float(os.getenv('TEMPERATURE', '0.7')),
            'max_output_tokens': int(os.getenv('MAX_TOKENS', '1000')),
        }
        
        print(f"✓ Gemini configured with model: {self.gemini_model}")
    
    def _load_sentence_transformer(self):
        """Load the sentence transformer model"""
        print(f"Loading sentence transformer: {self.model_name}...")
        self.encoder = SentenceTransformer(self.model_name)
        print("✓ Sentence transformer loaded")
    
    def _connect_chromadb(self):
        """Connect to ChromaDB collection"""
        print("Connecting to ChromaDB...")
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_store_dir)
            self.collection = self.chroma_client.get_collection(self.collection_name)
            
            doc_count = self.collection.count()
            print(f"✓ Connected to '{self.collection_name}' with {doc_count} documents")
            
            if doc_count == 0:
                raise ValueError("Collection is empty. Run 'python rag_index.py' first.")
                
        except Exception as e:
            raise ValueError(f"Failed to connect to ChromaDB: {e}")
    
    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using similarity search"""
        
        # Generate query embedding
        query_embedding = self.encoder.encode([query], convert_to_tensor=False)
        
        # Search the collection
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=self.top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        # Format results
        context_docs = []
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for doc, metadata, distance in zip(documents, metadatas, distances):
            context_docs.append({
                'content': doc,
                'source': metadata.get('filename', 'Unknown'),
                'record_id': metadata.get('record_id', 'N/A'),
                'similarity_score': 1 - distance
            })
        
        return context_docs
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate answer using Gemini based on retrieved context"""
        
        if not context_docs:
            return "I couldn't find any relevant information to answer your question."
        
        # Prepare context for Gemini - more selective content to avoid token issues
        context_text = ""
        for i, doc in enumerate(context_docs, 1):
            # Get the most relevant parts of the document
            content = doc['content']
            
            # If content is very long, try to extract key sections
            if len(content) > 2000:
                # Split by separators and take key sections
                parts = content.split(' | ')
                key_parts = []
                for part in parts:
                    if any(keyword in part.lower() for keyword in ['vision', 'mission', 'services', 'advisory', 'investment', 'global', 'presence', 'founded', 'company']):
                        key_parts.append(part)
                    if len(' | '.join(key_parts)) > 1500:  # Limit total length
                        break
                content = ' | '.join(key_parts) if key_parts else content[:1500]
            
            context_text += f"Document {i} (Source: {doc['source']}):\n{content}\n\n"
        
        # Create prompt optimized for better Gemini responses
        prompt = f"""You are an AI assistant specializing in FPG Berhad, an investment holding company with M&A advisory services.

Based on the following documents, provide a comprehensive and well-structured answer to the user's question.

CONTEXT:
{context_text}

QUESTION: {query}

Please provide a detailed, professional response that:
1. Directly answers the question
2. Uses bullet points or structured format when appropriate
3. Cites information from the documents
4. Maintains a business-professional tone

ANSWER:"""

        try:
            # Use more conservative settings to avoid token issues
            model = genai.GenerativeModel(
                model_name=self.gemini_model,
                generation_config={
                    'temperature': 0.4,  # Balanced creativity
                    'max_output_tokens': 800,  # Reasonable length
                    'top_p': 0.8,
                    'top_k': 40
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )
            
            response = model.generate_content(prompt)
            
            # Better response handling
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                
                # Try to get the text first
                if hasattr(response, 'text') and response.text:
                    return response.text
                
                # Check finish reason if no text
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    
                    if finish_reason == 1:  # STOP - success
                        return response.text if response.text else "Response generated but empty."
                    elif finish_reason == 2:  # MAX_TOKENS
                        # Try with shorter context
                        return self._retry_with_shorter_context(query, context_docs)
                    elif finish_reason == 3:  # SAFETY
                        return self._format_structured_answer(context_docs, query)
                    else:
                        return self._format_structured_answer(context_docs, query)
                
                return self._format_structured_answer(context_docs, query)
            
            return self._format_structured_answer(context_docs, query)
            
        except Exception as e:
            print(f"⚠️  Gemini API error: {str(e)}")
            return self._format_structured_answer(context_docs, query)
    
    def _retry_with_shorter_context(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Retry with shorter context when hitting token limits"""
        
        try:
            # Use only the most relevant document with shorter content
            top_doc = context_docs[0]
            short_content = top_doc['content'][:800]  # Much shorter
            
            prompt = f"""Based on this FPG Berhad information, answer: {query}

Context: {short_content}

Provide a clear, professional answer:"""
            
            model = genai.GenerativeModel(
                model_name=self.gemini_model,
                generation_config={
                    'temperature': 0.3,
                    'max_output_tokens': 400,
                }
            )
            
            response = model.generate_content(prompt)
            
            if response.text:
                return response.text
            
        except:
            pass
        
        # Final fallback
        return self._format_structured_answer(context_docs, query)
    
    def _format_structured_answer(self, context_docs: List[Dict[str, Any]], query: str) -> str:
        """Format a structured answer when Gemini fails - but make it look professional"""
        
        top_doc = context_docs[0]
        content = top_doc['content']
        
        # Create a more professional structured response
        answer = f"Based on the information from FPG Berhad's {top_doc['source'].replace('.json', '').replace('_', ' ').title()}:\n\n"
        
        # Try to extract key information relevant to the query
        parts = content.split(' | ')
        
        # Filter parts based on query keywords
        query_keywords = query.lower().split()
        relevant_parts = []
        
        for part in parts[:8]:  # Limit to first 8 parts
            if any(keyword in part.lower() for keyword in query_keywords) or \
               any(keyword in part.lower() for keyword in ['vision', 'mission', 'services', 'advisory', 'founded', 'company']):
                relevant_parts.append(part.strip())
        
        # If no keyword matches, take the first few parts
        if not relevant_parts:
            relevant_parts = parts[:5]
        
        for part in relevant_parts:
            if part.strip() and ':' in part:
                answer += f"• {part.strip()}\n\n"
        
        return answer
    
    def query(self, question: str) -> Dict[str, Any]:
        """Main query method that combines retrieval and generation"""
        
        print(f"🔍 Searching for: '{question}'")
        
        # Step 1: Retrieve relevant documents
        context_docs = self.retrieve_context(question)
        
        if not context_docs:
            return {
                'question': question,
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'context_used': []
            }
        
        print(f"📄 Found {len(context_docs)} relevant documents")
        
        # Step 2: Generate answer with Gemini
        print("🤖 Generating answer with Gemini...")
        answer = self.generate_answer(question, context_docs)
        
        return {
            'question': question,
            'answer': answer,
            'sources': [doc['source'] for doc in context_docs],
            'context_used': context_docs
        }


def format_response(result: Dict[str, Any]) -> None:
    """Format and display the RAG response"""
    
    print("\n" + "="*70)
    print(f"QUESTION: {result['question']}")
    print("="*70)
    
    print(f"\n🤖 ANSWER:")
    print("-" * 50)
    print(result['answer'])
    
    print(f"\n📚 SOURCES:")
    print("-" * 50)
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. {source}")
    
    if result['context_used']:
        print(f"\n📄 CONTEXT SIMILARITY SCORES:")
        print("-" * 50)
        for i, doc in enumerate(result['context_used'], 1):
            print(f"{i}. {doc['source']}: {doc['similarity_score']:.3f}")


def main():
    """Main interactive loop"""
    
    print("="*70)
    print("🚀 RAG + Gemini Query System - FPG Berhad")
    print("="*70)
    print("This system combines document similarity search with Gemini AI")
    print("to provide intelligent answers about FPG Berhad's business.")
    print()
    
    try:
        # Initialize RAG system
        rag = RAGWithGemini()
        print("\n✅ System initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("\n💡 Make sure you have:")
        print("1. Set GEMINI_API_KEY in your .env file")
        print("2. Run 'python rag_index.py' to create the document index")
        return
    
    print("\n" + "-"*70)
    print("💡 Sample questions you can ask:")
    print("• What is FPG Berhad's vision and mission?")
    print("• Tell me about FPG's M&A advisory services")
    print("• What are FPG's investment sectors?")
    print("• Where does FPG operate globally?")
    print("• What successful M&A cases has FPG completed?")
    print("-"*70)
    print("\nType 'quit' or 'exit' to stop")
    print("="*70)
    
    while True:
        try:
            # Get user input
            question = input("\n💭 Ask a question: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using the RAG + Gemini system!")
                break
            
            # Process query
            result = rag.query(question)
            
            # Display result
            format_response(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            continue


if __name__ == "__main__":
    main()