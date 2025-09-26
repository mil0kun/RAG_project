from flask import Flask, render_template, request, jsonify, stream_template
import json
import os
from datetime import datetime
from rag_gemini import RAGWithGemini

app = Flask(__name__)

# Initialize RAG system
rag_system = None

def initialize_rag():
    """Initialize the RAG system"""
    global rag_system
    try:
        rag_system = RAGWithGemini()
        return True, "RAG system initialized successfully"
    except Exception as e:
        return False, f"Failed to initialize RAG system: {str(e)}"

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    global rag_system
    
    try:
        # Get message from request
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'error': True,
                'message': 'Please enter a message'
            })
        
        # Initialize RAG system if not already done
        if rag_system is None:
            success, msg = initialize_rag()
            if not success:
                return jsonify({
                    'error': True,
                    'message': f'System not ready: {msg}'
                })
        
        # Get response from RAG system
        result = rag_system.query(user_message)
        
        # Format response
        response_data = {
            'error': False,
            'message': result['answer'],
            'sources': result['sources'],
            'similarity_scores': [
                {
                    'source': doc['source'],
                    'score': round(doc['similarity_score'], 3)
                }
                for doc in result['context_used']
            ],
            'timestamp': datetime.now().strftime('%H:%M')
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Error processing your message: {str(e)}'
        })

@app.route('/api/status')
def status():
    """Check system status"""
    global rag_system
    
    if rag_system is None:
        success, msg = initialize_rag()
        return jsonify({
            'ready': success,
            'message': msg
        })
    
    return jsonify({
        'ready': True,
        'message': 'System ready'
    })

@app.route('/api/sample-questions')
def sample_questions():
    """Get sample questions for the UI"""
    questions = [
        "What is FPG Berhad and what do they do?",
        "Tell me about FPG's M&A advisory services",
        "What is FPG's global presence?",
        "What are FPG's investment sectors?",
        "Tell me about FPG's vision and mission",
        "What successful M&A cases has FPG completed?"
    ]
    
    return jsonify({
        'questions': questions
    })

if __name__ == '__main__':
    print("🚀 Starting FPG Berhad RAG Chatbot...")
    print("🔧 Initializing system...")
    
    # Pre-initialize RAG system
    success, msg = initialize_rag()
    if success:
        print(f"✅ {msg}")
        print("🌐 Starting web server...")
        print("📱 Open http://localhost:5000 in your browser")
        print("🛑 Press Ctrl+C to stop")
    else:
        print(f"❌ {msg}")
        print("⚠️  Starting server anyway - system will initialize on first request")
    
    app.run(debug=True, host='0.0.0.0', port=5000)