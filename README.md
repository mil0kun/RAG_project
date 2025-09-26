# Local RAG Pipeline with ChromaDB

A complete Retrieval-Augmented Generation (RAG) pipeline using ChromaDB and sentence-transformers for local document search and retrieval.

## Project Structure

```
my_rag_project/
├── content/                 # JSON files containing your documents
│   ├── about_us_our_team.json
│   ├── investment_portfolio.json
│   ├── ma_advisory_complete.json
│   └── ... (16 JSON files total)
├── chroma_store/           # Persistent ChromaDB storage (created automatically)
├── rag_index.py           # Indexing script - processes JSON files and creates embeddings
├── rag_query.py           # Query script - search indexed documents
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install chromadb sentence-transformers
```

## Usage

### Step 1: Index Your Documents

First, run the indexing script to process all JSON files in the `./content/` directory:

```bash
python rag_index.py
```

This script will:
- Load all JSON files from `./content/`
- Convert each record into text format
- Generate embeddings using `all-MiniLM-L6-v2`
- Store everything in ChromaDB collection `company_docs`
- Persist the database in `./chroma_store/`

### Step 2: Query Your Documents

Run the query script to search your indexed documents:

```bash
python rag_query.py
```

Example queries you can try with your FPG Berhad data:
- "What is FPG Berhad's vision and mission?"
- "Tell me about FPG's M&A advisory services"
- "What are FPG's investment portfolio sectors?"
- "Where does FPG have global presence?"
- "What are FPG's successful M&A transactions?"
- "Tell me about FPG's team and leadership"

Type 'quit' or 'exit' to stop the query interface.

## Features

### Indexing Script (`rag_index.py`)
- **Automatic JSON Processing**: Loads all `.json` files from `./content/`
- **Flexible Data Structure**: Handles both single objects and arrays in JSON files
- **Text Conversion**: Converts nested JSON structures to flat text for embedding
- **Metadata Preservation**: Stores filename and record ID for each document
- **Batch Processing**: Processes documents in batches for efficiency
- **Persistent Storage**: ChromaDB data survives script restarts

### Query Script (`rag_query.py`)
- **Interactive Interface**: Command-line interface for real-time querying
- **Similarity Search**: Returns top 3 most relevant documents
- **Rich Results**: Shows document content, source file, and similarity scores
- **Error Handling**: Graceful handling of connection and query errors
- **Content Truncation**: Long documents are truncated for readability

## Customization

### Change Embedding Model
Modify the `MODEL_NAME` variable in both scripts to use a different sentence-transformer model:
```python
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, good quality
# MODEL_NAME = "all-mpnet-base-v2"  # Higher quality, slower
```

### Adjust Number of Results
Change `TOP_K_RESULTS` in `rag_query.py`:
```python
TOP_K_RESULTS = 5  # Return top 5 results instead of 3
```

### Modify Collection Name
Change `COLLECTION_NAME` in both scripts:
```python
COLLECTION_NAME = "my_documents"
```

## Adding Your Own Data

1. Place your JSON files in the `./content/` directory
2. Each file should contain either:
   - A single JSON object: `{"key": "value", ...}`
   - An array of JSON objects: `[{"key": "value"}, {"key": "value"}, ...]`
3. Run `python rag_index.py` to re-index with your new data

## Technical Details

- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional vectors)
- **Database**: ChromaDB with persistent storage
- **Similarity Metric**: Cosine similarity (default ChromaDB behavior)
- **Text Processing**: Nested JSON flattened to "key: value" format

## Troubleshooting

### "Collection not found" error
Run `python rag_index.py` first to create the collection.

### "No JSON files found"
Ensure your JSON files are in the `./content/` directory and have `.json` extension.

### Import errors
Install the required packages:
```bash
pip install chromadb sentence-transformers
```

### Performance issues
- Use smaller batch sizes in indexing for limited memory
- Try a smaller embedding model for faster processing
- Consider limiting the number of documents for testing

## Example Output

### Indexing:
```
=== RAG Indexing Pipeline ===
Found 16 JSON files to process...
Processing: about_us_our_team.json
Processing: investment_portfolio.json
Processing: ma_advisory_complete.json
...
Loaded documents total.

Successfully indexed 7 documents
ChromaDB collection 'company_docs' saved to './chroma_store'
```

### Querying:
```
Enter your query: What products does Acme Corp make?

===============================================================
QUERY: What products does Acme Corp make?
===============================================================

--- RESULT 1 ---
Source File: company1.json
Record ID: 0
Similarity Score: 0.8234
Content:
company_name: Acme Corporation | industry: Technology | products[0]: Software Solutions | products[1]: Cloud Services | products[2]: AI Tools | founded: 2015...
```