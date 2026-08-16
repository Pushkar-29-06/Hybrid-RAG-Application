# Hybrid RAG Application

A advanced Retrieval-Augmented Generation (RAG) system that combines sparse and dense search techniques with cross-encoder reranking to provide accurate, context-aware answers using local LLMs.

## Overview

This application implements a hybrid RAG pipeline that leverages both keyword-based (BM25) and semantic (FAISS) search methods to retrieve relevant document chunks. The retrieved results are re-ranked using a cross-encoder model to ensure the most relevant context is passed to a local LLM for answer generation. The system is designed to run entirely locally, ensuring privacy and eliminating dependency on external API services.

The project addresses the limitation of single-method retrieval systems by combining multiple search strategies, reducing the chance of missing relevant information while improving retrieval accuracy through intelligent reranking.

## Features

- **Document Ingestion**: Upload text directly or via text files
- **Intelligent Chunking**: Automatic text segmentation with configurable overlap to maintain context
- **Dual Indexing**: 
  - Sparse indexing using BM25 for keyword-based retrieval
  - Dense indexing using FAISS for semantic similarity search
- **Document Management**: List and delete uploaded documents
- **Local LLM Integration**: Uses Ollama for private, on-device inference
- **RESTful API**: FastAPI-based backend for easy integration

## Architecture

```mermaid
graph TD
    A[User] -->|Upload Text/File| B[FastAPI Backend]
    B --> C[Text Chunker]
    C --> D[Document Store]
    D --> E[BM25 Index Builder]
    D --> F[FAISS Index Builder]
    E --> G[Sparse Search BM25]
    F --> H[Dense Search FAISS]
    G --> I[Hybrid Search Combiner]
    H --> I
    I --> J[Cross-Encoder Reranker]
    J --> K[Context Builder]
    K --> L[Ollama LLM]
    L --> M[Response]
    M --> A
```

## How It Works

### Document Ingestion Pipeline

1. **Upload**: User submits text or a text file via the `/upload` endpoint
2. **Chunking**: The text is divided into smaller chunks (default 500 characters) with 100-character overlap to preserve context across boundaries
3. **Indexing**: 
   - Chunks are tokenized and indexed using BM25 for keyword search
   - Chunks are converted to 384-dimensional embeddings using `all-MiniLM-L6-v2` and stored in a FAISS index
4. **Storage**: Chunks are stored in memory for retrieval

### Query Processing Pipeline

1. **Query Reception**: User submits a question via the `/rag/search` endpoint
2. **Hybrid Search**: 
   - BM25 retrieves top chunks based on keyword matching
   - FAISS retrieves top chunks based on semantic similarity
   - Results are combined and deduplicated
3. **Reranking**: Cross-encoder model scores each query-document pair to identify the most relevant chunks
4. **Context Construction**: Top-ranked chunks are formatted into a context string
5. **LLM Generation**: The context and query are sent to a local LLM (via Ollama) to generate the final answer
6. **Response**: The answer along with the used context is returned to the user

## Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python |
| **Web Framework** | FastAPI |
| **ASGI Server** | Uvicorn |
| **Sparse Search** | rank_bm25 |
| **Dense Search** | FAISS (CPU) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Reranking** | sentence-transformers (cross-encoder/ms-macro-MiniLM-L-6-v2) |
| **LLM** | Ollama (deepseek-r1:1.5b) |
| **Numerical Computing** | NumPy |
| **License** | MIT |

## Project Structure

```
RAG/
├── main.py              # FastAPI application with RAG pipeline
├── requirements.txt     # Python dependencies
├── LICENSE             # MIT License
├── .gitignore          # Git ignore rules
├── .venv/              # Virtual environment (not tracked)
└── README.md           # This file
```

### File Descriptions

- **main.py**: Core application containing the FastAPI app, document processing logic, indexing functions, hybrid search, reranking, and RAG endpoint
- **requirements.txt**: All Python package dependencies with pinned versions
- **LICENSE**: MIT license for the project
- **.gitignore**: Excludes virtual environment, cache files, and environment variables

## Installation

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally
- Ollama model `deepseek-r1:1.5b` pulled (or modify code to use a different model)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RAG
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   
   On Windows:
   ```bash
   .venv\Scripts\activate
   ```
   
   On macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start Ollama** (if not already running)
   ```bash
   ollama serve
   ```

6. **Pull the required model** (if not already available)
   ```bash
   ollama pull deepseek-r1:1.5b
   ```

7. **Run the application**
   ```bash
   python main.py
   ```

   The server will start on `http://0.0.0.0:8000`

## API Endpoints

### Upload Document

**Endpoint**: `POST /upload`

**Description**: Upload text or a text file to be indexed

**Request Body** (form-data):
- `text` (optional): Raw text string
- `file` (optional): Text file

**Response**: JSON with chunk count and total documents

**Example**:
```bash
curl -X POST "http://localhost:8000/upload" -F "text=Your document text here"
```

### Delete All Documents

**Endpoint**: `GET /delete`

**Description**: Clear all uploaded documents and reset indexes

**Response**: JSON with count of cleared documents

**Example**:
```bash
curl -X GET "http://localhost:8000/delete"
```

### List Documents

**Endpoint**: `GET /list/documents`

**Description**: Retrieve all currently stored document chunks

**Response**: JSON with total count and document list

**Example**:
```bash
curl -X GET "http://localhost:8000/list/documents"
```

### RAG Search

**Endpoint**: `POST /rag/search`

**Description**: Perform hybrid search and generate answer using LLM

**Request Body** (JSON):
```json
{
  "query": "Your question here"
}
```

**Response**: JSON with query, generated answer, and context used

**Example**:
```bash
curl -X POST "http://localhost:8000/rag/search" -H "Content-Type: application/json" -d "{\"query\": \"What is the main topic?\"}"
```

## Configuration

### Models Used

- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Cross-Encoder**: `cross-encoder/ms-macro-MiniLM-L-6-v2` (for reranking)
- **LLM**: `deepseek-r1:1.5b` (via Ollama)

### Chunking Parameters

- **Chunk Size**: 500 characters
- **Overlap**: 100 characters

### Retrieval Parameters

- **BM25 Top K**: 2 documents
- **FAISS Top K**: 2 documents
- **Reranker Top K**: 2 documents

These parameters can be modified in `main.py` to suit different use cases.

## Environment Variables

Currently, the application does not require any environment variables. All configuration is done directly in the code.

If you wish to use a different Ollama model, modify the `model` parameter in the `ollama.chat()` call in `main.py` (line 165).

## Usage Example

1. **Start the server**
   ```bash
   python main.py
   ```

2. **Upload a document**
   ```bash
   curl -X POST "http://localhost:8000/upload" -F "text=Artificial intelligence is transforming industries worldwide. Machine learning, a subset of AI, enables computers to learn from data. Deep learning uses neural networks with multiple layers to process complex patterns."
   ```

3. **Perform a search**
   ```bash
   curl -X POST "http://localhost:8000/rag/search" -H "Content-Type: application/json" -d "{\"query\": \"What is machine learning?\"}"
   ```

4. **Expected Response**
   ```json
   {
     "query": "What is machine learning?",
     "answer": "Based on the context, machine learning is a subset of AI that enables computers to learn from data.",
     "context": ["Machine learning, a subset of AI, enables computers to learn from data."]
   }
   ```

## Future Improvements

- **Persistent Storage**: Currently documents are stored in-memory; adding database persistence would allow data to survive server restarts
- **Async Processing**: Implement async document processing for large files
- **Batch Upload**: Support uploading multiple documents at once
- **Configuration File**: Move configuration to a separate config file or environment variables
- **Additional LLM Support**: Add support for other local LLM providers or OpenAI
- **Frontend Interface**: Build a web UI for easier interaction
- **Streaming Responses**: Implement streaming for LLM responses
- **Metadata Support**: Add document metadata (source, timestamp, etc.)
- **Advanced Chunking**: Implement semantic chunking or recursive chunking strategies
- **Evaluation Metrics**: Add retrieval and generation quality metrics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built following the comprehensive Hindi tutorial on building advanced RAG pipelines, combining best practices from the retrieval-augmented generation community.
