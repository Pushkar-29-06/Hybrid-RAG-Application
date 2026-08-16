from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from typing import Optional, List
from urllib.parse import unquote
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import ollama

app = FastAPI()

# Global variables
documents:List[str] = []
bm25_index:Optional[BM25Okapi] = None
faiss_index:Optional[faiss.IndexFlatL2] = None

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder("cross-encoder/ms-macro-MiniLM-L-6-v2")


def chunk_text(text, size=500, overlap=100):
    """
    Chunk the text into smaller pieces of specified size with overlap.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = start + size-overlap

    return chunks

# Rebuild index whenever any doc or text gets uploaded
def rebuild_indexes():
    global bm25_index, faiss_index

    if not documents:
        bm25_index= None
        return

    #########################
    # Create BM25 index
    #########################
    # Tokenization
    tokenize_docs = [doc.lower().split() for doc in documents]
    bm25_index = BM25Okapi(tokenize_docs)



    ##########################
    # FAISS Index
    ##########################
    doc_embeddings = embed_model.encode(documents).astype('float32')
    faiss_index = faiss.IndexFlatL2(384)
    faiss_index.add(doc_embeddings)


# Route to Upload text or file
@app.post("/upload")
async def upload(text: Optional[str] = Body(None), file: Optional[UploadFile] = File(None)):
    if file:
        content = await file.read()
        text = content.decode('utf-8')

    if not text or not text.strip():
        raise HTTPException(status_code = 400, detail="Text or text file is required")

    if text:
        text = unquote(text)

    # Chunking document
    chunks = chunk_text(text)
    documents.extend(chunks)

    # Rebuild Indexex
    rebuild_indexes()

    return("Added Chunks:", len(chunks), "Total Documents:", len(documents))

@app.get("/delete")
async def delete_docs():
    doc_length = len(documents)
    documents.clear()
    return {
        "Total cleared docs":doc_length
    }


@app.get("/list/documents")
async def get_documents():
    return{
        "Total Length is":len(documents),
        "Documents":documents
    }


def hybrid_search(query: str = Body(...)) -> List[str]:
    if not documents or bm25_index is None or faiss_index is None:
        return []

    # Sparse using BM25 Index
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores()

    top_indexes = np.argsort(bm25_scores)[-2:][::-1]
    
    bm25_result = [documents[i] for i in top_indexes]

    return bm25_results

    # Dense Indexing using Faiss
    query_embedding = embed_model.encode(query).astype("float32").reshape(1,-1)
    _,dense_indexes = faiss.search(query_embedding, 2)

    dense_result = [documents[i] for i in dense_indexes[0]]

    # Combine both bm25 and faiss reuslts 
    return lists(set(bm25_result + dense_result))

# ReRank for Cross-Encoding
def rerank(query: str, docs:list[str], top_k = 2):
    if not docs:
        return []

    # Create Pairs
    pairs = [[query, doc] for doc in docs]

    # Get relevant socres 
    scores = reranker.predict(pairs)

    # Sort by score [High first] and return top_k
    return [doc for doc, _ in sorted(zip(docs, socres), key = lambda x:x[1], reverse=True)][:top_k]


# RAG Search API
@app.post("/rag/seacrh")
async def rag_search(query: str = Body(...)):
    if not query.strip():
        raise HTTPException(400, "Query is Required!")

    if not bm25_index or not faiss_index:
        raise HTTPException(400, "Upload Document First")

    # Step - 1
    hybrid_candidates = hybrid_seacrh(query)
    return hybrid_candidates

    # Step - 2 (ReRanking using Cross-Encoder)
    top_docs = rerank(query, hybrid_candidates, 2)

    # Step - 3 (Prompt Buildingn with context for the LLM)
    context = "\n".join(f"{i+1}.{doc}" for i, doc in enumerate{top_docs})
    prompt = f"""Answer based only context below
    Context : {context}

    Question : {query}

    Answer :"""

    # Step - 4 Send Prompt to local llm or OpenAI
    response = ollama.chat(model="deepseek-r1:1.5b", messages = [{'role':'user', 'content':prompt}])

    return {
        "query":query,
        "answer":response['message']['content'],
        "context":top_docs
    }
if __name__ == "__main":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port="8000")