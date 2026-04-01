from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm import get_rag_chain
from app.services.ingestion import ingest_website

router = APIRouter()

class ChatRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    url: str
    max_depth: int = 1

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        chain = get_rag_chain()
        result = chain.invoke({"query": request.query})
        return {
            "answer": result["result"],
            "sources": [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest/website")
async def ingest(request: IngestRequest):
    try:
        num_chunks = ingest_website(request.url, request.max_depth)
        return {"message": f"Successfully ingested {num_chunks} chunks from {request.url}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
