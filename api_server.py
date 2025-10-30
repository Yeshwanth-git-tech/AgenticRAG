"""
FastAPI Server for Alzheimer's RAG System
Production-ready API with full LLM integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import argparse
import logging
from datetime import datetime
import os
import sys

# Import the RAG system
from query_system import AgenticRAGSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Alzheimer's Research RAG API",
    description="AI-powered research assistant for Alzheimer's disease papers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system
rag_system: Optional[AgenticRAGSystem] = None
stats = {
    "queries_processed": 0,
    "start_time": None,
    "last_query_time": None,
    "total_documents": 0
}


# Pydantic Models
class QueryRequest(BaseModel):
    """Request model for queries"""
    question: str
    n_results: int = 5
    verbose: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are tau proteins in Alzheimer's disease?",
                "n_results": 5,
                "verbose": False
            }
        }


class SourceDocument(BaseModel):
    """Source document metadata"""
    pmid: str
    title: str
    authors: Optional[str] = None
    journal: Optional[str] = None
    pub_date: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for queries"""
    question: str
    answer: str
    sources: List[SourceDocument]
    classification: Dict[str, Any]
    sub_queries: Optional[List[str]] = None
    num_sources: int
    processing_time_seconds: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    total_documents: int
    queries_processed: int
    uptime_seconds: float
    last_query_time: Optional[str]


class StatsResponse(BaseModel):
    """System statistics"""
    queries_processed: int
    start_time: str
    last_query_time: Optional[str]
    total_documents: int
    uptime_seconds: float


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system, stats
    
    logger.info("=" * 80)
    logger.info("🚀 Starting Alzheimer's Research RAG API Server")
    logger.info("=" * 80)
    
    # Get configuration
    api_key = os.getenv("ANTHROPIC_API_KEY")
    chroma_dir = os.getenv("CHROMA_DIR", "./chroma_db")
    
    if not api_key:
        logger.error("❌ ANTHROPIC_API_KEY not set!")
        logger.error("Set it with: export ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)
    
    logger.info(f"📂 ChromaDB directory: {chroma_dir}")
    logger.info("🔄 Loading RAG system (this may take 15-20 seconds)...")
    
    try:
        rag_system = AgenticRAGSystem(api_key, chroma_dir)
        stats["start_time"] = datetime.now()
        stats["total_documents"] = rag_system.retriever.collection.count()
        
        logger.info("✅ RAG system initialized successfully!")
        logger.info(f"📚 Loaded {stats['total_documents']} documents")
        logger.info("=" * 80)
        logger.info("🌐 API is ready to accept requests!")
        logger.info("📖 Interactive docs: http://localhost:8000/docs")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        sys.exit(1)


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Alzheimer's Research RAG API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "query": "POST /api/query - Ask questions about Alzheimer's research",
            "health": "GET /api/health - Check system health",
            "stats": "GET /api/stats - Get usage statistics",
            "docs": "GET /docs - Interactive API documentation"
        },
        "total_documents": stats["total_documents"],
        "queries_processed": stats["queries_processed"]
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns system status and basic metrics
    """
    if rag_system is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG system not initialized"
        )
    
    uptime = (datetime.now() - stats["start_time"]).total_seconds()
    
    return HealthResponse(
        status="healthy",
        total_documents=stats["total_documents"],
        queries_processed=stats["queries_processed"],
        uptime_seconds=round(uptime, 2),
        last_query_time=stats["last_query_time"]
    )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get detailed system statistics
    """
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    uptime = (datetime.now() - stats["start_time"]).total_seconds()
    
    return StatsResponse(
        queries_processed=stats["queries_processed"],
        start_time=stats["start_time"].isoformat(),
        last_query_time=stats["last_query_time"],
        total_documents=stats["total_documents"],
        uptime_seconds=round(uptime, 2)
    )


@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system
    
    This endpoint processes your question through:
    1. Query classification (what type of question)
    2. Document retrieval (find relevant papers)
    3. Answer synthesis (generate comprehensive answer with LLM)
    
    Returns a detailed answer with citations to source papers.
    """
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    start_time = datetime.now()
    logger.info(f"📝 New query: {request.question}")
    
    try:
        # Process query through RAG system
        result = rag_system.query(
            request.question,
            verbose=request.verbose
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Update statistics
        stats["queries_processed"] += 1
        stats["last_query_time"] = end_time.isoformat()
        
        # Format sources
        sources = [
            SourceDocument(
                pmid=src.get('pmid', 'N/A'),
                title=src.get('title', 'N/A'),
                authors=src.get('authors'),
                journal=src.get('journal'),
                pub_date=src.get('pub_date')
            )
            for src in result['sources']
        ]
        
        # Build response
        response = QueryResponse(
            question=request.question,
            answer=result['answer'],
            sources=sources,
            classification=result['classification'],
            sub_queries=result.get('sub_queries'),
            num_sources=result['num_sources'],
            processing_time_seconds=round(processing_time, 2),
            timestamp=end_time.isoformat()
        )
        
        logger.info(f"✅ Query completed in {processing_time:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/api/documents/count")
async def get_document_count():
    """Get total number of indexed documents"""
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    return {
        "total_documents": stats["total_documents"]
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "available_endpoints": {
            "root": "GET /",
            "query": "POST /api/query",
            "health": "GET /api/health",
            "stats": "GET /api/stats",
            "docs": "GET /docs"
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Alzheimer's RAG API Server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)"
    )
    parser.add_argument(
        "--chroma_dir",
        default="./chroma_db",
        help="ChromaDB directory path"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Set environment variables
    if not os.getenv("CHROMA_DIR"):
        os.environ["CHROMA_DIR"] = args.chroma_dir
    
    # Start server
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()