"""
Agentic RAG System for Alzheimer's Research

A production-ready retrieval-augmented generation system for querying
Alzheimer's disease research papers with intelligent document retrieval
and LLM-powered synthesis.

Author: Yeshwanth Satheesh
Date: October 2025
"""

import json
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer
import chromadb
from anthropic import Anthropic
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Agent:
    """
    Base agent class for LLM-powered operations.
    
    Provides common functionality for calling Claude API across different
    agent types (classifier, decomposer, synthesizer).
    """
    
    def __init__(self, anthropic_client: Anthropic) -> None:
        """
        Initialize agent with Anthropic client.
        
        Args:
            anthropic_client: Configured Anthropic API client
        """
        self.client = anthropic_client
    
    def call_claude(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Call Claude API with given prompt.
        
        Args:
            prompt: The prompt to send to Claude
            max_tokens: Maximum tokens in response (default: 1000)
        
        Returns:
            Claude's response text
        
        Raises:
            anthropic.APIError: If API call fails
        """
        message = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text


class QueryClassifierAgent(Agent):
    """
    Classifies research queries by type and complexity.
    
    Analyzes incoming queries to determine the category (biomarker, clinical, etc.),
    complexity level, and whether decomposition into sub-queries is needed.
    """
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify a research query.
        
        Args:
            query: The user's research question
        
        Returns:
            Dictionary containing:
                - category: Query type (biomarker/clinical/genetic/etc.)
                - complexity: simple/moderate/complex
                - needs_decomposition: Boolean indicating if query should be split
        
        Example:
            >>> classifier.classify("What are tau proteins?")
            {'category': 'biomarker', 'complexity': 'simple', 'needs_decomposition': False}
        """
        
        prompt = f"""Analyze this Alzheimer's research query and provide:
1. Category (biomarker/clinical/genetic/methodology/progression/treatment/general)
2. Complexity (simple/moderate/complex)
3. Whether it needs decomposition (yes/no)

Query: {query}

Return JSON only:
{{"category": "...", "complexity": "...", "needs_decomposition": true/false}}"""

        response = self.call_claude(prompt, max_tokens=200)
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            return json.loads(response[json_start:json_end])
        except:
            return {
                "category": "general",
                "complexity": "moderate",
                "needs_decomposition": False
            }


class QueryDecomposerAgent(Agent):
    """
    Decomposes complex queries into simpler sub-queries.
    
    Takes complex, multi-faceted research questions and breaks them down
    into 2-4 focused sub-queries that can be answered independently.
    """
    
    def decompose(self, query: str) -> List[str]:
        """
        Break a complex query into sub-queries.
        
        Args:
            query: Complex research question
        
        Returns:
            List of 2-4 simpler sub-queries
        
        Example:
            >>> decomposer.decompose("How do tau and amyloid interact in AD progression?")
            ["What is tau protein's role?", "What is amyloid's role?", "How do they interact?"]
        """
        
        prompt = f"""Break this complex Alzheimer's research query into 2-4 focused sub-queries.
Each sub-query should be answerable independently.

Query: {query}

Return ONLY a JSON array of sub-queries:
["sub-query 1", "sub-query 2", ...]"""

        response = self.call_claude(prompt, max_tokens=400)
        
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            return json.loads(response[json_start:json_end])
        except:
            return [query]


class RetrievalAgent:
    """
    Retrieves relevant documents from ChromaDB vector store.
    
    Uses sentence transformers to convert queries to embeddings and performs
    semantic similarity search against indexed research papers.
    """
    
    def __init__(self, chroma_path: str, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize retrieval agent.
        
        Args:
            chroma_path: Path to ChromaDB persistent storage
            model_name: Sentence transformer model name (default: all-MiniLM-L6-v2)
        """
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection("alzheimers_research")
        self.model = SentenceTransformer(model_name)
        
        logger.info(f"Loaded collection with {self.collection.count()} documents")
    
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve semantically similar documents.
        
        Args:
            query: Search query string
            n_results: Number of documents to return (default: 5)
        
        Returns:
            List of documents, each containing:
                - id: Document ID (PMID)
                - text: Full document text
                - metadata: Paper metadata (title, authors, journal, etc.)
                - distance: Similarity distance score (lower = more similar)
        
        Example:
            >>> docs = agent.retrieve("tau proteins", n_results=3)
            >>> print(docs[0]['metadata']['title'])
            'Tau protein structure and dynamics'
        """
        
        # Create query embedding
        embedding = self.model.encode([query])[0].tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        
        # Format results
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return documents


class SynthesisAgent(Agent):
    """
    Synthesizes comprehensive answers from retrieved documents.
    
    Uses Claude LLM to read retrieved research papers and generate
    natural, technically accurate answers in a conversational style.
    """
    
    def synthesize(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        sub_queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate answer from retrieved documents.
        
        Args:
            query: Original user question
            documents: List of retrieved documents with metadata
            sub_queries: Optional list of sub-queries if decomposed
        
        Returns:
            Dictionary containing:
                - answer: Generated answer text
                - sources: List of source paper metadata
                - num_sources: Number of sources used
        
        Example:
            >>> result = agent.synthesize("What are tau proteins?", docs)
            >>> print(result['answer'])
            'Tau proteins are microtubule-associated...'
        """
        
        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(documents[:5], 1):
            metadata = doc['metadata']
            context_parts.append(
                f"[Document {i}]\n"
                f"PMID: {metadata.get('pmid', 'N/A')}\n"
                f"Title: {metadata.get('title', 'N/A')}\n"
                f"Journal: {metadata.get('journal', 'N/A')}\n"
                f"Date: {metadata.get('pub_date', 'N/A')}\n"
                f"Content: {doc['text'][:500]}...\n"
            )
        
        context_text = "\n\n".join(context_parts)
        
        # Build synthesis prompt
        prompt = f"""You are an expert in Alzheimer's disease research. 
Answer the question using the provided research documents.

Question: {query}

Research Documents:
{context_text}

Instructions:
1. Provide a comprehensive, accurate answer
2. Cite specific documents using [Document N] format
3. Highlight key findings and consensus/disagreements
4. Note any limitations or gaps in the evidence
5. Use scientific terminology appropriately

Answer:"""

        if sub_queries:
            prompt += f"\n\nNote: This question was decomposed into: {', '.join(sub_queries)}"
        
        answer = self.call_claude(prompt, max_tokens=2000)
        
        return {
            'answer': answer,
            'sources': [doc['metadata'] for doc in documents[:5]],
            'num_sources': len(documents)
        }


class AgenticRAGSystem:
    """
    Complete Retrieval-Augmented Generation system.
    
    Orchestrates the full RAG pipeline: classification, decomposition,
    retrieval, and synthesis to answer research questions about Alzheimer's disease.
    
    Author: Yeshwanth Satheesh
    """
    
    def __init__(self, anthropic_api_key: str, chroma_path: str) -> None:
        """
        Initialize the RAG system.
        
        Args:
            anthropic_api_key: Anthropic API key for Claude access
            chroma_path: Path to ChromaDB vector store
        """
        anthropic_client = Anthropic(api_key=anthropic_api_key)
        
        self.classifier = QueryClassifierAgent(anthropic_client)
        self.decomposer = QueryDecomposerAgent(anthropic_client)
        self.retriever = RetrievalAgent(chroma_path)
        self.synthesizer = SynthesisAgent(anthropic_client)
    
    def query(self, user_query: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Process a research query through the full RAG pipeline.
        
        Pipeline steps:
            1. Classify query type and complexity
            2. Decompose into sub-queries if needed
            3. Retrieve relevant documents from vector store
            4. Synthesize comprehensive answer using LLM
        
        Args:
            user_query: The research question to answer
            verbose: If True, log detailed processing steps
        
        Returns:
            Dictionary containing:
                - query: Original question
                - answer: Generated answer
                - sources: Source paper metadata
                - classification: Query classification info
                - sub_queries: Sub-queries if decomposed
                - num_sources: Number of sources used
                - timestamp: ISO format timestamp
        
        Example:
            >>> rag = AgenticRAGSystem(api_key, "./chroma_db")
            >>> result = rag.query("What are tau proteins in Alzheimer's?")
            >>> print(result['answer'])
        """
        
        result = {
            'query': user_query,
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: Classify query
        classification = self.classifier.classify(user_query)
        result['classification'] = classification
        
        if verbose:
            logger.info(f"Classification: {classification}")
        
        # Step 2: Decompose if needed
        sub_queries = [user_query]
        if classification.get('needs_decomposition', False):
            sub_queries = self.decomposer.decompose(user_query)
            result['sub_queries'] = sub_queries
            
            if verbose:
                logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
        
        # Step 3: Retrieve documents for each sub-query
        all_documents = []
        for sq in sub_queries:
            docs = self.retriever.retrieve(sq, n_results=3)
            all_documents.extend(docs)
        
        # Remove duplicates by ID
        seen_ids = set()
        unique_docs = []
        for doc in all_documents:
            if doc['id'] not in seen_ids:
                seen_ids.add(doc['id'])
                unique_docs.append(doc)
        
        if verbose:
            logger.info(f"Retrieved {len(unique_docs)} unique documents")
        
        # Step 4: Synthesize answer
        synthesis = self.synthesizer.synthesize(
            user_query, 
            unique_docs,
            sub_queries if len(sub_queries) > 1 else None
        )
        
        result.update(synthesis)
        
        return result


def main() -> None:
    """
    Main entry point for command-line usage.
    
    Supports three modes:
        - Single query: --query "your question"
        - Batch queries: --query_file queries.txt
        - Interactive: No arguments (prompt for input)
    """
    parser = argparse.ArgumentParser(
        description="Query Alzheimer's research using agentic RAG"
    )
    parser.add_argument(
        "--chroma_dir",
        type=str,
        required=True,
        help="ChromaDB directory"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process"
    )
    parser.add_argument(
        "--query_file",
        type=str,
        help="File with queries (one per line)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    # Initialize system
    logger.info("Initializing agentic RAG system...")
    rag = AgenticRAGSystem(api_key, args.chroma_dir)
    
    # Get queries
    queries = []
    if args.query:
        queries = [args.query]
    elif args.query_file:
        with open(args.query_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        print("\n=== Alzheimer's Research Query System ===")
        print("Enter 'quit' to exit\n")
        while True:
            query = input("Query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query:
                queries = [query]
                result = rag.query(query, verbose=args.verbose)
                
                # Print answer - clean format
                print(f"\n{result['answer']}\n")
                
                queries = []  # Reset for next iteration
    
    # Process batch queries
    if queries:
        results = []
        for query in queries:
            logger.info(f"\nProcessing query: {query}")
            result = rag.query(query, verbose=args.verbose)
            results.append(result)
            
            # Print answer - clean format
            print(f"\n{result['answer']}\n")
        
        # Save results
        if args.output_file and results:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()