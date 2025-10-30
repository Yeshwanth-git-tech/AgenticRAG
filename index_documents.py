"""
Document Indexing System for Alzheimer's Research

Processes PubMed JSON files and creates a searchable vector database
using sentence transformers and ChromaDB for semantic search.

Author: Yeshwanth Satheesh
Date: October 2025
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Indexes research documents with vector embeddings.
    
    Processes PubMed JSON files, creates semantic embeddings using
    sentence transformers, and stores them in ChromaDB for fast retrieval.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the document indexer.
        
        Args:
            model_name: Sentence transformer model name (default: all-MiniLM-L6-v2)
                       This model produces 384-dimensional embeddings.
        """
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully")
        
    def parse_pubmed_json(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Parse a PubMed JSON file into structured format.
        
        Extracts key fields (title, abstract, keywords, MeSH terms) and
        combines them into searchable text with metadata.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            Dictionary containing:
                - id: Document identifier (PMID)
                - text: Combined searchable text
                - metadata: Paper metadata (authors, journal, etc.)
            Returns None if parsing fails.
        
        Example:
            >>> indexer.parse_pubmed_json("paper.json")
            {'id': '40973401', 'text': 'Tau protein...', 'metadata': {...}}
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create searchable text
            title = data.get('TI', '')
            abstract = data.get('AB', '')
            keywords = ' '.join(data.get('OT', []))
            mesh_terms = ' '.join(data.get('MH', []))
            
            text = f"{title}\n{abstract}\n{keywords}\n{mesh_terms}"
            
            return {
                'id': data.get('PMID', str(filepath)),
                'text': text,
                'metadata': {
                    'pmid': data.get('PMID', ''),
                    'title': title,
                    'authors': ', '.join(data.get('AU', [])[:5]),
                    'journal': data.get('TA', ''),
                    'pub_date': data.get('DP', ''),
                    'keywords': ', '.join(data.get('OT', [])),
                    'mesh_terms': ', '.join(data.get('MH', [])[:10]),
                    'abstract_length': len(abstract)
                }
            }
        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return None
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create vector embeddings for documents.
        
        Converts text documents into 384-dimensional vectors that capture
        semantic meaning for similarity search.
        
        Args:
            texts: List of document texts to embed
        
        Returns:
            List of embedding vectors (each is a list of 384 floats)
        
        Example:
            >>> embeddings = indexer.create_embeddings(["tau proteins", "amyloid plaques"])
            >>> len(embeddings[0])  # 384
        """
        logger.info(f"Creating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def index_directory(self, data_dir: str, output_dir: str) -> int:
        """
        Index all JSON files in a directory.
        
        Complete indexing pipeline:
            1. Initialize ChromaDB
            2. Parse all JSON files
            3. Create embeddings
            4. Store in vector database
        
        Args:
            data_dir: Directory containing PubMed JSON files
            output_dir: Directory to store ChromaDB database
        
        Returns:
            Number of documents successfully indexed
        
        Raises:
            ValueError: If no JSON files found or parsing fails
        
        Example:
            >>> indexer.index_directory("./papers/", "./chroma_db/")
            100  # Successfully indexed 100 documents
        """
        
        # Initialize ChromaDB
        logger.info(f"Creating ChromaDB at: {output_dir}")
        client = chromadb.PersistentClient(path=output_dir)
        
        # Create or get collection
        try:
            client.delete_collection("alzheimers_research")
            logger.info("Deleted existing collection")
        except:
            pass
        
        collection = client.create_collection(
            name="alzheimers_research",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Indexing documents from: {data_dir}")
        
        # Get all JSON files
        json_files = list(Path(data_dir).glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files")
        
        if len(json_files) == 0:
            logger.error("No JSON files found!")
            return 0
        
        # Process all documents
        all_docs = []
        all_texts = []
        all_ids = []
        all_metadata = []
        
        for filepath in json_files:
            doc = self.parse_pubmed_json(str(filepath))
            
            if doc is None:
                continue
            
            all_docs.append(doc['text'])
            all_texts.append(doc['text'])
            all_ids.append(doc['id'])
            all_metadata.append(doc['metadata'])
        
        if not all_docs:
            logger.error("No valid documents parsed!")
            return 0
        
        logger.info(f"Successfully parsed {len(all_docs)} documents")
        
        # Create embeddings for all documents at once
        embeddings = self.create_embeddings(all_texts)
        
        # Add to ChromaDB
        logger.info(f"Adding {len(all_docs)} documents to ChromaDB...")
        collection.add(
            ids=all_ids,
            embeddings=embeddings,
            documents=all_docs,
            metadatas=all_metadata
        )
        
        logger.info(f"✅ Indexing complete! Total documents: {collection.count()}")
        
        return collection.count()


def main() -> None:
    """
    Main entry point for document indexing.
    
    Command-line interface for indexing PubMed JSON files into
    a searchable vector database.
    
    Usage:
        python index_documents.py --data_dir ./papers/ --output_dir ./chroma_db/
    
    Author: Yeshwanth Satheesh
    """
    parser = argparse.ArgumentParser(
        description="Index Alzheimer's research documents"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./chroma_db",
        help="Output directory for vector database"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name"
    )
    
    args = parser.parse_args()
    
    # Create indexer
    indexer = DocumentIndexer(model_name=args.model)
    
    # Index documents
    num_docs = indexer.index_directory(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    logger.info(f"✅ Successfully indexed {num_docs} documents")
    logger.info(f"📁 Vector database saved to: {args.output_dir}")


if __name__ == "__main__":
    main()