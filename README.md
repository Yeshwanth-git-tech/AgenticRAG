# AgenticRAG: AI-Powered Research Assistant for Alzheimer's Disease

An intelligent Retrieval-Augmented Generation (RAG) system that enables natural language queries over Alzheimer's disease research papers using semantic search and large language models.

**Author:** Yeshwanth Satheesh  
**Institution:** Indiana University  
**Date:** October 2025

---

## 🎯 Overview

AgenticRAG is a production-ready system that combines:
- **Vector Search** (ChromaDB + Sentence Transformers)
- **Agentic AI** (Query classification, decomposition, synthesis)
- **Large Language Models** (Claude Sonnet 4.5)
- **REST API** (FastAPI)

to provide conversational, citation-backed answers to research questions about Alzheimer's disease.

### Key Features

**Semantic Search**: Find relevant papers by meaning, not just keywords  
**Intelligent Query Processing**: Classifies and decomposes complex questions  
**Natural Language Answers**: Generates comprehensive, technical explanations  
**Source Citations**: Every answer backed by research papers (PMIDs)  
**Fast Retrieval**: ~0.1s search across 100+ papers  
**RESTful API**: Easy integration with frontends

---

## 📁 Repository Structure

```
AgenticRAG/
├── scripts/
│   ├── index_documents.py      # Document indexing pipeline
│   ├── query_system.py         # Agentic RAG query system
│   └── api_server.py           # FastAPI REST server
├── data/
│   └── sample_papers/          # Sample PubMed JSON files
├── docs/
│   ├── architecture.png        # System architecture diagram
│   ├── bigred_demo.png         # BigRed deployment screenshot
│   └── api_example.png         # API response example
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   USER QUERY                            │
│         "What are tau proteins in AD?"                  │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│              AGENTIC RAG PIPELINE                       │
│                                                         │
│  1. Query Classifier → Determines type & complexity    │
│  2. Query Decomposer → Breaks complex queries          │
│  3. Vector Retrieval → Semantic search (ChromaDB)      │
│  4. LLM Synthesis    → Generate answer (Claude)        │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│            COMPREHENSIVE ANSWER                         │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
PubMed JSON Files (100 papers)
         ↓
[index_documents.py]
         ↓
Vector Embeddings (384-dim)
         ↓
ChromaDB Storage
         ↓
[query_system.py] ← User Query
         ↓
Retrieved Documents
         ↓
[Claude LLM]
         ↓
Natural Language Answer
```

---

##  Data Storage: ChromaDB

### What is ChromaDB?

ChromaDB is a vector database that stores:
1. **Document embeddings** (384-dimensional vectors)
2. **Original text** (full paper content)
3. **Metadata** (PMID, title, authors, journal, date)

### Storage Structure

```
chroma_db/
├── chroma.sqlite3              # Main database file (~50-100 MB)
├── index/
│   └── hnsw_index              # Fast similarity search index
└── metadata/
    └── document_metadata.db    # Paper metadata
```

### How Documents are Stored

Each paper is converted into:

```python
{
    'id': '40973401',  # PMID
    'embedding': [0.234, -0.156, 0.872, ...],  # 384 numbers
    'text': 'PMID: 40973401\nTitle: Tau protein...',  # Full text
    'metadata': {
        'title': 'Tau protein structure and dynamics',
        'authors': 'Chinnathambi S, Velmurugan G',
        'journal': 'Adv Protein Chem Struct Biol',
        'pub_date': '2025'
    }
}
```

### Why ChromaDB?

- **Fast**: HNSW indexing for sub-second search
- **Compact**: 100 papers = ~1 MB compressed
- **Portable**: Single directory, easy to copy
- **No server**: Embedded database, no separate process

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- 16GB RAM
- Anthropic API key

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/AgenticRAG.git
cd AgenticRAG

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup

```bash
# Set API key
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Index documents (one-time setup)
python scripts/index_documents.py \
    --data_dir data/sample_papers \
    --output_dir chroma_db

# Expected: "✅ Indexing complete! Total documents: 100"
```

### Usage

#### Option 1: Command Line

```bash
# Interactive mode
python scripts/query_system.py --chroma_dir chroma_db --verbose

# Single query
python scripts/query_system.py \
    --chroma_dir chroma_db \
    --query "What are tau proteins in Alzheimer's disease?"
```

#### Option 2: REST API

```bash
# Start server
python scripts/api_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --chroma_dir chroma_db

# Server starts at: http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

#### Option 3: Python API

```python
from query_system import AgenticRAGSystem

# Initialize
rag = AgenticRAGSystem(
    anthropic_api_key="your-key",
    chroma_path="./chroma_db"
)

# Query
result = rag.query("What are tau proteins?")
print(result['answer'])
```

---

## 🧪 Example Query

**Input:**
```json
{
  "question": "What are tau proteins in Alzheimer's disease?",
  "n_results": 3
}
```

**Output:**
```
Tau proteins are microtubule-associated proteins that stabilize the 
neuronal cytoskeleton. In Alzheimer's disease, tau undergoes 
hyperphosphorylation, causing it to detach from microtubules and 
aggregate into neurofibrillary tangles - one of the hallmark 
pathologies of AD.

This pathological transformation involves the formation of toxic 
oligomers that progress into highly ordered paired helical filaments. 
Research shows that cellular defense mechanisms including chaperone 
complexes, the ubiquitin system, and various proteases attempt to 
clear these abnormal tau proteins, but their failure leads to 
progressive neuronal death and the clinical symptoms of memory 
deficits, behavioral changes, and cognitive decline characteristic 
of Alzheimer's disease.

Sources: PMIDs 40973401, 40973404, 40610075
```

---

## 🖥️ Deployment on BigRed (IU HPC)

### Current Setup

The system is deployed on Indiana University's BigRed 200 supercomputer:

```bash
# Location
/N/slate/ysathees/alzheimers_rag/

# Structure
alzheimers_rag/
├── alz/                    # Virtual environment
├── chroma_db/             # Vector database
├── scripts/               # Python scripts
└── logs/                  # Server logs
```

### Running on BigRed

```bash
# SSH to BigRed
ssh ysathees@bigred200.uits.iu.edu

# Load Python
module load python/3.11

# Activate environment
cd /N/slate/ysathees/alzheimers_rag
source alz/bin/activate

# Start server (using screen for persistence)
screen -S rag_api
python scripts/api_server.py --chroma_dir chroma_db
# Ctrl+A, D to detach

# Reconnect anytime
screen -r rag_api
```

### Performance on BigRed

| Operation | Time |
|-----------|------|
| Model loading (first time) | ~15-20s |
| Query vectorization | ~0.3s |
| ChromaDB search (100 docs) | ~0.1s |
| LLM synthesis | ~3-5s |
| **Total per query** | **~4-6s** |

### Limitations

- ⚠️ Not meant for 24/7 production hosting
- ⚠️ SSH tunnel required for external access
- ⚠️ Job time limits (max 7 days)

---

## 🌐 Future Work: Production Deployment

For permanent, publicly accessible deployment, consider:

### Option 1: Jetstream2 (IU Cloud) ⭐ RECOMMENDED

**Jetstream2** is IU's free research cloud infrastructure.

#### Setup Steps:

1. **Request Allocation**
   - Visit: https://jetstream-cloud.org/
   - Apply for "Startup Allocation" (free!)
   - Project: "Alzheimer's RAG API"

2. **Create VM Instance**
   ```
   - OS: Ubuntu 22.04
   - Size: m3.medium (4 vCPU, 16GB RAM)
   - Storage: 50GB
   - Public IP: Yes
   ```

3. **Deploy**
   ```bash
   # On Jetstream2 VM
   sudo apt update
   sudo apt install python3.11 python3-pip
   
   # Copy project
   scp -r alzheimers_rag/* user@jetstream-vm:/home/user/rag/
   
   # Install & run
   cd /home/user/rag
   pip install -r requirements.txt
   
   # Run with systemd (auto-restart)
   sudo systemctl start rag-api
   ```

4. **Access**
   - Public URL: `http://jetstream-vm-ip:8000`
   - No SSH tunnel needed!
   - 24/7 uptime

**Cost:** FREE for research  
**Support:** IU Research Technologies (rt@iu.edu)

---

### Option 2: AWS EC2

#### Quick Deploy:

```bash
# 1. Launch EC2 instance
# - Type: t3.medium (2 vCPU, 4GB RAM)
# - OS: Ubuntu 22.04
# - Cost: ~$30/month

# 2. Configure security group
# - Allow port 8000 (HTTP)
# - Allow port 22 (SSH)

# 3. Deploy
ssh -i key.pem ubuntu@ec2-instance
git clone https://github.com/yourusername/AgenticRAG.git
cd AgenticRAG
pip install -r requirements.txt
nohup python scripts/api_server.py &

# 4. Access
# http://your-ec2-ip:8000
```

**Estimated Cost:** $25-40/month

---

### Option 3: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY scripts/ ./scripts/
COPY chroma_db/ ./chroma_db/

ENV ANTHROPIC_API_KEY=""
EXPOSE 8000

CMD ["python", "scripts/api_server.py", "--host", "0.0.0.0", "--chroma_dir", "chroma_db"]
```

```bash
# Build & run
docker build -t agentic-rag .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY="key" agentic-rag
```

Deploy to:
- AWS ECS/Fargate
- Google Cloud Run
- DigitalOcean App Platform

---

## 📊 Performance & Scalability

### Current Capacity

| Metric | Value |
|--------|-------|
| Documents indexed | 100 papers |
| Vector dimensions | 384 |
| ChromaDB size | ~1 MB |
| Query latency | 4-6 seconds |
| Concurrent users | 1-5 (single instance) |

### Scaling Strategies

**For 1,000 papers:**
- Same setup, query time: ~4-6s
- ChromaDB size: ~10 MB

**For 10,000 papers:**
- Query time: ~5-7s
- ChromaDB size: ~100 MB
- Consider GPU for embeddings

**For 100,000+ papers:**
- Use FAISS instead of ChromaDB
- GPU-accelerated embeddings
- Caching layer (Redis)
- Load balancer for API

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | Convert text to vectors |
| **Vector DB** | ChromaDB | Fast similarity search |
| **LLM** | Claude Sonnet 4.5 (Anthropic) | Answer synthesis |
| **API Framework** | FastAPI | REST API |
| **Language** | Python 3.11 | Core implementation |

---

## 📝 API Reference

### Endpoints

#### `POST /api/query`

Query the RAG system.

**Request:**
```json
{
  "question": "What are tau proteins?",
  "n_results": 5,
  "verbose": false
}
```

**Response:**
```json
{
  "question": "What are tau proteins?",
  "answer": "Tau proteins are microtubule-associated...",
  "sources": [...],
  "classification": {"category": "biomarker", "complexity": "simple"},
  "num_sources": 3,
  "processing_time_seconds": 4.5,
  "timestamp": "2025-10-28T15:30:00"
}
```

#### `GET /api/health`

Check system health.

**Response:**
```json
{
  "status": "healthy",
  "total_documents": 100,
  "queries_processed": 42,
  "uptime_seconds": 3600.5
}
```

Full API documentation: `http://localhost:8000/docs`

---

## 🔬 Research Applications

This system can be used for:

- 📚 **Literature Review**: Quickly understand research domains
- 🔍 **Hypothesis Generation**: Discover connections between concepts
- 📊 **Meta-Analysis**: Aggregate findings across papers
- 🎓 **Education**: Interactive learning about Alzheimer's research
- 🤖 **AI Research**: Study RAG system design patterns

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add more data sources (beyond PubMed)
- [ ] Implement caching for common queries
- [ ] Add support for PDF uploads
- [ ] Create web frontend
- [ ] Multi-language support
- [ ] Fine-tune embeddings on domain data

---

## 📄 License

This project is for academic research purposes. 

**Data:** PubMed papers are public domain  
**Code:** MIT License (or specify your license)

---

## 📧 Contact

**Yeshwanth Satheesh**  
Indiana University  
Email: ysathees@iu.edu  


---

## Acknowledgments

- **Indiana University** for BigRed computing resources
- **Anthropic** for Claude API
- **ChromaDB** for vector database
- **Sentence Transformers** for embeddings
- **Course**: [Course Name/Number] - [Instructor Name]

---

## References

1. Lewis, P. et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
2. ChromaDB Documentation: https://docs.trychroma.com/
3. Sentence Transformers: https://www.sbert.net/
4. Anthropic Claude: https://www.anthropic.com/

---

**Last Updated:** October 2025  
**Version:** 1.0.0