# arXiv RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system that answers questions about research papers using GPT-4o, ChromaDB, and a two-stage retrieval pipeline.

## Evaluation Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.800 |
| Answer Relevancy | 0.900 |
| Context Precision | 0.780 |
| Context Recall | 0.760 |
| **Overall Average** | **0.810** |

## Architecture
```
arXiv PDFs → PyMuPDF → Chunking (512 tokens, 50 overlap)
    → OpenAI Embeddings → ChromaDB
    → Vector Search (top-20) → Cross-Encoder Reranking (top-5)
    → GPT-4o Generation with Citations → FastAPI REST API
```

## Stack

- **LLM**: GPT-4o (generation) + GPT-4o-mini (evaluation judge)
- **Embeddings**: text-embedding-3-small
- **Vector DB**: ChromaDB (persistent, local)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **API**: FastAPI + Uvicorn
- **PDF Parsing**: PyMuPDF

## Project Structure
```
arxiv-rag/
├── src/
│   ├── ingest.py       # PDF → chunks → embeddings → ChromaDB
│   ├── retriever.py    # query → vector search → reranking
│   ├── generator.py    # chunks + question → GPT-4o answer
│   ├── pipeline.py     # orchestrates retrieve + generate
│   └── evaluate.py     # GPT-4o-mini evaluation judge
├── api/
│   └── main.py         # FastAPI REST endpoint
├── notebooks/
│   ├── eval_results.csv
│   └── eval_summary.csv
└── download_papers.py  # fetch arXiv PDFs
```

## Setup
```bash
git clone https://github.com/hcsuraj07/arxiv-rag.git
cd arxiv-rag
python -m venv venv
source venv/bin/activate
pip install openai chromadb pymupdf langchain langchain-openai \
            sentence-transformers ragas fastapi uvicorn \
            python-dotenv arxiv tqdm datasets pandas
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

## Usage
```bash
# 1. Download papers
python download_papers.py

# 2. Ingest into ChromaDB
python src/ingest.py

# 3. Start the API
uvicorn api.main:app --reload --port 8000

# 4. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What methods are used to evaluate RAG systems?"}'

# 5. Run evaluation
python src/evaluate.py
```

## API

**POST /query**
```json
{
  "question": "What are the main challenges in RAG systems?"
}
```

Response includes answer, inline citations, source papers, and token usage.

## Key Design Decisions

**Two-stage retrieval**: Vector search retrieves top-20 candidates by cosine similarity, then a cross-encoder reranks to top-5 by actual relevance. This significantly improves answer quality over single-stage retrieval.

**Citation enforcement**: System prompt forces GPT-4o to cite [SOURCE N] inline and list papers used. Every claim is traceable.

**Custom evaluator**: Uses GPT-4o-mini as a judge to score faithfulness, relevancy, precision, and recall — no dependency on unstable third-party eval libraries.

## Author

Suraj Chandra — AI/ML Engineer
[LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/hcsuraj07)
