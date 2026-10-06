# Financial Intelligence RAG System

A Retrieval-Augmented Generation (RAG) system that answers natural-language questions about public companies using their actual SEC 10-K filings — grounding every answer in real filing text instead of relying on an LLM's general (and often outdated) knowledge.

Built as a final-year engineering project to demonstrate applied RAG architecture: document ingestion, chunking, embedding, vector retrieval, and grounded generation.

## What it does

Ask a question like:

> "What are Apple's main risk factors?"

...and the system:
1. Searches Apple's actual 2025 10-K filing for the most relevant passages
2. Feeds those passages to an LLM as context
3. Returns an answer grounded only in that retrieved text — not the model's training data

It also supports **cross-company comparison questions**, synthesizing answers across multiple companies' filings even when the source documents don't directly compare themselves.

## Example output

**Query:** `"Which of these companies faces the most regulatory risk?"`

> **Answer: Apple faces the most regulatory risk.**
>
> | Company | Key regulatory-risk themes | Breadth / intensity |
> |---|---|---|
> | Apple | Antitrust litigation worldwide, data-privacy (DMA, EU rules), AI/ML regulation, supply-chain safety, pandemic restrictions | Broadest — six distinct regulatory domains |
> | Microsoft | Antitrust enforcement (US, EU, UK, China), data-privacy compliance (GDPR) | Concentrated in two main domains |
> | Tesla | Vehicle/infrastructure safety compliance, ESG scrutiny, foreign market conditions | Geographically wide but less regulatory breadth than Apple |
>
> *(Full reasoning in the system's actual output — see `/notebooks`)*

## Architecture

```
PDF filings (Apple, Microsoft, Tesla 10-Ks)
        │
        ▼
 1. Load & extract text ───────── pypdf
        │
        ▼
 2. Chunk into ~1000-char pieces ─ RecursiveCharacterTextSplitter (200-char overlap)
        │
        ▼
 3. Embed chunks into vectors ──── all-MiniLM-L6-v2 (local, sentence-transformers)
        │
        ▼
 4. Store & index ──────────────── ChromaDB (persistent vector store)
        │
        ▼
 5. Retrieve + Generate ────────── top-k similarity search → Groq LLM (openai/gpt-oss-20b)
        │
        ▼
   Grounded answer
```

### Why each piece

- **Chunking**: 10-K filings run 65–140+ pages — too long to pass to an LLM directly. Chunking with overlap keeps related sentences together across chunk boundaries.
- **Local embeddings**: `all-MiniLM-L6-v2` runs on-device, so every chunk gets a vector representing its meaning without needing an API call for ingestion.
- **ChromaDB**: stores those vectors and lets the system find the *k* most semantically similar chunks to any question — similarity search instead of keyword search.
- **Hosted generation via Groq**: the generation model is far too large to run locally, so it's called via API. The embedding step stays local because that model is small enough to run on a laptop.

## Key design decision: multi-company comparison

Early testing showed that naive retrieval (top-5 chunks across all companies, unlabeled) failed on comparison questions — the model would say "the context doesn't mention other companies," because unlabeled chunks gave it no way to attribute facts to a specific company, and because no single 10-K compares itself to competitors.

**Fix:** retrieve a balanced set of chunks *per company*, label each section explicitly (`--- APPLE ---`, `--- MICROSOFT ---`, etc.), and instruct the model to synthesize the comparison itself from the labeled sections rather than search for a pre-existing one. This is implemented as a separate `rag_compare()` function alongside the single-company `rag_query()`.

## Tech stack

| Component | Tool |
|---|---|
| PDF parsing | `pypdf` |
| Text chunking | `langchain-text-splitters` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector database | `chromadb` |
| LLM generation | Groq API (`openai/gpt-oss-20b`) |
| Environment | Python, Jupyter (VS Code) |

## Setup

```bash
pip install pypdf langchain-text-splitters sentence-transformers chromadb groq python-dotenv
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```
(Get a free key at [console.groq.com](https://console.groq.com))

Run the notebooks in order:
1. `01_load_and_explore.ipynb` — load, chunk, embed, and store the filings
2. Query the system with `rag_query()` (single company) or `rag_compare()` (cross-company)

## Limitations & future work

- The system retrieves balanced context across companies but will note when a question can't be fully answered from the source text — a deliberate grounding behavior over hallucinating an unsupported answer.
- Currently limited to three companies and one filing type (10-K); could extend to 10-Qs, earnings calls, or a larger company universe.
- No persistent chat interface yet — currently run cell-by-cell in a notebook. A CLI or lightweight web UI would be a natural next step.

## Dataset

SEC 10-K filings (fiscal year 2025) for Apple, Microsoft, and Tesla, sourced from [SEC EDGAR](https://www.sec.gov/edgar/search/).
