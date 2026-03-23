# Learning RAG (Retrieval-Augmented Generation)

A hands-on project for learning RAG fundamentals, built for portfolio and interview demonstrations.

---

## What is RAG?

RAG is a technique that improves LLM responses by **retrieving relevant context** from an external knowledge base before generating an answer. Instead of relying solely on the model's training data, RAG grounds answers in your own documents.

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────────────┐
│  Embed      │────▶│  Vector Store        │
│  the query  │     │  (search for similar │
└─────────────┘     │   document chunks)   │
                    └──────────┬───────────┘
                               │  Top-K relevant chunks
                               ▼
                    ┌──────────────────────┐
                    │  Augment the prompt  │
                    │  with retrieved docs │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       LLM            │
                    │  (generates answer   │
                    │   grounded in docs)  │
                    └──────────────────────┘
```

---

## Project Structure

```
Learning_RAG/
├── notebooks/
│   ├── 01_embeddings_and_similarity.ipynb   # What are embeddings? Semantic search
│   ├── 02_vector_store_with_chromadb.ipynb  # Storing & querying embeddings
│   ├── 03_document_chunking.ipynb           # Splitting docs for retrieval
│   └── 04_full_rag_pipeline.ipynb           # End-to-end RAG pipeline
├── src/
│   └── rag_pipeline.py                      # Clean, reusable RAG implementation
├── data/
│   └── knowledge_base.txt                   # Sample knowledge base documents
├── requirements.txt
└── .env.example
```

---

## Core Concepts Covered

| Concept | Notebook | Description |
|---|---|---|
| Embeddings | 01 | Converting text to vectors that capture semantic meaning |
| Cosine Similarity | 01 | Measuring how similar two vectors are |
| Vector Store | 02 | Database optimized for similarity search |
| Document Chunking | 03 | Splitting documents into retrievable pieces |
| Full RAG Pipeline | 04 | Putting it all together with an LLM |

---

## Tech Stack

- **Embeddings**: [`sentence-transformers`](https://www.sbert.net/) — free, local, no API key needed
- **Vector Store**: [`ChromaDB`](https://www.trychroma.com/) — lightweight, persistent, easy to use
- **LLM**: [`OpenAI`](https://platform.openai.com/) (GPT-4o-mini) — for the generation step
- **Notebooks**: Jupyter

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/jringler30/Learning_RAG.git
cd Learning_RAG

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key (only needed for notebook 04)
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Launch Jupyter
jupyter notebook notebooks/
```

---

## Learning Path

Work through the notebooks in order:

1. **Notebook 01** — Understand what embeddings are and how semantic similarity works. No API key needed.
2. **Notebook 02** — Learn to store and query embeddings using ChromaDB. No API key needed.
3. **Notebook 03** — Learn how to chunk documents effectively for retrieval. No API key needed.
4. **Notebook 04** — Build the full RAG pipeline end-to-end with an LLM.

Each notebook is self-contained with explanations before every code block.

---

## Key Takeaways

- RAG separates **knowledge** (your documents) from **reasoning** (the LLM)
- Embedding quality and chunking strategy have a huge impact on retrieval quality
- RAG dramatically reduces hallucinations on domain-specific questions
- The pipeline has four stages: **Load → Embed → Retrieve → Generate**
