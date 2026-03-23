"""
RAG Pipeline — Clean Implementation

A reusable, end-to-end Retrieval-Augmented Generation pipeline.

Usage:
    from src.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    rag.index_file("data/knowledge_base.txt")
    answer = rag.ask("What is the difference between RAG and fine-tuning?")
    print(answer)
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv


@dataclass
class RetrievedChunk:
    title: str
    content: str
    score: float


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk]
    query: str


class RAGPipeline:
    """
    A complete RAG pipeline:
      1. Index documents (chunk → embed → store in ChromaDB)
      2. Answer questions (embed query → retrieve → augment → generate)
    """

    def __init__(
        self,
        collection_name: str = "rag_collection",
        chroma_path: str = "data/chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
        top_k: int = 3,
    ):
        load_dotenv()

        self.llm_model = llm_model
        self.top_k = top_k

        # Embedding function (local, no API key needed)
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # ChromaDB (persistent)
        self._chroma = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )

        # OpenAI client (for generation)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")
        self._openai = OpenAI(api_key=api_key)

    # ------------------------------------------------------------------ #
    # Indexing                                                             #
    # ------------------------------------------------------------------ #

    def index_file(self, file_path: str) -> int:
        """
        Load a knowledge base file, chunk it, and index it.
        Supports files with DOCUMENT: section headers.
        Returns the number of chunks indexed.
        """
        text = Path(file_path).read_text()
        chunks = self._chunk_by_document(text)

        if not chunks:
            # Fall back to paragraph chunking for generic text files
            chunks = self._chunk_by_paragraph(text)

        self._collection.add(
            documents=[c["content"] for c in chunks],
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"title": c.get("title", f"chunk_{i}")} for i, c in enumerate(chunks)],
        )

        return len(chunks)

    def index_texts(self, texts: list[str], titles: list[str] | None = None) -> int:
        """
        Index a list of plain text strings directly.
        Returns the number of chunks indexed.
        """
        if titles is None:
            titles = [f"Document {i}" for i in range(len(texts))]

        start_id = self._collection.count()
        self._collection.add(
            documents=texts,
            ids=[f"chunk_{start_id + i}" for i in range(len(texts))],
            metadatas=[{"title": t} for t in titles],
        )
        return len(texts)

    def clear_index(self) -> None:
        """Delete all documents from the collection."""
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, n_results: int | None = None) -> list[RetrievedChunk]:
        """
        Find the most relevant chunks for a query.
        """
        k = n_results or self.top_k
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                RetrievedChunk(
                    title=meta.get("title", ""),
                    content=doc,
                    score=round(1 / (1 + dist), 3),
                )
            )
        return chunks

    # ------------------------------------------------------------------ #
    # Generation                                                           #
    # ------------------------------------------------------------------ #

    def ask(self, question: str) -> RAGResponse:
        """
        Full RAG pipeline: retrieve relevant context, then generate an answer.
        """
        # 1. Retrieve
        context_chunks = self.retrieve(question)

        # 2. Build augmented prompt
        context_text = "\n\n".join(
            f"[Source: {chunk.title}]\n{chunk.content}"
            for chunk in context_chunks
        )

        system_prompt = (
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the provided context. If the context does not contain enough information "
            "to fully answer the question, say so. Do not fabricate information."
        )

        user_prompt = f"Context:\n---\n{context_text}\n---\n\nQuestion: {question}"

        # 3. Generate
        response = self._openai.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=600,
        )

        return RAGResponse(
            answer=response.choices[0].message.content,
            sources=context_chunks,
            query=question,
        )

    def ask_without_rag(self, question: str) -> str:
        """Ask the LLM directly without any retrieved context (for comparison)."""
        response = self._openai.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=600,
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------ #
    # Chunking helpers                                                     #
    # ------------------------------------------------------------------ #

    def _chunk_by_document(self, text: str) -> list[dict]:
        """Split text at DOCUMENT: section headers."""
        sections = text.strip().split("\n\n")
        chunks = []
        for section in sections:
            section = section.strip()
            if section.startswith("DOCUMENT:"):
                lines = section.split("\n", 1)
                title = lines[0].replace("DOCUMENT:", "").strip()
                content = lines[1].strip() if len(lines) > 1 else ""
                if content:
                    chunks.append({"title": title, "content": content})
        return chunks

    def _chunk_by_paragraph(self, text: str, max_chars: int = 600) -> list[dict]:
        """Split text by double newlines (paragraphs), merging short ones."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < max_chars:
                current = (current + " " + para).strip()
            else:
                if current:
                    chunks.append({"title": current[:50], "content": current})
                current = para
        if current:
            chunks.append({"title": current[:50], "content": current})
        return chunks

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    @property
    def document_count(self) -> int:
        return self._collection.count()

    def __repr__(self) -> str:
        return (
            f"RAGPipeline(model={self.llm_model!r}, "
            f"top_k={self.top_k}, "
            f"indexed={self.document_count} chunks)"
        )


# ------------------------------------------------------------------ #
# Example usage (run this file directly to test)                     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("Initializing RAG pipeline...")
    rag = RAGPipeline(chroma_path="data/chroma_src_demo")

    print("Indexing knowledge base...")
    n = rag.index_file("data/knowledge_base.txt")
    print(f"Indexed {n} chunks. Pipeline: {rag}\n")

    questions = [
        "What is RAG and why does it reduce hallucinations?",
        "What are the tradeoffs between fine-tuning and RAG?",
        "How do vector databases enable fast similarity search?",
    ]

    for question in questions:
        print(f"Q: {question}")
        result = rag.ask(question)
        print(f"A: {result.answer}")
        print(f"Sources: {[s.title for s in result.sources]}")
        print("-" * 60)
