"""
agents/rag_engine.py — Medical Knowledge RAG Engine
=====================================================
WHAT IT DOES:
  Provides Retrieval-Augmented Generation (RAG) for the medical advisor.
  Embeds medical knowledge articles into ChromaDB and retrieves relevant
  context based on patient symptoms and predicted conditions.

WHY RAG:
  - Grounds LLM responses in verified medical literature
  - Reduces hallucination by providing factual reference material
  - Makes the system's advice citable and verifiable

USAGE:
  from agents.rag_engine import medical_rag

  context = medical_rag.query(
      symptoms=["fever", "headache"],
      conditions=["malaria"],
      k=5
  )
"""

import json
from pathlib import Path

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("[RAG] ChromaDB not installed. Run: pip install chromadb sentence-transformers")


BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_PATH = BASE_DIR / "knowledge" / "medical_knowledge.json"
CHROMA_DIR = BASE_DIR / "knowledge" / "chroma_db"
COLLECTION_NAME = "medical_knowledge"


class MedicalRAG:
    """
    Medical knowledge retrieval engine using ChromaDB.

    Singleton pattern — index is built once and persisted.
    On subsequent loads, the existing index is reused.
    """

    def __init__(self):
        self._collection = None
        self._ready = False

        if not CHROMADB_AVAILABLE:
            print("[RAG] ChromaDB not available — RAG disabled")
            return

        try:
            self._init_chromadb()
        except Exception as e:
            print(f"[RAG] Initialization error: {e}")
            self._ready = False

    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        # Create persistent client
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        # Use sentence-transformers for embedding
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"description": "Medical knowledge base for RAG"}
        )

        # Build index if empty
        if self._collection.count() == 0:
            self._build_index()

        print(f"[RAG] ChromaDB ready: {self._collection.count()} documents")
        self._ready = True

    def _build_index(self):
        """Load medical knowledge JSON and embed into ChromaDB."""
        if not KNOWLEDGE_PATH.exists():
            print(f"[RAG] Knowledge file not found: {KNOWLEDGE_PATH}")
            return

        print(f"[RAG] Building index from {KNOWLEDGE_PATH}...")

        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            entries = json.load(f)

        documents = []
        metadatas = []
        ids = []

        for i, entry in enumerate(entries):
            # Combine topic and content for embedding
            doc_text = f"{entry['topic']}: {entry['content']}"
            documents.append(doc_text)
            metadatas.append({
                "topic": entry["topic"],
                "category": entry.get("category", "general"),
                "source": entry.get("source", "Medical Literature")
            })
            ids.append(f"med_{i:04d}")

        # Add to ChromaDB in batches
        batch_size = 50
        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            self._collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end]
            )

        print(f"[RAG] Indexed {len(documents)} medical knowledge entries")

    @property
    def is_ready(self) -> bool:
        return self._ready

    def query(self, symptoms: list[str] = None, conditions: list[str] = None,
              user_query: str = None, k: int = 5) -> str:
        """
        Retrieve relevant medical knowledge based on symptoms and conditions.

        Args:
            symptoms: list of patient symptoms
            conditions: list of predicted conditions
            user_query: raw user input (optional additional context)
            k: number of results to return

        Returns:
            Formatted string of relevant medical knowledge for injection into prompts
        """
        if not self._ready:
            return ""

        # Build search query from symptoms + conditions
        query_parts = []
        if symptoms:
            query_parts.append(f"Symptoms: {', '.join(symptoms)}")
        if conditions:
            condition_names = []
            for c in conditions:
                if isinstance(c, dict):
                    condition_names.append(c.get("name", ""))
                else:
                    condition_names.append(str(c))
            query_parts.append(f"Conditions: {', '.join(condition_names)}")
        if user_query:
            query_parts.append(user_query)

        if not query_parts:
            return ""

        search_text = ". ".join(query_parts)

        try:
            results = self._collection.query(
                query_texts=[search_text],
                n_results=min(k, self._collection.count()),
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"[RAG] Query error: {e}")
            return ""

        if not results or not results["documents"] or not results["documents"][0]:
            return ""

        # Format results
        context_parts = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # Only include results with reasonable similarity
            # ChromaDB uses L2 distance — lower is more similar
            if dist > 1.8:  # threshold for relevance
                continue

            source = meta.get("source", "Medical Literature")
            context_parts.append(f"[{source}]\n{doc}")

        if not context_parts:
            return ""

        formatted = "\n\n".join(context_parts[:k])
        return f"""
━━━ MEDICAL KNOWLEDGE (from verified sources) ━━━
{formatted}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    def rebuild_index(self):
        """Force rebuild the index from knowledge file."""
        if not self._ready:
            return

        # Delete existing collection and recreate
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"description": "Medical knowledge base for RAG"}
        )
        self._build_index()
        print(f"[RAG] Index rebuilt: {self._collection.count()} documents")


# ── Singleton instance ────────────────────────────────────────────────────────
# Initialized once when first imported. Subsequent imports reuse the same object.
try:
    medical_rag = MedicalRAG()
except Exception as e:
    print(f"[RAG] Failed to initialize: {e}")

    # Create a dummy object so imports don't fail
    class _DummyRAG:
        is_ready = False
        def query(self, **kwargs): return ""

    medical_rag = _DummyRAG()
