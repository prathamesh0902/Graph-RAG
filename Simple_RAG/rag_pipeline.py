"""Podcast RAG pipeline: TXT/SRT parsing, chunking, Chroma, BM25, RRF and reranking."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ─── Env-Driven Defaults ──────────────────────────────────────────────────────
_DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
_DEFAULT_OVERLAP = int(os.getenv("RAG_OVERLAP", "80"))
_DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "20"))
_DEFAULT_TOP_N = int(os.getenv("RAG_TOP_N", "5"))

# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SRTBlock:
    index: int
    start_time: str
    end_time: str
    text: str


@dataclass
class TextChunk:
    chunk_id: int
    text: str
    start_time: str
    end_time: str
    char_start: int
    char_end: int
    source: str
    token_count: int = 0
    source_key: str = ""


# ─── Phase 1: SRT Parser ──────────────────────────────────────────────────────

def _read_text(file_path: str) -> str:
    path = Path(file_path)
    raw = path.read_bytes()
    # Try common transcript encodings; replace undecodable bytes as a last resort.
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_srt(file_path: str) -> List[SRTBlock]:
    """Parse SRT blocks. Returns [] for plain text; use parse_transcript for either format."""
    content = _read_text(file_path).replace("\r\n", "\n").replace("\r", "\n")
    blocks: List[SRTBlock] = []
    # Regex is more tolerant than splitting on blank lines (some SRTs omit them).
    pattern = re.compile(
        r"(?m)^\s*(\d+)\s*\n\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
        r"(\d{2}:\d{2}:\d{2}[,.]\d{3}).*?\n(.*?)(?=\n\s*\n|\n\s*\d+\s*\n\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->|\Z)",
        re.S,
    )
    for match in pattern.finditer(content):
        text = " ".join(line.strip() for line in match.group(4).splitlines() if line.strip())
        if text:
            blocks.append(SRTBlock(
                index=int(match.group(1)),
                start_time=match.group(2).replace(",", "."),
                end_time=match.group(3).replace(",", "."),
                text=text,
            ))
    # Fallback parser for well-formed blocks separated by blank lines.
    if not blocks:
        for raw_block in re.split(r"\n\s*\n", content.strip()):
            lines = [line.strip() for line in raw_block.splitlines() if line.strip()]
            if len(lines) < 2:
                continue
            time_line_idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
            if time_line_idx is None:
                continue
            tm = re.search(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})", lines[time_line_idx])
            if not tm:
                continue
            idx = int(lines[0]) if lines[0].isdigit() else len(blocks) + 1
            text = " ".join(lines[time_line_idx + 1:])
            if text:
                blocks.append(SRTBlock(idx, tm.group(1).replace(",", "."), tm.group(2).replace(",", "."), text))
    return blocks

# ─── Phase 2: Cleaning & Reconstruction ───────────────────────────────────────

def clean_srt_blocks(blocks: List[SRTBlock]) -> List[SRTBlock]:
    noise_patterns = [r"\(Transcribed by.*?\)", r"\[music\]", r"♪.*?♪", r"<[^>]+>"]
    cleaned: List[SRTBlock] = []
    for block in blocks:
        text = block.text
        for pattern in noise_patterns:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= 2:
            cleaned.append(SRTBlock(block.index, block.start_time, block.end_time, text))
    return cleaned


def parse_transcript(file_path: str) -> List[SRTBlock]:
    """Load either SRT or plain TXT into timestamped blocks."""
    path = Path(file_path)
    content = _read_text(file_path).strip()
    if not content:
        return []
    if path.suffix.lower() == ".srt" or re.search(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->", content):
        parsed = clean_srt_blocks(parse_srt(file_path))
        if parsed:
            return parsed
    # Plain text has no trustworthy timestamps. Split into paragraph blocks.
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    if not paragraphs:
        paragraphs = [re.sub(r"\s+", " ", content)]
    return [SRTBlock(i + 1, "N/A", "N/A", p) for i, p in enumerate(paragraphs) if p]


def reconstruct_full_text(blocks: List[SRTBlock]) -> List[dict]:
    return [
        {"text": b.text, "start_time": b.start_time, "end_time": b.end_time}
        for b in blocks if b.text.strip()
    ]


def _token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
    except Exception:
        return lambda text: len(text.split())

# ─── Phase 3: Chunking ────────────────────────────────────────────────────────

def chunk_segments(segments: list, chunk_size: int = 500, overlap: int = 80,
                   source_name: str = "podcast", source_key: str = "") -> List[TextChunk]:
    """Create approximately token-sized sliding windows and map each to source timestamps."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size >= 1 and 0 <= overlap < chunk_size.")
    if not segments:
        return []
    count_tokens = _token_counter()
    # Keep character offsets and segment boundaries aligned to the joined transcript.
    full_text_parts: List[str] = []
    spans: List[Tuple[int, int, dict]] = []
    cursor = 0
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if full_text_parts:
            cursor += 1  # separator space
        start = cursor
        full_text_parts.append(text)
        cursor += len(text)
        spans.append((start, cursor, seg))
    full_text = " ".join(full_text_parts)
    words = list(re.finditer(r"\S+", full_text))
    if not words:
        return []

    chunks: List[TextChunk] = []
    start_word = 0
    chunk_id = 0
    while start_word < len(words):
        end_word = start_word
        chosen_text = ""
        while end_word < len(words):
            candidate = full_text[words[start_word].start():words[end_word].end()]
            if end_word > start_word and count_tokens(candidate) > chunk_size:
                break
            chosen_text = candidate
            end_word += 1
            if count_tokens(chosen_text) >= chunk_size:
                break
        if end_word == start_word:
            end_word += 1
            chosen_text = words[start_word].group(0)
        char_start = words[start_word].start()
        char_end = words[end_word - 1].end()
        overlapping_segments = [s for s, e, _ in spans if e > char_start and s < char_end]
        if overlapping_segments:
            first_idx = next(i for i, (s, e, _) in enumerate(spans) if e > char_start and s < char_end)
            last_idx = max(i for i, (s, e, _) in enumerate(spans) if e > char_start and s < char_end)
            start_time = spans[first_idx][2].get("start_time", "N/A")
            end_time = spans[last_idx][2].get("end_time", "N/A")
        else:
            start_time = end_time = "N/A"
        chunks.append(TextChunk(
            chunk_id=chunk_id, text=chosen_text, start_time=start_time,
            end_time=end_time, char_start=char_start, char_end=char_end,
            source=source_name, token_count=count_tokens(chosen_text), source_key=source_key,
        ))
        chunk_id += 1
        if end_word >= len(words):
            break
        # Walk backwards from the previous end until approximately overlap tokens remain.
        next_start = end_word
        overlap_start = end_word - 1
        while overlap_start > start_word:
            candidate = full_text[words[overlap_start].start():words[end_word - 1].end()]
            if count_tokens(candidate) > overlap:
                break
            overlap_start -= 1
        next_start = max(start_word + 1, overlap_start + 1)
        start_word = next_start
    return chunks

# ─── Phase 4: Vector Store (ChromaDB) ─────────────────────────────────────────

class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "podcast_rag"):
        import chromadb
        self.persist_dir = str(Path(persist_dir).expanduser().resolve())
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        print(f"ChromaDB persistent directory: {self.persist_dir}")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection_name = collection_name
        self._embedder = None
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
            print(f"Loading embedding model: {model_name}")
            self._embedder = SentenceTransformer(model_name)
        return self._embedder

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        model = self._get_embedder()
        prepared = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        vectors = model.encode(prepared, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, query: str) -> List[float]:
        model = self._get_embedder()
        vector = model.encode(
            f"Represent this question for searching relevant passages: {query}",
            normalize_embeddings=True,
        )
        return np.asarray(vector, dtype=np.float32).tolist()

    def index_chunks(self, chunks: List[TextChunk], source_key: str) -> int:
        if not chunks:
            return 0
        # IDs are stable for the same source content, not merely the same filename.
        ids = [f"{source_key}_{c.chunk_id}" for c in chunks]
        existing = self.collection.get(ids=ids, include=[])
        existing_ids = set(existing.get("ids", []))
        missing = [(doc_id, chunk) for doc_id, chunk in zip(ids, chunks) if doc_id not in existing_ids]
        if not missing:
            print(f"All {len(ids)} chunks already exist in ChromaDB.")
            return len(ids)
        new_ids = [item[0] for item in missing]
        new_chunks = [item[1] for item in missing]
        texts = [c.text for c in new_chunks]
        embeddings = self.embed_texts(texts)
        metadatas = [{
            "source": c.source, "source_key": source_key,
            "chunk_id": int(c.chunk_id), "start_time": c.start_time,
            "end_time": c.end_time, "token_count": int(c.token_count),
        } for c in new_chunks]
        for i in range(0, len(new_ids), 100):
            self.collection.upsert(
                ids=new_ids[i:i + 100], embeddings=embeddings[i:i + 100],
                documents=texts[i:i + 100], metadatas=metadatas[i:i + 100],
            )
        print(f"Indexed {len(new_ids)} new chunks.")
        return len(ids)

    def dense_search(self, query: str, top_k: int = 20) -> List[Tuple[str, dict, float]]:
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_embeddings=[self.embed_query(query)], n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            hits.append((doc, meta or {}, float(1.0 - distance)))
        return hits

# ─── Phase 5: BM25 Sparse Retrieval ───────────────────────────────────────────

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.chunks: List[TextChunk] = []

    @staticmethod
    def _tok(text: str) -> List[str]:
        return re.sub(r"[^\w\s]", " ", text.lower()).split()

    def build(self, chunks: List[TextChunk]):
        self.chunks = list(chunks)
        if not self.chunks:
            self.bm25 = None
            return
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([self._tok(c.text) for c in self.chunks])
        print(f"BM25 index built over {len(self.chunks)} chunks.")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, dict, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(self._tok(query))
        idxs = np.argsort(scores)[::-1][:top_k]
        hits = []
        for i in idxs:
            if scores[i] <= 0:
                continue
            c = self.chunks[int(i)]
            hits.append((c.text, {
                "source": c.source, "source_key": c.source_key,
                "chunk_id": c.chunk_id, "start_time": c.start_time,
                "end_time": c.end_time, "token_count": c.token_count,
            }, float(scores[i])))
        return hits

# ─── Phase 6: Hybrid Retrieval with RRF ───────────────────────────────────────

def reciprocal_rank_fusion(dense_hits, sparse_hits, k: int = 60,
                           dense_weight: float = 0.6, sparse_weight: float = 0.4):
    scores: Dict[str, float] = {}
    docs: Dict[str, tuple] = {}
    for weight, hits in ((dense_weight, dense_hits), (sparse_weight, sparse_hits)):
        for rank, (text, meta, _) in enumerate(hits, start=1):
            key = f"{meta.get('source_key', meta.get('source', ''))}:{meta.get('chunk_id', rank)}"
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            docs[key] = (text, meta)
    return [(docs[key][0], docs[key][1], scores[key]) for key in sorted(scores, key=scores.get, reverse=True)]

# ─── Phase 7: Cross-Encoder Reranking ─────────────────────────────────────────

class Reranker:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._model = None
        self.load_error = None

    def rerank(self, query: str, candidates, top_n: int = 5):
        if not candidates:
            return []
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                print(f"Loading reranker: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
            scores = self._model.predict([(query, text) for text, _, _ in candidates])
            ranked = sorted(zip(scores, candidates), key=lambda item: float(item[0]), reverse=True)
            return [(text, meta, float(score)) for score, (text, meta, _) in ranked[:top_n]]
        except Exception as exc:
            self.load_error = str(exc)
            print(f"Reranker unavailable; using fused retrieval order: {exc}")
            return candidates[:top_n]

# ─── Full Pipeline Orchestrator (Multi-Transcript) ────────────────────────────

class PodcastRAG:
    def __init__(self, persist_dir: str = "./rag_store", chunk_size: int = _DEFAULT_CHUNK_SIZE,
                 overlap: int = _DEFAULT_OVERLAP, top_k_retrieve: int = _DEFAULT_TOP_K,
                 top_n_rerank: int = _DEFAULT_TOP_N):
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Invalid chunk configuration: require chunk_size > overlap >= 0.")
        self.persist_dir = Path(persist_dir).expanduser().resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k_retrieve = top_k_retrieve
        self.top_n_rerank = top_n_rerank
        self.vector_store = VectorStore(str(self.persist_dir / "chroma"))
        self.bm25_index = BM25Index()
        self.reranker = Reranker()
        self._all_chunks: List[TextChunk] = []
        self._ingested_sources: Dict[str, dict] = {}
        print(f"RAG ready. persist_dir={self.persist_dir}; chunk_size={chunk_size}; overlap={overlap}; top_k={top_k_retrieve}; top_n={top_n_rerank}")

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]

    def ingest(self, file_path: str, source_name: Optional[str] = None) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript does not exist: {path}")
        raw = path.read_bytes()
        if not raw:
            raise ValueError("Transcript file is empty.")
        source_name = (source_name or path.stem).strip()
        if not source_name:
            source_name = path.stem
        content_hash = hashlib.sha256(raw).hexdigest()
        source_key = self._hash(f"{source_name}:{content_hash}")
        if source_key in self._ingested_sources:
            return {**self._ingested_sources[source_key], "already_loaded": True}

        cache_path = self.persist_dir / f"{source_key}_chunks.json"
        if cache_path.exists():
            source_chunks = self._load_source_chunks(cache_path)
            cached = True
        else:
            blocks = parse_transcript(str(path))
            stats_base = {"raw_blocks": len(blocks)}
            segments = reconstruct_full_text(blocks)
            source_chunks = chunk_segments(segments, self.chunk_size, self.overlap, source_name, source_key)
            if not source_chunks:
                raise ValueError("No text chunks were generated. Check the transcript content/format.")
            self._save_source_chunks(source_chunks, cache_path)
            cached = False
        # Always ensure Chroma has these chunks, even when loading a JSON cache.
        indexed_count = self.vector_store.index_chunks(source_chunks, source_key)
        self._all_chunks.extend(source_chunks)
        stats = {
            "source": source_name, "source_key": source_key,
            "chunks": len(source_chunks), "indexed_chunks": indexed_count,
            "cached": cached, "total_chunks": len(self._all_chunks),
            "total_sources": len(self._ingested_sources) + 1,
        }
        self._ingested_sources[source_key] = stats
        self.bm25_index.build(self._all_chunks)
        self._save_state()
        print(f"Ingested {source_name}: {len(source_chunks)} chunks (cached={cached}).")
        return stats

    def ingest_multiple(self, file_paths: List[str]) -> List[dict]:
        return [self.ingest(path) for path in file_paths]

    def retrieve(self, query: str) -> List[Tuple[str, dict, float]]:
        if not query or not query.strip():
            return []
        dense_hits = self.vector_store.dense_search(query, self.top_k_retrieve)
        sparse_hits = self.bm25_index.search(query, self.top_k_retrieve)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
        if not fused:
            return []
        return self.reranker.rerank(query, fused[:max(20, self.top_k_retrieve)], self.top_n_rerank)

    def format_context(self, hits) -> str:
        if not hits:
            return "No relevant excerpts found."
        parts = []
        for i, (text, meta, score) in enumerate(hits, start=1):
            parts.append(
                f"[Excerpt {i} | Source: {meta.get('source', 'unknown')} | "
                f"Timestamp: {meta.get('start_time', 'N/A')} -> {meta.get('end_time', 'N/A')} | "
                f"Retrieval score: {score:.4f}]\n{text}"
            )
        return "\n\n---\n\n".join(parts)

    def _save_state(self):
        state = {"sources": list(self._ingested_sources.keys())}
        tmp = self.persist_dir / "rag_state.tmp"
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.persist_dir / "rag_state.json")

    def load_state(self) -> bool:
        """Restore source chunks from local cache and rebuild BM25 for this process."""
        state_path = self.persist_dir / "rag_state.json"
        if not state_path.exists():
            return False
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            restored = []
            for source_key in state.get("sources", []):
                cache_path = self.persist_dir / f"{source_key}_chunks.json"
                if not cache_path.exists():
                    continue
                chunks = self._load_source_chunks(cache_path)
                if not chunks:
                    continue
                self._all_chunks.extend(chunks)
                source_name = chunks[0].source
                self._ingested_sources[source_key] = {
                    "source": source_name, "source_key": source_key,
                    "chunks": len(chunks), "cached": True,
                    "total_chunks": len(self._all_chunks),
                }
                # Re-index cached chunks if Chroma collection was cleared/missing records.
                self.vector_store.index_chunks(chunks, source_key)
                restored.append(source_name)
            if self._all_chunks:
                self.bm25_index.build(self._all_chunks)
            print(f"Restored {len(restored)} transcript(s): {restored}")
            return bool(restored)
        except Exception as exc:
            print(f"Could not restore RAG state: {exc}")
            return False

    def _save_source_chunks(self, chunks: List[TextChunk], path: Path):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _load_source_chunks(self, path: Path) -> List[TextChunk]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [TextChunk(**item) for item in data]
