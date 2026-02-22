import math
import re
from dataclasses import dataclass, field


TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class MemoryEvent:
    text: str
    meta: dict = field(default_factory=dict)


class SimpleMemory:
    """Simple in-memory TF-IDF retrieval (lightweight and dependency-free but not optimized for large corpora, etc.)."""
    def __init__(self) -> None:
        self.events: list[MemoryEvent] = []
        self._docs_tokens: list[list[str]] = []

    def add_event(self, text: str, meta: dict | None = None) -> None:
        self.events.append(MemoryEvent(text=text, meta=meta or {}))
        self._docs_tokens.append(tokenize(text))

    def retrieve(self, query: str, k: int = 3) -> list[MemoryEvent]:
        if not self.events:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        doc_scores = []
        idf = self._compute_idf()
        query_vec = self._tfidf(query_tokens, idf)
        for idx, tokens in enumerate(self._docs_tokens):
            doc_vec = self._tfidf(tokens, idf)
            score = self._cosine(query_vec, doc_vec)
            doc_scores.append((score, idx))
        doc_scores.sort(reverse=True)
        top = [self.events[idx] for score, idx in doc_scores[:k] if score > 0]
        return top

    def _compute_idf(self) -> dict:
        doc_count = len(self._docs_tokens)
        df: dict[str, int] = {}
        for tokens in self._docs_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log((doc_count + 1) / (df_val + 1)) + 1.0 for t, df_val in df.items()}
        return idf

    def _tfidf(self, tokens: list[str], idf: dict) -> dict:
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = {t: (tf_val / len(tokens)) * idf.get(t, 0.0) for t, tf_val in tf.items()}
        return vec

    def _cosine(self, v1: dict, v2: dict) -> float:
        if not v1 or not v2:
            return 0.0
        dot = 0.0
        for t, w in v1.items():
            dot += w * v2.get(t, 0.0)
        norm1 = math.sqrt(sum(w * w for w in v1.values()))
        norm2 = math.sqrt(sum(w * w for w in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
