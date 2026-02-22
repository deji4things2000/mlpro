from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import litellm


@dataclass
class CodeChunk:
    id: str
    filepath: str  # workspace-relative
    kind: str  # "function" | "class" | "module"
    name: str
    start_line: int
    end_line: int
    code: str

    def for_embedding(self) -> str:
        header = f"{self.kind} {self.name} ({self.filepath}:{self.start_line}-{self.end_line})"
        return f"{header}\n\n{self.code}"


def _workspace_root() -> Path:
    return Path(os.getcwd()).resolve()


def _iter_files(pattern: str) -> Iterable[Path]:
    root = _workspace_root()
    yield from root.glob(pattern)


def _relpath(p: Path) -> str:
    return str(p.resolve().relative_to(_workspace_root()))


def _safe_read(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _extract_python_chunks(code: str, filepath_rel: str) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If file can't parse, treat as module chunk.
        module_id = _sha1(f"module:{filepath_rel}")
        return [
            CodeChunk(
                id=module_id,
                filepath=filepath_rel,
                kind="module",
                name=Path(filepath_rel).name,
                start_line=1,
                end_line=max(1, len(code.splitlines())),
                code=code,
            )
        ]

    lines = code.splitlines()

    def get_segment(node: ast.AST) -> tuple[int, int, str] | None:
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return None
        start = int(getattr(node, "lineno"))
        end = int(getattr(node, "end_lineno"))
        seg = "\n".join(lines[start - 1 : end])
        return start, end, seg

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = get_segment(node)
            if not seg:
                continue
            start, end, seg_text = seg
            cid = _sha1(f"fn:{filepath_rel}:{node.name}:{start}:{end}")
            chunks.append(
                CodeChunk(
                    id=cid,
                    filepath=filepath_rel,
                    kind="function",
                    name=node.name,
                    start_line=start,
                    end_line=end,
                    code=seg_text,
                )
            )
        elif isinstance(node, ast.ClassDef):
            seg = get_segment(node)
            if not seg:
                continue
            start, end, seg_text = seg
            cid = _sha1(f"class:{filepath_rel}:{node.name}:{start}:{end}")
            chunks.append(
                CodeChunk(
                    id=cid,
                    filepath=filepath_rel,
                    kind="class",
                    name=node.name,
                    start_line=start,
                    end_line=end,
                    code=seg_text,
                )
            )

    if not chunks:
        module_id = _sha1(f"module:{filepath_rel}")
        chunks.append(
            CodeChunk(
                id=module_id,
                filepath=filepath_rel,
                kind="module",
                name=Path(filepath_rel).name,
                start_line=1,
                end_line=max(1, len(lines)),
                code=code,
            )
        )

    return chunks


class CodeRAG:
    """AST-based chunking + embedding search with a simple on-disk cache.

    Uses `litellm.embedding` so we don't need extra dependencies.
    """

    def __init__(
        self,
        embedder_model: str,
        index_pattern: str = "**/*.py",
        cache_dir: str = ".simplecoder",
    ):
        self.embedder_model = embedder_model
        self.index_pattern = index_pattern
        self.cache_dir = (_workspace_root() / cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._chunks: list[CodeChunk] = []
        self._embeddings: np.ndarray | None = None

    def _cache_key(self) -> str:
        root = str(_workspace_root())
        return _sha1(f"{root}|{self.embedder_model}|{self.index_pattern}")

    def _paths(self) -> tuple[Path, Path]:
        key = self._cache_key()
        meta = self.cache_dir / f"rag_{key}.json"
        vecs = self.cache_dir / f"rag_{key}.npy"
        return meta, vecs

    def _load_cache(self) -> bool:
        meta_path, vecs_path = self._paths()
        if not meta_path.exists() or not vecs_path.exists():
            return False

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            file_metas: dict[str, float] = meta.get("files", {})
            # Validate mtimes
            for fp, mtime in file_metas.items():
                p = _workspace_root() / fp
                if not p.exists() or abs(p.stat().st_mtime - float(mtime)) > 1e-6:
                    return False

            self._chunks = [CodeChunk(**c) for c in meta["chunks"]]
            self._embeddings = np.load(vecs_path)
            if self._embeddings.ndim != 2 or self._embeddings.shape[0] != len(self._chunks):
                return False
            return True
        except Exception:
            return False

    def _save_cache(self, file_metas: dict[str, float]) -> None:
        meta_path, vecs_path = self._paths()
        meta = {
            "embedder_model": self.embedder_model,
            "index_pattern": self.index_pattern,
            "files": file_metas,
            "chunks": [c.__dict__ for c in self._chunks],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        assert self._embeddings is not None
        np.save(vecs_path, self._embeddings)

    def ensure_index(self, progress_cb: Any | None = None) -> None:
        """Build or load the index."""
        if self._embeddings is not None and self._chunks:
            return

        if self._load_cache():
            if progress_cb:
                progress_cb(f"RAG: loaded cached index with {len(self._chunks)} chunks")
            return

        root = _workspace_root()
        files = [p for p in _iter_files(self.index_pattern) if p.is_file()]
        file_metas = {str(p.resolve().relative_to(root)): p.stat().st_mtime for p in files}

        chunks: list[CodeChunk] = []
        for i, p in enumerate(files):
            if progress_cb and i % 25 == 0:
                progress_cb(f"RAG: parsing {i+1}/{len(files)} files")
            text = _safe_read(p)
            if text is None:
                continue
            rel = _relpath(p)
            if p.suffix == ".py":
                chunks.extend(_extract_python_chunks(text, rel))
            else:
                cid = _sha1(f"file:{rel}")
                chunks.append(
                    CodeChunk(
                        id=cid,
                        filepath=rel,
                        kind="module",
                        name=p.name,
                        start_line=1,
                        end_line=max(1, len(text.splitlines())),
                        code=text,
                    )
                )

        if not chunks:
            self._chunks = []
            self._embeddings = np.zeros((0, 1), dtype=np.float32)
            return

        # Embed in batches
        texts = [c.for_embedding() for c in chunks]
        vectors: list[list[float]] = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            end = min(len(texts), start + batch_size)
            if progress_cb:
                progress_cb(f"RAG: embedding chunks {start+1}-{end}/{len(texts)}")
            resp = litellm.embedding(model=self.embedder_model, input=texts[start:end])
            # litellm returns {"data": [{"embedding": [...]}, ...]}
            data = resp.get("data", [])
            for item in data:
                vectors.append(item["embedding"])

        emb = np.array(vectors, dtype=np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms

        self._chunks = chunks
        self._embeddings = emb
        self._save_cache(file_metas)
        if progress_cb:
            progress_cb(f"RAG: indexed {len(files)} files into {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self.ensure_index()
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        resp = litellm.embedding(model=self.embedder_model, input=[query])
        q = np.array(resp["data"][0]["embedding"], dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)

        sims = self._embeddings @ q
        idxs = np.argsort(sims)[-top_k:][::-1]
        results: list[dict[str, Any]] = []
        for idx in idxs:
            c = self._chunks[int(idx)]
            results.append(
                {
                    "score": float(sims[int(idx)]),
                    "filepath": c.filepath,
                    "kind": c.kind,
                    "name": c.name,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "code": c.code,
                }
            )
        return results