"""
Bước 4: Vector Database.

Bản local dùng FAISS IndexFlatIP (Inner Product).
Vector InsightFace đã L2-normalize -> inner product == cosine similarity.

Khi muốn scale lên hàng triệu records, chỉ cần thay class này bằng adapter
gọi Milvus / Pinecone / pgvector — interface (`add`, `search`, `remove`,
`save`, `load`) giữ nguyên.
"""
from __future__ import annotations
import json
import os
import threading
from pathlib import Path
from typing import Optional
import numpy as np
import faiss
from loguru import logger

from config import settings


class FaissVectorStore:
    """
    IndexFlatIP + ID mapping ngoài (vì IndexFlatIP không nhớ id).
    Ý tưởng: faiss row i  <->  self._ids[i]  (chuỗi 'employee_id').
    Khi xoá, ta rebuild index — ổn cho <100k records; trên đó dùng IndexIDMap2.
    """

    def __init__(self,
                 dim: int = settings.embedding_dim,
                 index_path: str = settings.faiss_index_path,
                 meta_path: str = settings.faiss_meta_path):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self._lock = threading.Lock()
        self._ids: list[str] = []
        self._index: faiss.Index = faiss.IndexFlatIP(dim)
        self._load_if_exists()

    # ---------------------------------------------------------------- IO

    def _load_if_exists(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self._ids = json.load(f)
            logger.info(f"FAISS loaded: {len(self._ids)} vectors")

    def save(self) -> None:
        with self._lock:
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self._ids, f, ensure_ascii=False)

    # ---------------------------------------------------------------- API

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, employee_id: str, embedding: np.ndarray) -> None:
        if embedding.shape != (self.dim,):
            raise ValueError(f"embedding shape != ({self.dim},)")
        with self._lock:
            self._index.add(embedding.reshape(1, -1).astype(np.float32))
            self._ids.append(employee_id)

    def add_many(self, employee_id: str, embeddings: np.ndarray) -> None:
        """Thêm nhiều ảnh cho cùng 1 nhân viên (mỗi vector 1 row)."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"embeddings shape != (N, {self.dim})")
        with self._lock:
            self._index.add(embeddings.astype(np.float32))
            self._ids.extend([employee_id] * embeddings.shape[0])

    def search(self, embedding: np.ndarray, top_k: int = 1) -> list[tuple[str, float]]:
        """Trả về [(employee_id, similarity), ...] sắp xếp giảm dần."""
        if len(self._ids) == 0:
            return []
        sims, idxs = self._index.search(
            embedding.reshape(1, -1).astype(np.float32), top_k
        )
        out: list[tuple[str, float]] = []
        for sim, i in zip(sims[0], idxs[0]):
            if i == -1:
                continue
            out.append((self._ids[i], float(sim)))
        return out

    def best_match(self, embedding: np.ndarray) -> Optional[tuple[str, float]]:
        res = self.search(embedding, top_k=1)
        return res[0] if res else None

    def remove(self, employee_id: str) -> int:
        """Rebuild index, bỏ tất cả vectors thuộc employee_id. Trả về số bị xoá."""
        with self._lock:
            keep_mask = [eid != employee_id for eid in self._ids]
            removed = self._ids.count(employee_id)
            if removed == 0:
                return 0

            # Lấy lại tất cả vectors hiện có (chỉ khả thi với IndexFlatIP)
            vectors = faiss.rev_swig_ptr(
                self._index.get_xb(), self._index.ntotal * self.dim
            ).reshape(-1, self.dim).copy()

            kept_vectors = vectors[keep_mask]
            kept_ids = [eid for eid, k in zip(self._ids, keep_mask) if k]

            self._index = faiss.IndexFlatIP(self.dim)
            if len(kept_ids) > 0:
                self._index.add(kept_vectors)
            self._ids = kept_ids
        self.save()
        return removed

    def list_employee_ids(self) -> list[str]:
        return list(set(self._ids))
