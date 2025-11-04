from typing import List, Dict, Any
import numpy as np
from .embeddings import Embedder
from .faiss_index import FaissIndex
from .cache_store import CacheStore


class Retriever:
    def __init__(
        self, embedder: Embedder, index: FaissIndex, store: CacheStore, topk: int = 3):
        self.embedder = embedder
        self.index = index
        self.store = store
        self.topk = topk
        self._next_vec_id = self._bootstrap_vec_id()

    def _bootstrap_vec_id(self) -> int:
        # 简化处理：用 turn_id 作为 vec_id，也可以单独维护计数器
        latest = self.store.latest(1)
        return latest[0]["turn_id"] + 1 if latest else 1

    def add_history(self, speaker: str, text: str, ts: int = 0, tags: str = ""):
        turn_id = self.store.append_turn(speaker, text, ts, tags)
        vec = self.embedder.encode([text])
        vec_id = np.array([turn_id], dtype=np.int64)
        self.index.add(vec, vec_id)
        self.store.bind_vec(turn_id, turn_id)
        return turn_id

    def search(self, query: str) -> List[Dict[str, Any]]:
        qv = self.embedder.encode([query])
        D, I = self.index.search(qv, self.topk)
        results = []
        for vec_id in I[0]:
            if int(vec_id) < 0:
                continue
            item = self.store.get_turn(int(vec_id))
            if item:
                results.append(item)
        return results
