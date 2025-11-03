from typing import List, Tuple
import faiss
import numpy as np
import os


class FaissIndex:
    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            # 小规模先用 FlatL2，无需训练，易增量
            base_index = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIDMap(base_index)

    def add(self, vectors: np.ndarray, ids: np.ndarray):
        assert vectors.dtype == np.float32
        assert ids.dtype == np.int64
        self.index.add_with_ids(vectors, ids)

    def search(
        self, query_vecs: np.ndarray, topk: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        D, I = self.index.search(query_vecs.astype("float32"), topk)
        return D, I

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
