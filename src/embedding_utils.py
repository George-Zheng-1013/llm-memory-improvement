# src/embedding_utils.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

# 使用一个高效且常用的模型，其维度为 384
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 

class EmbeddingClient:
    """封装 Sentence-Transformer 模型的客户端"""
    def __init__(self):
        try:
            print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
            # 将设备设置为 'cpu' 以确保在没有 GPU 的机器上也能运行
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu') 
            self.vector_dim = self.model.get_sentence_embedding_dimension()
            print(f"Embedding Model Loaded. Dimension: {self.vector_dim}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.vector_dim = 384 # 默认维度
            self.model = None

    def get_embedding(self, text: str) -> np.ndarray:
        """对单个文本生成嵌入向量"""
        if self.model is None:
            return np.zeros(self.vector_dim, dtype=np.float32) 
        
        # 返回形状为 (D,) 的 NumPy 数组
        return self.model.encode(text, convert_to_numpy=True)

# 全局初始化 EmbeddingClient，以避免重复加载模型
EMBEDDING_CLIENT = EmbeddingClient()