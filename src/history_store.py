# src/history_store.py

import sqlite3
import os
import time
from datetime import datetime
from typing import Optional, List, Tuple
import faiss
import numpy as np
import pickle 
from .embedding_utils import EMBEDDING_CLIENT 

class HistoryStore:
    """对话历史存储管理类 (集成 FAISS 语义索引)"""

    def __init__(self, db_path: str, faiss_index_dir: str = './cache/'):
        self.db_path = db_path
        self.faiss_index_dir = faiss_index_dir
        self.faiss_path = os.path.join(faiss_index_dir, 'history.faiss')
        self.map_path = os.path.join(faiss_index_dir, 'history_map.pkl')

        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        if not os.path.exists(faiss_index_dir):
            os.makedirs(faiss_index_dir)

        self._init_database()
        self.vector_dim = EMBEDDING_CLIENT.vector_dim
        self.index, self.faiss_map = self._load_or_init_faiss_index()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_number INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp REAL NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(session_id, turn_number, role)
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            start_time REAL NOT NULL,
            start_time_str TEXT NOT NULL,
            last_update REAL NOT NULL,
            total_turns INTEGER DEFAULT 0
        )
        """)
        conn.commit()
        conn.close()

    def _load_or_init_faiss_index(self):
        if self.vector_dim == 384 and EMBEDDING_CLIENT.model is None:
             print("FAISS indexing disabled: Embedding model not loaded.")
             return None, {}

        if os.path.exists(self.faiss_path) and os.path.exists(self.map_path):
            try:
                index = faiss.read_index(self.faiss_path)
                with open(self.map_path, 'rb') as f:
                    faiss_map = pickle.load(f)
                print(f"Loaded FAISS index with {index.ntotal} vectors.")
                return index, faiss_map
            except Exception as e:
                print(f"Error loading index: {e}. Creating new one.")

        print("Initializing new FAISS index...")
        index = faiss.IndexFlatIP(self.vector_dim) 
        faiss_map = {} 
        return index, faiss_map

    def _save_faiss_index(self):
        if self.index is None: return
        faiss.write_index(self.index, self.faiss_path)
        with open(self.map_path, 'wb') as f:
            pickle.dump(self.faiss_map, f)

    def _index_text(self, text: str, turn_id: str):
        """辅助函数：将文本向量化并添加到索引"""
        vector = EMBEDDING_CLIENT.get_embedding(text)
        vector = vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(vector) 
        self.index.add(vector)
        new_faiss_id = self.index.ntotal - 1
        self.faiss_map[new_faiss_id] = turn_id

    def rebuild_faiss_index(self):
        """
        【修复点】重建 FAISS 索引 (遍历 SQLite 中所有历史记录)
        """
        if self.index is None: return
        
        # 1. 检查数据库是否有数据
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM conversation_history")
        count = cursor.fetchone()[0]
        
        if count == 0:
            conn.close()
            return

        # 如果索引为空但数据库有数据，或者强制重建
        if self.index.ntotal < count:
            print(f"Rebuilding index for {count} turns (current index: {self.index.ntotal})...")
            self.index.reset()
            self.faiss_map = {}
            
            cursor.execute("SELECT session_id, turn_number, role, content FROM conversation_history")
            rows = cursor.fetchall()
            
            for session_id, turn_number, role, content in rows:
                text = f"[{role}]: {content}"
                turn_id = f"{session_id}_{turn_number}"
                self._index_text(text, turn_id)
            
            self._save_faiss_index()
            print("Index rebuild complete.")
        
        conn.close()

    def save_turn(self, session_id: str, turn_number: int, user_content: str, assistant_content: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        current_time = time.time()
        created_at = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
        
        # 保存到 SQLite
        cursor.execute(
            "INSERT OR REPLACE INTO conversation_history (session_id, turn_number, role, content, timestamp, created_at) VALUES (?, ?, 'user', ?, ?, ?)",
            (session_id, turn_number, user_content, current_time, created_at)
        )
        cursor.execute(
            "INSERT OR REPLACE INTO conversation_history (session_id, turn_number, role, content, timestamp, created_at) VALUES (?, ?, 'assistant', ?, ?, ?)",
            (session_id, turn_number, assistant_content, current_time + 0.001, created_at)
        )
        # 更新 Session
        cursor.execute(
            "UPDATE sessions SET last_update = ?, total_turns = ? WHERE session_id = ?",
            (current_time, turn_number, session_id),
        )
        conn.commit()
        conn.close()

        # 添加到 FAISS
        if self.index is not None:
            self._index_text(f"[user]: {user_content}", f"{session_id}_{turn_number}")
            self._index_text(f"[assistant]: {assistant_content}", f"{session_id}_{turn_number}")
            self._save_faiss_index()

    def update_session_total_turns(self, session_id: str, total_turns: int):
        """【修复点】更新会话总轮数"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET total_turns = ?, last_update = ? WHERE session_id = ?",
            (total_turns, time.time(), session_id)
        )
        conn.commit()
        conn.close()

    def search_history_index(self, query: str, top_k: int = 5, similarity_threshold: float = 0.0) -> List[Tuple[str, float]]:
        if self.index is None or self.index.ntotal == 0: return []

        query_vector = EMBEDDING_CLIENT.get_embedding(query).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vector) 

        D, I = self.index.search(query_vector, top_k * 5) # 搜索更多以备过滤
        
        results = []
        seen_turns = set()
        
        for distance, faiss_id in zip(D[0], I[0]):
            if faiss_id == -1: continue
            if distance < similarity_threshold: continue
            
            turn_id = self.faiss_map.get(faiss_id)
            if turn_id and turn_id not in seen_turns:
                results.append((turn_id, float(distance)))
                seen_turns.add(turn_id)
            
            if len(results) >= top_k: break
                
        return results

    def start_session(self) -> str:
        session_id = str(int(time.time() * 1000000))
        current_time = time.time()
        start_time_str = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (session_id, start_time, start_time_str, last_update, total_turns) VALUES (?, ?, ?, ?, ?)",
            (session_id, current_time, start_time_str, current_time, 0)
        )
        conn.commit()
        conn.close()
        return session_id

    def get_session_history(self, session_id: str, limit: Optional[int] = None) -> List[Tuple]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        limit_query = f"LIMIT {limit}" if limit is not None else ""
        query = f"""
            SELECT turn_number, role, content, timestamp, created_at
            FROM conversation_history
            WHERE session_id = ?
            ORDER BY turn_number ASC, role DESC
            {limit_query}
            """
        cursor.execute(query, (session_id,))
        results = cursor.fetchall()
        conn.close()
        return results