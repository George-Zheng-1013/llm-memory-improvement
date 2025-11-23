import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import io
import time
from typing import Optional

# 初始化模型
bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
bi_encoder.max_seq_length = 256

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def np_to_blob(arr: np.ndarray) -> bytes:
    """将numpy数组转换为二进制blob"""
    bio = io.BytesIO()
    np.save(bio, arr)
    return bio.getvalue()


def blob_to_np(blob: bytes) -> np.ndarray:
    """将二进制blob转换回numpy数组"""
    bio = io.BytesIO(blob)
    return np.load(bio)


def encode_and_save_turn(db_path: str, session_id: str, turn_number: int):
    """
    对指定会话的某一轮对话进行编码并保存embedding

    Args:
        db_path: SQLite数据库文件路径
        session_id: 会话ID
        turn_number: 对话轮数
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 初始化embedding表（如果不存在）
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                convo_row_id INTEGER NOT NULL UNIQUE,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY(convo_row_id) REFERENCES conversation_history(id)
            )
            """
        )
        conn.commit()

        # 获取该轮对话的用户输入和助手回复
        cursor.execute(
            """
            SELECT id, content FROM conversation_history
            WHERE session_id = ? AND turn_number = ? AND role = 'user'
            """,
            (session_id, turn_number),
        )
        user_row = cursor.fetchone()

        cursor.execute(
            """
            SELECT id, content FROM conversation_history
            WHERE session_id = ? AND turn_number = ? AND role = 'assistant'
            """,
            (session_id, turn_number),
        )
        assistant_row = cursor.fetchone()

        if not user_row or not assistant_row:
            print(f"警告: 未找到会话 {session_id} 第 {turn_number} 轮的对话")
            return

        # 合并用户输入和助手回复进行编码
        combined_text = f"{user_row[1]} {assistant_row[1]}"
        embedding = bi_encoder.encode(combined_text, convert_to_tensor=True)
        emb_blob = np_to_blob(
            embedding.cpu().numpy() if hasattr(embedding, "cpu") else embedding
        )

        # 分别保存用户和助手的embedding
        ts = time.time()

        cursor.execute(
            """
            INSERT OR REPLACE INTO conversation_embeddings 
            (convo_row_id, embedding, created_at)
            VALUES (?, ?, ?)
            """,
            (user_row[0], sqlite3.Binary(emb_blob), ts),
        )

        cursor.execute(
            """
            INSERT OR REPLACE INTO conversation_embeddings 
            (convo_row_id, embedding, created_at)
            VALUES (?, ?, ?)
            """,
            (assistant_row[0], sqlite3.Binary(emb_blob), ts),
        )

        conn.commit()
        print(f"✓ 第 {turn_number} 轮对话的embedding已保存")

    except Exception as e:
        conn.rollback()
        print(f"保存embedding失败: {str(e)}")
        raise e
    finally:
        conn.close()
