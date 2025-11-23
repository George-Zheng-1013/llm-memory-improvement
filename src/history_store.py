import sqlite3
import os
import time
from datetime import datetime
from typing import Optional, List, Tuple


class HistoryStore:
    """对话历史存储管理类"""

    def __init__(self, db_path: str):
        """
        初始化历史存储

        Args:
            db_path: SQLite数据库文件路径
        """
        self.db_path = db_path

        # 确保目录存在
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        # 初始化数据库
        self._init_database()

    def _init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建对话历史表
        cursor.execute(
            """
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
        """
        )

        # 创建会话信息表
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            start_time REAL NOT NULL,
            start_time_str TEXT NOT NULL,
            last_update REAL NOT NULL,
            total_turns INTEGER DEFAULT 0
        )
        """
        )

        # 创建索引以提高查询效率
        cursor.execute(
            """
        CREATE INDEX IF NOT EXISTS idx_session_turn 
        ON conversation_history(session_id, turn_number)
        """
        )

        cursor.execute(
            """
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON conversation_history(timestamp)
        """
        )

        conn.commit()
        conn.close()

    def start_session(self) -> str:
        """
        开始新的对话会话

        Returns:
            session_id: 会话ID
        """
        session_id = f"session_{int(time.time() * 1000)}"
        timestamp = time.time()
        timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT INTO sessions (session_id, start_time, start_time_str, last_update, total_turns)
        VALUES (?, ?, ?, ?, 0)
        """,
            (session_id, timestamp, timestamp_str, timestamp),
        )

        conn.commit()
        conn.close()

        return session_id

    def save_turn(
        self,
        session_id: str,
        turn_number: int,
        user_input: str,
        assistant_response: str,
    ):
        """
        保存一轮对话

        Args:
            session_id: 会话ID
            turn_number: 对话轮数
            user_input: 用户输入
            assistant_response: 助手回复
        """
        timestamp = time.time()
        timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 保存用户输入
            cursor.execute(
                """
            INSERT OR REPLACE INTO conversation_history 
            (session_id, turn_number, role, content, timestamp, created_at)
            VALUES (?, ?, 'user', ?, ?, ?)
            """,
                (session_id, turn_number, user_input, timestamp, timestamp_str),
            )

            # 保存助手回复
            cursor.execute(
                """
            INSERT OR REPLACE INTO conversation_history 
            (session_id, turn_number, role, content, timestamp, created_at)
            VALUES (?, ?, 'assistant', ?, ?, ?)
            """,
                (session_id, turn_number, assistant_response, timestamp, timestamp_str),
            )

            # 更新会话信息
            cursor.execute(
                """
            UPDATE sessions 
            SET last_update = ?, total_turns = ?
            WHERE session_id = ?
            """,
                (timestamp, turn_number, session_id),
            )

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def get_session_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Tuple]:
        """
        获取指定会话的历史记录

        Args:
            session_id: 会话ID
            limit: 限制返回的对话轮数

        Returns:
            历史记录列表 [(turn_number, role, content, timestamp, created_at), ...]
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if limit:
            query = """
            SELECT turn_number, role, content, timestamp, created_at
            FROM conversation_history
            WHERE session_id = ?
            ORDER BY turn_number DESC, role DESC
            LIMIT ?
            """
            cursor.execute(
                query, (session_id, limit * 2)
            )  # *2 因为每轮有用户和助手两条
        else:
            query = """
            SELECT turn_number, role, content, timestamp, created_at
            FROM conversation_history
            WHERE session_id = ?
            ORDER BY turn_number ASC, role DESC
            """
            cursor.execute(query, (session_id,))

        results = cursor.fetchall()
        conn.close()

        return results

    def get_all_sessions(self) -> List[Tuple]:
        """
        获取所有会话列表

        Returns:
            会话列表 [(session_id, start_time_str, total_turns), ...]
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT session_id, start_time_str, total_turns
        FROM sessions
        ORDER BY start_time DESC
        """
        )

        results = cursor.fetchall()
        conn.close()

        return results

    def get_recent_history(self, session_id: str, n_turns: int) -> str:
        """
        获取最近N轮对话的格式化字符串，用于构建上下文

        Args:
            session_id: 会话ID
            n_turns: 获取最近的对话轮数

        Returns:
            格式化的历史对话字符串
        """
        history = self.get_session_history(session_id, limit=n_turns)

        if not history:
            return ""

        # 按轮次和角色重新排序（用户在前，助手在后）
        history_sorted = sorted(
            history, key=lambda x: (x[0], 0 if x[1] == "user" else 1)
        )

        formatted_history = []
        for turn_num, role, content, _, created_at in history_sorted:
            role_name = "用户" if role == "user" else "助手"
            formatted_history.append(f"[{created_at}] {role_name}: {content}")

        return "\n".join(formatted_history)
