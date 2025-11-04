import sqlite3
from typing import Dict, Any, List, Optional, Tuple
import os


SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
turn_id INTEGER PRIMARY KEY,
speaker TEXT NOT NULL,
text TEXT NOT NULL,
ts INTEGER,
tags TEXT DEFAULT ''
);


CREATE TABLE IF NOT EXISTS vec_map (
turn_id INTEGER PRIMARY KEY,
vec_id INTEGER UNIQUE
);
"""


class CacheStore:
    def __init__(self, sqlite_path: str):
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        self.conn = sqlite3.connect(sqlite_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def append_turn(self, speaker: str, text: str, ts: int = 0, tags: str = ""):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO turns(speaker, text, ts, tags) VALUES(?,?,?,?)",
            (speaker, text, ts, tags),
        )
        self.conn.commit()
        return cur.lastrowid

    def bind_vec(self, turn_id: int, vec_id: int):
        self.conn.execute(
            "INSERT OR REPLACE INTO vec_map(turn_id, vec_id) VALUES(?,?)",
            (turn_id, vec_id),
        )
        self.conn.commit()

    def get_turn(self, turn_id: int) -> Dict[str, Any]:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT turn_id, speaker, text, ts, tags FROM turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
        if not row:
            return {}
        return {
            "turn_id": row[0],
            "speaker": row[1],
            "text": row[2],
            "ts": row[3],
            "tags": row[4],
        }

    def latest(self, n: int = 10) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT turn_id, speaker, text, ts, tags FROM turns ORDER BY turn_id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [
            {"turn_id": r[0], "speaker": r[1], "text": r[2], "ts": r[3], "tags": r[4]}
            for r in rows
        ]
