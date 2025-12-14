# app/storage.py
import sqlite3
import json
import os
from typing import Optional
from datetime import datetime

class Storage:
    """
    Minimal SQLite storage. Synchronous but safe for use with FastAPI when called
    via run_in_executor or asyncio.to_thread.
    """
    def __init__(self, dbpath: str = "data/st22.db"):
        os.makedirs(os.path.dirname(dbpath) or ".", exist_ok=True)
        self.dbpath = dbpath
        self._conn = sqlite3.connect(self.dbpath, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                dump_id TEXT PRIMARY KEY,
                dump_json TEXT,
                llm_json TEXT,
                ai_summary TEXT,
                created_at TEXT
            )
            """
        )
        self._conn.commit()

    def save_analysis(self, dump_id: str, dump_json: dict, llm_resp: dict) -> dict:
        cur = self._conn.cursor()
        now = datetime.utcnow().isoformat() + "Z"
        dump_json_s = json.dumps(dump_json, default=str)
        llm_json_s = json.dumps(llm_resp, default=str)
        ai_summary = None
        parsed = llm_resp.get("json") or {}
        if isinstance(parsed, dict):
            ai_summary = parsed
        elif isinstance(parsed, str):
            ai_summary = parsed
        else:
            # fallback small summary
            ai_summary = {"raw": llm_resp.get("text","")[:200]}

        cur.execute(
            """
            INSERT OR REPLACE INTO analyses (dump_id, dump_json, llm_json, ai_summary, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (dump_id, dump_json_s, llm_json_s, json.dumps(ai_summary, default=str), now),
        )
        self._conn.commit()
        return {
            "dump_id": dump_id,
            "dump": dump_json,
            "ai_summary": ai_summary,
            "priority": (ai_summary.get("priority") if isinstance(ai_summary, dict) else None),
            "raw_llm": llm_resp,
        }

    def get_analysis(self, dump_id: str) -> Optional[dict]:
        cur = self._conn.cursor()
        cur.execute("SELECT dump_json, llm_json, ai_summary, created_at FROM analyses WHERE dump_id = ?", (dump_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "dump_id": dump_id,
            "dump": json.loads(row["dump_json"]),
            "ai_summary": json.loads(row["ai_summary"]) if row["ai_summary"] else None,
            "priority": (json.loads(row["ai_summary"]).get("priority") if row["ai_summary"] else None),
            "raw_llm": json.loads(row["llm_json"]) if row["llm_json"] else None,
            "created_at": row["created_at"],
        }

    def list_recent(self, limit: int = 50) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute("SELECT dump_id, ai_summary, created_at FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        out = []
        for r in rows:
            ai_summary = None
            try:
                ai_summary = json.loads(r["ai_summary"]) if r["ai_summary"] else None
            except Exception:
                ai_summary = r["ai_summary"]
            out.append({"dump_id": r["dump_id"], "ai_summary": ai_summary, "created_at": r["created_at"]})
        return out
