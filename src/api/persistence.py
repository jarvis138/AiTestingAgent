import os
import json
import sqlite3
from typing import Any, Dict, List

DB_PATH = os.getenv('AGENT_DB_PATH', 'data/agent.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute('PRAGMA journal_mode=WAL;')
    return con


def init_db() -> None:
    con = _conn()
    try:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              action TEXT,
              tests TEXT,
              returncode INTEGER,
              stdout TEXT,
              stderr TEXT
            );
                        CREATE TABLE IF NOT EXISTS artifacts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            run_id INTEGER,
                            key TEXT,
                            path TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        );
            CREATE TABLE IF NOT EXISTS generated_tests (
              id TEXT PRIMARY KEY,
              path TEXT NOT NULL,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        con.commit()
    finally:
        con.close()


def log_run(action: str, tests: List[str], returncode: int, stdout: str, stderr: str) -> None:
    con = _conn()
    try:
        cur = con.execute(
            'INSERT INTO runs(action, tests, returncode, stdout, stderr) VALUES(?,?,?,?,?)',
            (action, json.dumps(tests or []), int(returncode), stdout or '', stderr or '')
        )
        run_id = cur.lastrowid
        con.commit()
    finally:
        con.close()


def add_generated_test(test_id: str, path: str) -> None:
    con = _conn()
    try:
        con.execute(
            'INSERT OR REPLACE INTO generated_tests(id, path) VALUES(?, ?)',
            (test_id, path)
        )
        con.commit()
    finally:
        con.close()


def recent_runs(limit: int = 20) -> List[Dict[str, Any]]:
    con = _conn()
    try:
        cur = con.execute('SELECT id, created_at, action, tests, returncode FROM runs ORDER BY id DESC LIMIT ?', (limit,))
        rows = cur.fetchall()
        out = []
        for rid, created_at, action, tests, rc in rows:
            acur = con.execute('SELECT COUNT(1) FROM artifacts WHERE run_id = ?', (rid,))
            acount = acur.fetchone()[0]
            out.append({
                'id': rid,
                'created_at': created_at,
                'action': action,
                'tests': json.loads(tests or '[]'),
                'returncode': rc,
                'artifacts': acount
            })
        return out
    finally:
        con.close()


def add_artifact(run_id: int, key: str, path: str) -> None:
    con = _conn()
    try:
        con.execute('INSERT INTO artifacts(run_id, key, path) VALUES(?,?,?)', (run_id, key, path))
        con.commit()
    finally:
        con.close()
