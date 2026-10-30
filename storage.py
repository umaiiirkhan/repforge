"""Optional SQLite storage for workout logs and saved training preferences.

Only non-sensitive data is stored, keyed by an anonymous random user id.
Users can delete everything about themselves at any time.
"""
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS workout_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    log_date     TEXT NOT NULL,
    workout      TEXT NOT NULL,
    duration_min INTEGER NOT NULL,
    notes        TEXT DEFAULT '',
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_user ON workout_logs(user_id);
CREATE TABLE IF NOT EXISTS preferences (
    user_id    TEXT PRIMARY KEY,
    experience TEXT,
    equipment  TEXT,
    days       INTEGER,
    updated_at TEXT NOT NULL
);
"""


class Storage:
    def __init__(self, db_path=config.DB_PATH):
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------------------------------------------------- workout logs
    def add_log(self, user_id, workout, duration_min, log_date=None, notes=""):
        log_date = log_date or date.today().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO workout_logs (user_id, log_date, workout, duration_min, notes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, log_date, workout, int(duration_min), notes, datetime.now().isoformat(timespec="seconds")))
            return cur.lastrowid

    def get_logs(self, user_id, limit=50):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, log_date, workout, duration_min, notes FROM workout_logs "
                "WHERE user_id = ? ORDER BY log_date DESC, id DESC LIMIT ?", (user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def delete_log(self, user_id, log_id):
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM workout_logs WHERE id = ? AND user_id = ?", (log_id, user_id))
            return cur.rowcount > 0

    def summary(self, user_id):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS sessions, COALESCE(SUM(duration_min), 0) AS minutes, MAX(log_date) AS last "
                "FROM workout_logs WHERE user_id = ?", (user_id,)).fetchone()
            week = conn.execute(
                "SELECT COUNT(*) FROM workout_logs WHERE user_id = ? AND log_date >= date('now', '-6 days')",
                (user_id,)).fetchone()[0]
        return {"sessions": row["sessions"], "minutes": row["minutes"], "last_date": row["last"], "this_week": week}

    # ----------------------------------------------------------- preferences
    def save_preferences(self, user_id, experience, equipment, days):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO preferences (user_id, experience, equipment, days, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET experience=excluded.experience, equipment=excluded.equipment, "
                "days=excluded.days, updated_at=excluded.updated_at",
                (user_id, experience, equipment, int(days), datetime.now().isoformat(timespec="seconds")))

    def get_preferences(self, user_id):
        with self._connect() as conn:
            row = conn.execute("SELECT experience, equipment, days FROM preferences WHERE user_id = ?",
                               (user_id,)).fetchone()
        return dict(row) if row else None

    # --------------------------------------------------------------- privacy
    def delete_user_data(self, user_id):
        with self._connect() as conn:
            conn.execute("DELETE FROM workout_logs WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM preferences WHERE user_id = ?", (user_id,))
