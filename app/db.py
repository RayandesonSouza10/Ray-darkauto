from __future__ import annotations

import sqlite3
from typing import Any

from config import DB_PATH


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                niche TEXT NOT NULL,
                content_strategy TEXT,
                visual_style TEXT,
                script_prompt_base TEXT,
                image_prompt_base TEXT,
                thumbnail_prompt TEXT,
                default_persona TEXT,
                extra_settings TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                visual_style TEXT,
                base_prompt TEXT,
                reference_image_path TEXT,
                tags TEXT,
                language TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                language TEXT NOT NULL,
                country TEXT NOT NULL,
                niche TEXT NOT NULL,
                subniche TEXT,
                platform TEXT NOT NULL,
                creation_date TEXT NOT NULL,
                status TEXT NOT NULL,
                default_tts_model_name TEXT,
                default_voice_name TEXT,
                default_voice_language_code TEXT,
                default_tts_audio_format TEXT,
                default_tts_speaking_rate REAL,
                default_tts_pitch REAL,
                default_tts_volume_gain_db REAL,
                default_tts_sample_rate_hz INTEGER,
                template_id INTEGER,
                default_character_id INTEGER,
                strategic_description TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(template_id) REFERENCES templates(id) ON DELETE SET NULL,
                FOREIGN KEY(default_character_id) REFERENCES characters(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                language TEXT,
                full_text TEXT NOT NULL,
                channel_id INTEGER,
                template_id INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(channel_id) REFERENCES channels(id) ON DELETE SET NULL,
                FOREIGN KEY(template_id) REFERENCES templates(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                video_title TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                status TEXT NOT NULL,
                script_path TEXT,
                scenes_path TEXT,
                image_prompts_path TEXT,
                voice_path TEXT,
                run_summary TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(channel_id) REFERENCES channels(id) ON DELETE CASCADE
            );
            """
        )
        existing_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(channels)").fetchall()
        }
        if "default_voice_name" not in existing_columns:
            conn.execute("ALTER TABLE channels ADD COLUMN default_voice_name TEXT")
        if "default_voice_language_code" not in existing_columns:
            conn.execute("ALTER TABLE channels ADD COLUMN default_voice_language_code TEXT")
        if "default_tts_model_name" not in existing_columns:
            conn.execute("ALTER TABLE channels ADD COLUMN default_tts_model_name TEXT")
        if "default_tts_audio_format" not in existing_columns:
            conn.execute("ALTER TABLE channels ADD COLUMN default_tts_audio_format TEXT")
        if "default_tts_speaking_rate" not in existing_columns:
            conn.execute("ALTER TABLE channels ADD COLUMN default_tts_speaking_rate REAL")
        if "default_tts_pitch" not in existing_columns:
            conn.execute("ALTER TABLE channels ADD COLUMN default_tts_pitch REAL")
        if "default_tts_volume_gain_db" not in existing_columns:
            conn.execute("ALTER TABLE channels ADD COLUMN default_tts_volume_gain_db REAL")
        if "default_tts_sample_rate_hz" not in existing_columns:
            conn.execute("ALTER TABLE channels ADD COLUMN default_tts_sample_rate_hz INTEGER")
        conn.commit()


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
        return dict(row) if row else None


def execute(query: str, params: tuple[Any, ...] = ()) -> int:
    with get_connection() as conn:
        cur = conn.execute(query, params)
        conn.commit()
        return int(cur.lastrowid or 0)
