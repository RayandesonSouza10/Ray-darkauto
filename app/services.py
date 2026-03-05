from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
import wave
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    FFMPEG_BIN,
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_CLOUD_PROJECT,
    OUTPUTS_DIR,
    TTS_AUDIO_FORMAT,
    TTS_MODEL_NAME,
    TTS_LANGUAGE_CODE,
    TTS_PITCH,
    TTS_SAMPLE_RATE_HZ,
    TTS_SPEAKING_RATE,
    TTS_VOLUME_GAIN_DB,
    TTS_VOICE_NAME,
    VERTEX_IMAGE_MODEL,
    VERTEX_API_VERSION,
    VERTEX_MODEL,
    VERTEX_SCRIPT_MODEL,
    VERTEX_VIDEO_MODEL,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from db import execute, fetch_all, fetch_one


GEMINI_STUDIO_VOICE_PROFILES: list[dict[str, str]] = [
    {"name": "Zephyr", "style": "Bright"},
    {"name": "Kore", "style": "Firm"},
    {"name": "Orus", "style": "Firm"},
    {"name": "Autonoe", "style": "Bright"},
    {"name": "Umbriel", "style": "Easy-going"},
    {"name": "Erinome", "style": "Clear"},
    {"name": "Laomedeia", "style": "Upbeat"},
    {"name": "Schedar", "style": "Even"},
    {"name": "Achird", "style": "Friendly"},
    {"name": "Sadachbia", "style": "Lively"},
    {"name": "Puck", "style": "Upbeat"},
    {"name": "Fenrir", "style": "Excitable"},
    {"name": "Aoede", "style": "Breezy"},
    {"name": "Enceladus", "style": "Breathy"},
    {"name": "Algieba", "style": "Smooth"},
    {"name": "Algenib", "style": "Gravelly"},
    {"name": "Achernar", "style": "Soft"},
    {"name": "Gacrux", "style": "Mature"},
    {"name": "Zubenelgenubi", "style": "Casual"},
    {"name": "Sadaltager", "style": "Knowledgeable"},
    {"name": "Charon", "style": "Informative"},
    {"name": "Leda", "style": "Youthful"},
    {"name": "Callirrhoe", "style": "Easy-going"},
    {"name": "Iapetus", "style": "Clear"},
    {"name": "Despina", "style": "Smooth"},
    {"name": "Pulcherrima", "style": "Forward"},
    {"name": "Vindemiatrix", "style": "Gentle"},
    {"name": "Sulafat", "style": "Warm"},
    {"name": "Rasalgethi", "style": "Informative"},
    {"name": "Alnilam", "style": "Firm"},
]

PREFERRED_VERTEX_TTS_MODELS: list[dict[str, str]] = [
    {"name": "chirp-3.0-generate-001", "label": "Chirp 3: HD Voices"},
    {"name": "gemini-2.5-flash-tts", "label": "Gemini 2.5 Flash TTS"},
    {"name": "gemini-2.5-pro-tts", "label": "Gemini 2.5 Pro TTS"},
    {"name": "gemini-2.5-flash-lite-preview-tts", "label": "Gemini 2.5 Flash Lite Preview TTS"},
]

PREFERRED_TTS_LANGUAGE_OPTIONS: list[dict[str, str]] = [
    {"code": "pt-BR", "label": "Português (Brasil)"},
    {"code": "en-US", "label": "English (United States)"},
    {"code": "en-GB", "label": "English (United Kingdom)"},
    {"code": "es-US", "label": "Español (Estados Unidos)"},
    {"code": "fr-FR", "label": "Français (France)"},
    {"code": "de-DE", "label": "Deutsch (Deutschland)"},
    {"code": "it-IT", "label": "Italiano (Italia)"},
    {"code": "ja-JP", "label": "日本語 (日本)"},
    {"code": "ko-KR", "label": "한국어 (대한민국)"},
    {"code": "hi-IN", "label": "हिन्दी (भारत)"},
    {"code": "id-ID", "label": "Bahasa Indonesia"},
    {"code": "nl-NL", "label": "Nederlands (Nederland)"},
    {"code": "pl-PL", "label": "Polski (Polska)"},
    {"code": "ru-RU", "label": "Русский (Россия)"},
    {"code": "th-TH", "label": "ไทย (ไทย)"},
    {"code": "tr-TR", "label": "Türkçe (Türkiye)"},
    {"code": "vi-VN", "label": "Tiếng Việt (Việt Nam)"},
]

VISUAL_PROMPT_MODE_IMAGE_ONLY = "somente_imagens"
VISUAL_PROMPT_MODE_IMAGE_AND_ANIMATION = "imagens_e_animacao"
VISUAL_PROMPT_MODES = {
    VISUAL_PROMPT_MODE_IMAGE_ONLY,
    VISUAL_PROMPT_MODE_IMAGE_AND_ANIMATION,
}

SUBTITLE_MODE_DISABLED = "desativada"
SUBTITLE_MODE_SRT_ONLY = "arquivo_srt"
SUBTITLE_MODE_BURN_IN = "embutida_no_video"
SUBTITLE_MODES = {
    SUBTITLE_MODE_DISABLED,
    SUBTITLE_MODE_SRT_ONLY,
    SUBTITLE_MODE_BURN_IN,
}
SUBTITLE_DENSITY_BALANCED = "equilibrada"
SUBTITLE_DENSITY_COMPACT = "compacta"
SUBTITLE_DENSITY_DETAILED = "detalhada"
SUBTITLE_DENSITIES = {
    SUBTITLE_DENSITY_BALANCED,
    SUBTITLE_DENSITY_COMPACT,
    SUBTITLE_DENSITY_DETAILED,
}
SUBTITLE_POSITION_BOTTOM = "inferior"
SUBTITLE_POSITION_CENTER = "centro"
SUBTITLE_POSITION_TOP = "superior"
SUBTITLE_POSITIONS = {
    SUBTITLE_POSITION_BOTTOM,
    SUBTITLE_POSITION_CENTER,
    SUBTITLE_POSITION_TOP,
}
SUBTITLE_SIZE_SMALL = "pequena"
SUBTITLE_SIZE_MEDIUM = "media"
SUBTITLE_SIZE_LARGE = "grande"
SUBTITLE_SIZES = {
    SUBTITLE_SIZE_SMALL,
    SUBTITLE_SIZE_MEDIUM,
    SUBTITLE_SIZE_LARGE,
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "canal"


def list_channels(status: str | None = None) -> list[dict[str, Any]]:
    query = """
        SELECT c.*, t.name AS template_name, ch.name AS character_name
        FROM channels c
        LEFT JOIN templates t ON t.id = c.template_id
        LEFT JOIN characters ch ON ch.id = c.default_character_id
    """
    params: tuple[Any, ...] = ()
    if status and status != "Todos":
        query += " WHERE c.status = ?"
        params = (status,)
    query += " ORDER BY c.created_at DESC, c.id DESC"
    return fetch_all(query, params)


def get_channel(channel_id: int) -> dict[str, Any] | None:
    return fetch_one("SELECT * FROM channels WHERE id = ?", (channel_id,))


def create_channel(data: dict[str, Any]) -> int:
    return execute(
        """
        INSERT INTO channels (
            name, language, country, niche, subniche, platform, creation_date,
            status, default_tts_model_name, default_voice_name, default_voice_language_code,
            default_tts_audio_format, default_tts_speaking_rate, default_tts_pitch,
            default_tts_volume_gain_db, default_tts_sample_rate_hz,
            template_id, default_character_id, strategic_description, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (
            data["name"],
            data["language"],
            data["country"],
            data["niche"],
            data.get("subniche", ""),
            data["platform"],
            str(data["creation_date"]),
            data["status"],
            data.get("default_tts_model_name") or TTS_MODEL_NAME,
            data.get("default_voice_name") or TTS_VOICE_NAME,
            data.get("default_voice_language_code") or TTS_LANGUAGE_CODE,
            data.get("default_tts_audio_format") or TTS_AUDIO_FORMAT,
            float(data.get("default_tts_speaking_rate") or TTS_SPEAKING_RATE),
            float(data.get("default_tts_pitch") or TTS_PITCH),
            float(data.get("default_tts_volume_gain_db") or TTS_VOLUME_GAIN_DB),
            int(data.get("default_tts_sample_rate_hz") or TTS_SAMPLE_RATE_HZ),
            data.get("template_id"),
            data.get("default_character_id"),
            data.get("strategic_description", ""),
        ),
    )


def update_channel(channel_id: int, data: dict[str, Any]) -> None:
    execute(
        """
        UPDATE channels
        SET name=?, language=?, country=?, niche=?, subniche=?, platform=?, creation_date=?,
            status=?, template_id=?, default_character_id=?, strategic_description=?, updated_at=datetime('now')
        WHERE id=?
        """,
        (
            data["name"],
            data["language"],
            data["country"],
            data["niche"],
            data.get("subniche", ""),
            data["platform"],
            str(data["creation_date"]),
            data["status"],
            data.get("template_id"),
            data.get("default_character_id"),
            data.get("strategic_description", ""),
            channel_id,
        ),
    )


def update_channel_voice_defaults(
    channel_id: int,
    *,
    default_tts_model_name: str,
    default_voice_name: str,
    default_voice_language_code: str,
    default_tts_audio_format: str,
    default_tts_speaking_rate: float,
    default_tts_pitch: float,
    default_tts_volume_gain_db: float,
    default_tts_sample_rate_hz: int,
) -> None:
    execute(
        """
        UPDATE channels
        SET default_tts_model_name=?, default_voice_name=?, default_voice_language_code=?,
            default_tts_audio_format=?, default_tts_speaking_rate=?, default_tts_pitch=?,
            default_tts_volume_gain_db=?, default_tts_sample_rate_hz=?, updated_at=datetime('now')
        WHERE id=?
        """,
        (
            str(default_tts_model_name or "").strip(),
            str(default_voice_name or "").strip(),
            str(default_voice_language_code or "").strip(),
            str(default_tts_audio_format or "").strip().lower(),
            float(default_tts_speaking_rate),
            float(default_tts_pitch),
            float(default_tts_volume_gain_db),
            int(default_tts_sample_rate_hz),
            channel_id,
        ),
    )


def delete_channel(channel_id: int) -> None:
    execute("DELETE FROM channels WHERE id = ?", (channel_id,))


def clone_channel(channel_id: int) -> int:
    row = get_channel(channel_id)
    if not row:
        raise ValueError("Canal não encontrado")
    data = dict(row)
    data["name"] = f"{row['name']} (Clone)"
    data["creation_date"] = row["creation_date"]
    return create_channel(data)


def list_templates() -> list[dict[str, Any]]:
    return fetch_all("SELECT * FROM templates ORDER BY created_at DESC, id DESC")


def get_template(template_id: int) -> dict[str, Any] | None:
    return fetch_one("SELECT * FROM templates WHERE id = ?", (template_id,))


def create_template(data: dict[str, Any]) -> int:
    return execute(
        """
        INSERT INTO templates (
            name, niche, content_strategy, visual_style, script_prompt_base, image_prompt_base,
            thumbnail_prompt, default_persona, extra_settings, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (
            data["name"],
            data["niche"],
            data.get("content_strategy", ""),
            data.get("visual_style", ""),
            data.get("script_prompt_base", ""),
            data.get("image_prompt_base", ""),
            data.get("thumbnail_prompt", ""),
            data.get("default_persona", ""),
            data.get("extra_settings", ""),
        ),
    )


def update_template(template_id: int, data: dict[str, Any]) -> None:
    execute(
        """
        UPDATE templates
        SET name=?, niche=?, content_strategy=?, visual_style=?, script_prompt_base=?, image_prompt_base=?,
            thumbnail_prompt=?, default_persona=?, extra_settings=?, updated_at=datetime('now')
        WHERE id=?
        """,
        (
            data["name"],
            data["niche"],
            data.get("content_strategy", ""),
            data.get("visual_style", ""),
            data.get("script_prompt_base", ""),
            data.get("image_prompt_base", ""),
            data.get("thumbnail_prompt", ""),
            data.get("default_persona", ""),
            data.get("extra_settings", ""),
            template_id,
        ),
    )


def delete_template(template_id: int) -> None:
    execute("DELETE FROM templates WHERE id = ?", (template_id,))


def clone_template(template_id: int) -> int:
    row = get_template(template_id)
    if not row:
        raise ValueError("Template não encontrado")
    data = dict(row)
    data["name"] = f"{row['name']} (Clone)"
    return create_template(data)


def list_characters() -> list[dict[str, Any]]:
    return fetch_all("SELECT * FROM characters ORDER BY created_at DESC, id DESC")


def get_character(character_id: int) -> dict[str, Any] | None:
    return fetch_one("SELECT * FROM characters WHERE id = ?", (character_id,))


def create_character(data: dict[str, Any]) -> int:
    return execute(
        """
        INSERT INTO characters (
            name, description, visual_style, base_prompt, reference_image_path, tags, language, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (
            data["name"],
            data.get("description", ""),
            data.get("visual_style", ""),
            data.get("base_prompt", ""),
            data.get("reference_image_path", ""),
            data.get("tags", ""),
            data.get("language", ""),
        ),
    )


def update_character(character_id: int, data: dict[str, Any]) -> None:
    execute(
        """
        UPDATE characters
        SET name=?, description=?, visual_style=?, base_prompt=?, reference_image_path=?, tags=?, language=?, updated_at=datetime('now')
        WHERE id=?
        """,
        (
            data["name"],
            data.get("description", ""),
            data.get("visual_style", ""),
            data.get("base_prompt", ""),
            data.get("reference_image_path", ""),
            data.get("tags", ""),
            data.get("language", ""),
            character_id,
        ),
    )


def delete_character(character_id: int) -> None:
    execute("DELETE FROM characters WHERE id = ?", (character_id,))


def list_prompts() -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT p.*, c.name AS channel_name, t.name AS template_name
        FROM prompts p
        LEFT JOIN channels c ON c.id = p.channel_id
        LEFT JOIN templates t ON t.id = p.template_id
        ORDER BY p.created_at DESC, p.id DESC
        """
    )


def get_prompt(prompt_id: int) -> dict[str, Any] | None:
    return fetch_one("SELECT * FROM prompts WHERE id = ?", (prompt_id,))


def create_prompt(data: dict[str, Any]) -> int:
    return execute(
        """
        INSERT INTO prompts (name, type, language, full_text, channel_id, template_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (
            data["name"],
            data["type"],
            data.get("language", ""),
            data["full_text"],
            data.get("channel_id"),
            data.get("template_id"),
        ),
    )


def update_prompt(prompt_id: int, data: dict[str, Any]) -> None:
    execute(
        """
        UPDATE prompts
        SET name=?, type=?, language=?, full_text=?, channel_id=?, template_id=?, updated_at=datetime('now')
        WHERE id=?
        """,
        (
            data["name"],
            data["type"],
            data.get("language", ""),
            data["full_text"],
            data.get("channel_id"),
            data.get("template_id"),
            prompt_id,
        ),
    )


def delete_prompt(prompt_id: int) -> None:
    execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT r.*, c.name AS channel_name
        FROM runs r
        JOIN channels c ON c.id = r.channel_id
        ORDER BY r.created_at DESC, r.id DESC
        LIMIT ?
        """,
        (limit,),
    )


def create_run_record(payload: dict[str, Any]) -> int:
    return execute(
        """
        INSERT INTO runs (
            channel_id, video_title, output_dir, status, script_path, scenes_path,
            image_prompts_path, voice_path, run_summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["channel_id"],
            payload["video_title"],
            payload["output_dir"],
            payload["status"],
            payload.get("script_path", ""),
            payload.get("scenes_path", ""),
            payload.get("image_prompts_path", ""),
            payload.get("voice_path", ""),
            payload.get("run_summary", ""),
        ),
    )


def _load_generation_context(channel: dict[str, Any]) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    template = get_template(channel["template_id"]) if channel.get("template_id") else None
    character = get_character(channel["default_character_id"]) if channel.get("default_character_id") else None

    channel_prompts = fetch_all("SELECT * FROM prompts WHERE channel_id = ?", (channel["id"],))
    template_prompts = []
    if channel.get("template_id"):
        template_prompts = fetch_all("SELECT * FROM prompts WHERE template_id = ?", (channel["template_id"],))

    return template, character, channel_prompts, template_prompts


def _normalize_visual_prompt_mode(value: str | None) -> str:
    mode = (value or VISUAL_PROMPT_MODE_IMAGE_ONLY).strip().lower()
    if mode not in VISUAL_PROMPT_MODES:
        return VISUAL_PROMPT_MODE_IMAGE_ONLY
    return mode


def _filter_prompts_by_visual_mode(
    prompts: list[dict[str, Any]],
    *,
    visual_prompt_mode: str,
) -> list[dict[str, Any]]:
    _ = _normalize_visual_prompt_mode(visual_prompt_mode)
    # Keep all associated prompts and let the selected mode control how visual instructions are interpreted.
    return prompts[:]


def _persist_run_outputs(
    *,
    channel: dict[str, Any],
    video_title: str,
    template: dict[str, Any] | None,
    character: dict[str, Any] | None,
    channel_prompts: list[dict[str, Any]],
    template_prompts: list[dict[str, Any]],
    script_text: str,
    scenes: list[dict[str, Any]],
    image_prompts: dict[str, Any],
    voice_text: str,
    provider: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / slugify(channel["name"]) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    template_name = template["name"] if template else "Nenhum"
    character_name = character["name"] if character else "Nenhum"
    template_style = template["visual_style"] if template else "cinemático"
    character_style = character["visual_style"] if character else ""

    if not str(script_text or "").strip():
        script_text = (
            f"Título: {video_title}\n"
            f"Canal: {channel['name']} ({channel['platform']})\n"
            f"Nicho: {channel['niche']} / {channel.get('subniche', '')}\n"
            f"Template: {template_name}\n"
            f"Personagem: {character_name}\n\n"
            "[FALLBACK] Roteiro não retornado; gerado automaticamente.\n"
        )

    if not scenes:
        scenes = [
            {"scene": 1, "description": "Gancho com curiosidade", "duration_sec": 8},
            {"scene": 2, "description": "Explicação principal", "duration_sec": 35},
            {"scene": 3, "description": "Fechamento e CTA", "duration_sec": 12},
        ]

    if not isinstance(image_prompts, dict) or not (image_prompts.get("items") or []):
        image_prompts = {
            "channel": channel["name"],
            "video_title": video_title,
            "template_visual_style": template_style,
            "character_visual_style": character_style,
            "items": [
                {
                    "scene": item["scene"],
                    "prompt": f"{video_title} | Cena {item['scene']} | estilo {template_style}",
                    "duration_sec": int(item.get("duration_sec", 5)),
                }
                for item in scenes
            ],
        }

    if not str(voice_text or "").strip():
        voice_text = (
            "[FALLBACK] Configuração de voz\n"
            f"Idioma: {channel['language']}\n"
            f"Tom sugerido: {'dramático' if channel['niche'] else 'neutro'}\n"
            f"Observação: personagem base = {character_name}\n"
        )

    script_path = run_dir / "script.txt"
    scenes_path = run_dir / "scenes.json"
    image_prompts_path = run_dir / "image_prompts.json"
    voice_path = run_dir / "voice.txt"

    script_path.write_text(script_text, encoding="utf-8")
    scenes_path.write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")
    image_prompts_path.write_text(json.dumps(image_prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    voice_path.write_text(voice_text, encoding="utf-8")

    summary = {
        "provider": provider,
        "model": model_name,
        "template_loaded": template_name if template else None,
        "character_loaded": character_name if character else None,
        "channel_prompts_count": len(channel_prompts),
        "template_prompts_count": len(template_prompts),
    }

    create_run_record(
        {
            "channel_id": channel["id"],
            "video_title": video_title,
            "output_dir": str(run_dir),
            "status": "completed",
            "script_path": str(script_path),
            "scenes_path": str(scenes_path),
            "image_prompts_path": str(image_prompts_path),
            "voice_path": str(voice_path),
            "run_summary": json.dumps(summary, ensure_ascii=False),
        }
    )

    return {
        "provider": provider,
        "model": model_name,
        "output_dir": run_dir,
        "template": template,
        "character": character,
        "channel_prompts": channel_prompts,
        "template_prompts": template_prompts,
        "files": {
            "script.txt": script_path,
            "scenes.json": scenes_path,
            "image_prompts.json": image_prompts_path,
            "voice.txt": voice_path,
        },
    }


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _scene_duration_map(scenes: list[dict[str, Any]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for idx, scene in enumerate(scenes, start=1):
        scene_num = _safe_int(scene.get("scene"), idx) or idx
        duration = _safe_int(scene.get("duration_sec"), 5) or 5
        mapping[scene_num] = max(duration, 1)
    return mapping


def _normalize_image_items_with_durations(
    *,
    raw_items: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    video_title: str,
    template_style: str,
) -> list[dict[str, Any]]:
    durations_by_scene = _scene_duration_map(scenes)
    items: list[dict[str, Any]] = []
    for i, item in enumerate(raw_items, start=1):
        scene_num = _safe_int(item.get("scene"), i) or i
        prompt_text = str(item.get("prompt") or "").strip()
        if not prompt_text:
            continue
        duration_value = _safe_int(item.get("duration_sec"), None)
        items.append(
            {
                "scene": scene_num,
                "prompt": prompt_text,
                "duration_sec": max(duration_value, 1) if duration_value is not None else None,
            }
        )

    if not items:
        return [
            {
                "scene": scene_num,
                "prompt": f"{video_title} | Cena {scene_num} | estilo {template_style}",
                "duration_sec": scene_duration,
            }
            for scene_num, scene_duration in durations_by_scene.items()
        ]

    scenes_with_item = {int(item["scene"]) for item in items}
    for scene_num in durations_by_scene.keys():
        if scene_num not in scenes_with_item:
            items.append(
                {
                    "scene": scene_num,
                    "prompt": f"{video_title} | Cena {scene_num} | estilo {template_style}",
                    "duration_sec": durations_by_scene[scene_num],
                }
            )

    indexes_by_scene: dict[int, list[int]] = {}
    for idx, item in enumerate(items):
        scene_num = int(item["scene"])
        indexes_by_scene.setdefault(scene_num, []).append(idx)

    for scene_num, indexes in indexes_by_scene.items():
        scene_total = durations_by_scene.get(scene_num)
        explicit_sum = 0
        missing_indexes: list[int] = []
        for idx in indexes:
            duration = items[idx].get("duration_sec")
            if isinstance(duration, int) and duration > 0:
                explicit_sum += duration
            else:
                missing_indexes.append(idx)

        if missing_indexes:
            if scene_total is not None:
                remaining = max(scene_total - explicit_sum, len(missing_indexes))
            else:
                remaining = 5 * len(missing_indexes)
            base = max(remaining // len(missing_indexes), 1)
            remainder = max(remaining - (base * len(missing_indexes)), 0)
            for pos, idx in enumerate(missing_indexes):
                bonus = 1 if pos < remainder else 0
                items[idx]["duration_sec"] = base + bonus

    for item in items:
        duration = _safe_int(item.get("duration_sec"), 5) or 5
        item["duration_sec"] = max(duration, 1)

    return items


def _normalize_generated_payload(
    raw_payload: dict[str, Any],
    *,
    channel: dict[str, Any],
    video_title: str,
    template: dict[str, Any] | None,
    character: dict[str, Any] | None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], str]:
    template_style = (template or {}).get("visual_style") or "cinemático"
    character_style = (character or {}).get("visual_style") or ""

    script_text = str(raw_payload.get("script_text") or "").strip()
    if not script_text:
        script_text = (
            f"Título: {video_title}\n"
            f"Canal: {channel['name']}\n\n"
            "[Gemini] Roteiro não retornado em formato esperado.\n"
        )

    raw_scenes = raw_payload.get("scenes") or []
    scenes: list[dict[str, Any]] = []
    for i, item in enumerate(raw_scenes, start=1):
        if not isinstance(item, dict):
            continue
        try:
            scene_num = int(item.get("scene", i))
        except (TypeError, ValueError):
            scene_num = i
        try:
            duration = int(item.get("duration_sec", 10))
        except (TypeError, ValueError):
            duration = 10
        scenes.append(
            {
                "scene": scene_num,
                "description": str(item.get("description") or f"Cena {scene_num}").strip(),
                "duration_sec": max(duration, 1),
            }
        )
    if not scenes:
        scenes = [
            {"scene": 1, "description": "Gancho inicial", "duration_sec": 8},
            {"scene": 2, "description": "Desenvolvimento", "duration_sec": 35},
            {"scene": 3, "description": "Fechamento", "duration_sec": 12},
        ]

    raw_image_prompts = raw_payload.get("image_prompts") or {}
    if isinstance(raw_image_prompts, list):
        raw_items = [item for item in raw_image_prompts if isinstance(item, dict)]
    elif isinstance(raw_image_prompts, dict):
        raw_items = [item for item in (raw_image_prompts.get("items") or []) if isinstance(item, dict)]
    else:
        raw_items = []

    image_items = _normalize_image_items_with_durations(
        raw_items=raw_items,
        scenes=scenes,
        video_title=video_title,
        template_style=template_style,
    )

    image_prompts = {
        "channel": channel["name"],
        "video_title": video_title,
        "template_visual_style": template_style,
        "character_visual_style": character_style,
        "items": image_items,
    }

    voice_text = str(raw_payload.get("voice_text") or "").strip()
    if not voice_text:
        voice_text = (
            "[Gemini] Configuração de voz gerada com fallback\n"
            f"Idioma: {channel['language']}\n"
            "Tom: narrativo\n"
        )

    return script_text, scenes, image_prompts, voice_text


def _gemini_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "script_text": {"type": "string"},
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "scene": {"type": "integer"},
                        "description": {"type": "string"},
                        "duration_sec": {"type": "integer"},
                    },
                    "required": ["scene", "description", "duration_sec"],
                },
            },
            "image_prompts": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "scene": {"type": "integer"},
                                "prompt": {"type": "string"},
                                "duration_sec": {"type": "integer"},
                            },
                            "required": ["scene", "prompt"],
                        },
                    }
                },
                "required": ["items"],
            },
            "voice_text": {"type": "string"},
        },
        "required": ["script_text", "scenes", "image_prompts", "voice_text"],
    }


def _build_gemini_prompt(
    *,
    channel: dict[str, Any],
    video_title: str,
    template: dict[str, Any] | None,
    character: dict[str, Any] | None,
    channel_prompts: list[dict[str, Any]],
    template_prompts: list[dict[str, Any]],
    visual_prompt_mode: str,
) -> str:
    normalized_visual_mode = _normalize_visual_prompt_mode(visual_prompt_mode)
    if normalized_visual_mode == VISUAL_PROMPT_MODE_IMAGE_ONLY:
        visual_rule = "- Use apenas prompts visuais de imagem estática. Ignore instruções de animação/video prompts.\n"
    else:
        visual_rule = (
            "- Pode aproveitar prompts de imagem animada para enriquecer composição visual das cenas, "
            "mas retorne somente image_prompts no JSON.\n"
        )
    context = {
        "channel": {
            "name": channel["name"],
            "language": channel["language"],
            "country": channel["country"],
            "niche": channel["niche"],
            "subniche": channel.get("subniche") or "",
            "platform": channel["platform"],
            "status": channel["status"],
            "strategic_description": channel.get("strategic_description") or "",
        },
        "template": template or {},
        "character": character or {},
        "generation": {
            "visual_prompt_mode": normalized_visual_mode,
        },
        "prompts": {
            "channel_prompts": [
                {
                    "name": p["name"],
                    "type": p["type"],
                    "language": p.get("language") or "",
                    "full_text": p["full_text"],
                }
                for p in channel_prompts
            ],
            "template_prompts": [
                {
                    "name": p["name"],
                    "type": p["type"],
                    "language": p.get("language") or "",
                    "full_text": p["full_text"],
                }
                for p in template_prompts
            ],
        },
    }

    return (
        "Você é um sistema de produção de conteúdo para canais dark.\n"
        "Gere um pacote de produção para UM vídeo com foco em clareza operacional.\n"
        "Respeite o idioma do canal e o contexto abaixo.\n"
        "Retorne APENAS JSON válido no schema solicitado.\n\n"
        f"TÍTULO DO VÍDEO: {video_title}\n\n"
        "Regras:\n"
        "- script_text: roteiro completo pronto para narração.\n"
        "- scenes: 6 a 12 cenas com descrição e duração (duration_sec).\n"
        "- image_prompts.items: 1 ou mais prompts por cena, cada item com scene, prompt e duration_sec.\n"
        f"{visual_rule}"
        "- No hook, prefira cortes mais rápidos (2-4s por imagem); no meio 4-6s; no final 5-8s.\n"
        "- A soma de duration_sec dos image_prompts de cada cena deve aproximar a duration_sec da cena.\n"
        "- voice_text: instruções de voz/narração (tom, ritmo, ênfases).\n"
        "- Não use markdown.\n\n"
        f"CONTEXTO:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def _call_vertex_generate_payload(
    *,
    model_name: str,
    prompt: str,
) -> dict[str, Any]:
    client = _build_vertex_client()

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "temperature": 0.7,
            "response_mime_type": "application/json",
            "response_json_schema": _gemini_json_schema(),
        },
    )

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    if parsed is not None and hasattr(parsed, "model_dump"):
        return parsed.model_dump()

    response_text = str(getattr(response, "text", "") or "").strip()
    if not response_text:
        raise RuntimeError("Vertex AI não retornou texto parseável.")
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Vertex AI retornou JSON inválido: {exc}") from exc


def _build_vertex_client(api_version: str | None = None):
    if not GOOGLE_CLOUD_PROJECT:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT não configurado no .env.")
    if not GOOGLE_CLOUD_LOCATION:
        raise RuntimeError("GOOGLE_CLOUD_LOCATION não configurado no .env.")

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:
        raise RuntimeError(
            "SDK google-genai não instalado. Rode: pip install -r requirements.txt"
        ) from exc

    client = genai.Client(
        vertexai=True,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
        http_options=genai_types.HttpOptions(api_version=(api_version or VERTEX_API_VERSION or "v1")),
    )
    return client


def test_vertex_connection(model_name: str | None = None) -> dict[str, Any]:
    selected_model = (model_name or VERTEX_MODEL).strip() or VERTEX_MODEL
    client = _build_vertex_client()

    start = time.perf_counter()
    response = client.models.generate_content(
        model=selected_model,
        contents=(
            "Teste de conectividade. Responda apenas um JSON com as chaves "
            "status e message. Exemplo: {\"status\":\"ok\",\"message\":\"pong\"}"
        ),
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_json_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["status", "message"],
            },
        },
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    parsed = getattr(response, "parsed", None)
    payload: dict[str, Any]
    if isinstance(parsed, dict):
        payload = parsed
    elif parsed is not None and hasattr(parsed, "model_dump"):
        payload = parsed.model_dump()
    else:
        response_text = str(getattr(response, "text", "") or "").strip()
        if not response_text:
            raise RuntimeError("Vertex AI respondeu sem conteúdo no teste.")
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            payload = {"status": "unknown", "message": response_text}

    return {
        "ok": True,
        "provider": "vertex_ai",
        "project": GOOGLE_CLOUD_PROJECT,
        "location": GOOGLE_CLOUD_LOCATION,
        "model": selected_model,
        "api_version": VERTEX_API_VERSION or "v1",
        "latency_ms": elapsed_ms,
        "response": payload,
    }


def list_gemini_studio_voice_profiles() -> list[dict[str, str]]:
    return GEMINI_STUDIO_VOICE_PROFILES[:]


def list_preferred_tts_models() -> list[dict[str, str]]:
    return PREFERRED_VERTEX_TTS_MODELS[:]


def list_preferred_tts_languages() -> list[dict[str, str]]:
    return PREFERRED_TTS_LANGUAGE_OPTIONS[:]


def _extract_voice_alias(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split("-")[-1].strip()


def _is_gemini_tts_model(model_name: str) -> bool:
    return str(model_name or "").strip().lower().startswith("gemini-")


def _normalize_voice_name_for_tts_model(
    *,
    tts_model_name: str,
    voice_name: str,
) -> str:
    selected = str(voice_name or "").strip()
    if not selected:
        return selected
    if _is_gemini_tts_model(tts_model_name):
        alias = _extract_voice_alias(selected)
        return alias or selected
    return selected


def _classify_vertex_model(name: str) -> str:
    model_name = name.lower()
    if any(token in model_name for token in ["veo", "video"]):
        return "video"
    if any(token in model_name for token in ["image", "imagen"]):
        return "image"
    if any(token in model_name for token in ["tts", "speech", "chirp"]):
        return "voice"
    if "embed" in model_name:
        return "embedding"
    return "text"


def list_vertex_models_catalog() -> list[dict[str, Any]]:
    # Listing publisher models in Vertex is exposed in v1beta1.
    client = _build_vertex_client(api_version="v1beta1")

    try:
        iterator = client.models.list()
    except Exception as exc:
        raise RuntimeError(f"Falha ao listar modelos Vertex: {exc}") from exc

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in iterator:
        if hasattr(item, "model_dump"):
            data = item.model_dump()
        elif isinstance(item, dict):
            data = item
        else:
            data = {}

        name = str(data.get("name") or getattr(item, "name", "") or "").strip()
        display_name = str(data.get("display_name") or getattr(item, "display_name", "") or "").strip()
        if not name and display_name:
            name = display_name
        if not name:
            continue
        short_name = name.split("/")[-1]
        if short_name in seen:
            continue
        seen.add(short_name)

        supported_actions = data.get("supported_actions") or getattr(item, "supported_actions", None) or []
        modality = _classify_vertex_model(short_name)
        results.append(
            {
                "name": short_name,
                "full_name": name,
                "display_name": display_name or short_name,
                "modality": modality,
                "supported_actions": ", ".join(str(x) for x in supported_actions) if supported_actions else "",
            }
        )

    results.sort(key=lambda x: (x["modality"], x["name"]))
    return results


def list_tts_voices_catalog(language_code: str | None = None) -> list[dict[str, Any]]:
    try:
        from google.cloud import texttospeech
    except ImportError as exc:
        raise RuntimeError(
            "SDK google-cloud-texttospeech não instalado. Rode: pip install -r requirements.txt"
        ) from exc

    client = texttospeech.TextToSpeechClient()
    try:
        response = client.list_voices(language_code=language_code or "")
    except Exception as exc:
        message = str(exc)
        if "SERVICE_DISABLED" in message or "texttospeech.googleapis.com" in message:
            raise RuntimeError(
                "Cloud Text-to-Speech API está desativada no projeto. "
                "Ative em: https://console.developers.google.com/apis/api/texttospeech.googleapis.com/overview"
            ) from exc
        raise RuntimeError(f"Falha ao listar vozes TTS: {exc}") from exc

    voices: list[dict[str, Any]] = []
    for voice in response.voices:
        voices.append(
            {
                "name": voice.name,
                "language_codes": list(voice.language_codes),
                "ssml_gender": getattr(voice.ssml_gender, "name", str(voice.ssml_gender)),
                "natural_sample_rate_hertz": int(voice.natural_sample_rate_hertz or 0),
            }
        )
    voices.sort(key=lambda x: (",".join(x["language_codes"]), x["name"]))
    return voices


def generate_mock_run(channel: dict[str, Any], video_title: str) -> dict[str, Any]:
    template, character, channel_prompts, template_prompts = _load_generation_context(channel)

    template_name = template["name"] if template else "Nenhum"
    character_name = character["name"] if character else "Nenhum"
    template_style = template["visual_style"] if template else "cinemático"
    character_style = character["visual_style"] if character else ""

    script_text = (
        f"Título: {video_title}\n"
        f"Canal: {channel['name']} ({channel['platform']})\n"
        f"Nicho: {channel['niche']} / {channel.get('subniche', '')}\n"
        f"Template: {template_name}\n"
        f"Personagem: {character_name}\n\n"
        "[MOCK] Estrutura de roteiro gerada para futura integração com IA.\n"
        "1. Gancho inicial\n2. Desenvolvimento\n3. CTA final\n"
    )

    scenes = [
        {"scene": 1, "description": "Gancho com curiosidade", "duration_sec": 8},
        {"scene": 2, "description": "Explicação principal", "duration_sec": 35},
        {"scene": 3, "description": "Fechamento e CTA", "duration_sec": 12},
    ]

    image_prompts = {
        "channel": channel["name"],
        "video_title": video_title,
        "template_visual_style": template_style,
        "character_visual_style": character_style,
        "items": [
            {
                "scene": item["scene"],
                "prompt": f"{video_title} | Cena {item['scene']} | estilo {template_style}",
                "duration_sec": int(item.get("duration_sec", 5)),
            }
            for item in scenes
        ],
    }

    voice_text = (
        "[MOCK] Configuração de voz\n"
        f"Idioma: {channel['language']}\n"
        f"Tom sugerido: {'dramático' if channel['niche'] else 'neutro'}\n"
        f"Observação: personagem base = {character_name}\n"
    )

    return _persist_run_outputs(
        channel=channel,
        video_title=video_title,
        template=template,
        character=character,
        channel_prompts=channel_prompts,
        template_prompts=template_prompts,
        script_text=script_text,
        scenes=scenes,
        image_prompts=image_prompts,
        voice_text=voice_text,
        provider="mock",
        model_name=None,
    )


def generate_vertex_run(
    channel: dict[str, Any],
    video_title: str,
    *,
    model_name: str | None = None,
    visual_prompt_mode: str | None = None,
) -> dict[str, Any]:
    template, character, channel_prompts, template_prompts = _load_generation_context(channel)
    selected_visual_mode = _normalize_visual_prompt_mode(visual_prompt_mode)
    selected_channel_prompts = _filter_prompts_by_visual_mode(
        channel_prompts,
        visual_prompt_mode=selected_visual_mode,
    )
    selected_template_prompts = _filter_prompts_by_visual_mode(
        template_prompts,
        visual_prompt_mode=selected_visual_mode,
    )
    selected_model = (model_name or VERTEX_MODEL).strip() or VERTEX_MODEL

    prompt = _build_gemini_prompt(
        channel=channel,
        video_title=video_title,
        template=template,
        character=character,
        channel_prompts=selected_channel_prompts,
        template_prompts=selected_template_prompts,
        visual_prompt_mode=selected_visual_mode,
    )
    raw_payload = _call_vertex_generate_payload(model_name=selected_model, prompt=prompt)
    script_text, scenes, image_prompts, voice_text = _normalize_generated_payload(
        raw_payload,
        channel=channel,
        video_title=video_title,
        template=template,
        character=character,
    )

    result = _persist_run_outputs(
        channel=channel,
        video_title=video_title,
        template=template,
        character=character,
        channel_prompts=selected_channel_prompts,
        template_prompts=selected_template_prompts,
        script_text=script_text,
        scenes=scenes,
        image_prompts=image_prompts,
        voice_text=voice_text,
        provider="vertex_ai",
        model_name=selected_model,
    )
    result["prompt_strategy"] = {
        "visual_prompt_mode": selected_visual_mode,
    }
    return result


def _decode_part_image_bytes(part: Any) -> bytes | None:
    inline_data = getattr(part, "inline_data", None)
    if inline_data is None:
        return None
    data = getattr(inline_data, "data", None)
    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        try:
            import base64

            return base64.b64decode(data)
        except Exception:
            return None
    return None


def _generate_vertex_image(
    prompt: str,
    output_path: Path,
    *,
    model_name: str,
    image_aspect_ratio: str | None = None,
) -> tuple[bool, str]:
    try:
        client = _build_vertex_client()
    except Exception as exc:
        return False, f"client_init_error: {exc}"

    config: dict[str, Any] = {
        "response_modalities": ["IMAGE", "TEXT"],
        "temperature": 0.4,
    }
    selected_aspect_ratio = _normalize_video_aspect_ratio(image_aspect_ratio)
    if selected_aspect_ratio:
        config["image_config"] = {
            "aspect_ratio": selected_aspect_ratio,
        }

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
    except Exception as exc:
        return False, f"generate_content_error: {exc}"

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            image_bytes = _decode_part_image_bytes(part)
            if not image_bytes:
                continue
            try:
                output_path.write_bytes(image_bytes)
                return True, ""
            except Exception as exc:
                return False, f"write_bytes_error: {exc}"

    # Some SDK responses may expose generated images differently; try fallback attrs.
    images_attr = getattr(response, "images", None)
    if images_attr:
        try:
            first = images_attr[0]
            data = getattr(first, "image_bytes", None) or getattr(first, "data", None)
            if isinstance(data, bytes):
                output_path.write_bytes(data)
                return True, ""
        except Exception as exc:
            return False, f"images_attr_error: {exc}"

    response_text = str(getattr(response, "text", "") or "").strip()
    if response_text:
        return False, f"no_image_data: {response_text[:300]}"
    return False, "no_image_data"


def _build_image_fallback_prompt(prompt: str, *, scene_number: int) -> str:
    base = " ".join((prompt or "").split())
    if not base:
        return (
            f"Corporate explainer vector illustration for scene {scene_number}. "
            "Single female business character with emerald green suit and black bob hair. "
            "Clean black outlines, flat colors, subtle cel shading, white background, high quality."
        )

    # Remove short quoted snippets to reduce hard requirements for exact on-image text.
    simplified = re.sub(r"[\"'“”‘’][^\"'“”‘’]{1,120}[\"'“”‘’]", "", base).strip()
    simplified = re.sub(r"\s+", " ", simplified)
    simplified = simplified[:700].rstrip(" ,.;:")
    return (
        f"{simplified}. "
        "Important: do not include written words, letters, logos, trademarks, or watermarks. "
        "Use simple neutral icons instead of text."
    )


def _generate_vertex_image_with_retries(
    prompt: str,
    output_path: Path,
    *,
    model_name: str,
    image_aspect_ratio: str | None = None,
    max_attempts: int = 3,
) -> tuple[bool, str, int]:
    attempts = max(int(max_attempts), 1)
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        ok, error = _generate_vertex_image(
            prompt,
            output_path,
            model_name=model_name,
            image_aspect_ratio=image_aspect_ratio,
        )
        if ok:
            return True, "", attempt
        if error:
            errors.append(error)
        if attempt < attempts:
            # Backoff helps with intermittent 429/availability errors.
            time.sleep(1.2 * attempt)
    joined_errors = " | ".join(errors[-3:]) if errors else "unknown_image_generation_error"
    return False, joined_errors, attempts


def _create_placeholder_image(
    output_path: Path,
    *,
    title: str,
    prompt: str,
    scene_number: int,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise RuntimeError("Pillow indisponível para gerar imagem placeholder.") from exc

    image = Image.new("RGB", (width, height), color=(17, 20, 30))
    draw = ImageDraw.Draw(image)
    # Accent gradient-ish blocks
    draw.rectangle((0, 0, width, 140), fill=(33, 43, 67))
    draw.rectangle((0, height - 120, width, height), fill=(24, 28, 42))

    try:
        title_font = ImageFont.truetype("arial.ttf", 42)
        body_font = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    draw.text((50, 34), f"{title} | Cena {scene_number}", fill=(240, 245, 255), font=title_font)
    wrapped = []
    line = ""
    for word in prompt.split():
        test = f"{line} {word}".strip()
        if len(test) > 72 and line:
            wrapped.append(line)
            line = word
        else:
            line = test
    if line:
        wrapped.append(line)

    y = 190
    for row in wrapped[:10]:
        draw.text((60, y), row, fill=(214, 221, 236), font=body_font)
        y += 40

    draw.text(
        (60, height - 80),
        "Placeholder local (falha/indisponibilidade no modelo de imagem)",
        fill=(255, 119, 119),
        font=body_font,
    )
    image.save(output_path, format="PNG")


def _enforce_image_canvas(
    image_path: Path,
    *,
    target_width: int,
    target_height: int,
) -> tuple[bool, str]:
    if target_width <= 0 or target_height <= 0:
        return False, "invalid_target_canvas"
    try:
        from PIL import Image
    except Exception as exc:
        return False, f"pillow_unavailable: {exc}"

    try:
        with Image.open(image_path) as source_image:
            src_width, src_height = source_image.size
            if src_width == target_width and src_height == target_height:
                return True, ""

            scale = max(target_width / max(src_width, 1), target_height / max(src_height, 1))
            resized_width = max(int(round(src_width * scale)), 1)
            resized_height = max(int(round(src_height * scale)), 1)
            resampling_ns = getattr(Image, "Resampling", Image)
            resample_filter = getattr(resampling_ns, "LANCZOS", Image.LANCZOS)
            resized = source_image.convert("RGB").resize((resized_width, resized_height), resample_filter)

            left = max((resized_width - target_width) // 2, 0)
            top = max((resized_height - target_height) // 2, 0)
            fitted = resized.crop((left, top, left + target_width, top + target_height))
            fitted.save(image_path, format="PNG")
            return True, f"canvas_adjusted:{src_width}x{src_height}->{target_width}x{target_height}"
    except Exception as exc:
        return False, f"canvas_adjust_error: {exc}"


def _prompt_cache_key(prompt: str) -> str:
    compact = re.sub(r"\s+", " ", str(prompt or "")).strip().lower()
    if not compact:
        return ""
    return hashlib.sha1(compact.encode("utf-8")).hexdigest()


def _generate_scene_images(
    run_result: dict[str, Any],
    *,
    video_title: str,
    image_model_name: str | None = None,
    image_aspect_ratio: str | None = None,
    video_width: int = VIDEO_WIDTH,
    video_height: int = VIDEO_HEIGHT,
) -> dict[str, Any]:
    model_name = (image_model_name or VERTEX_IMAGE_MODEL).strip() or VERTEX_IMAGE_MODEL
    selected_aspect_ratio = _normalize_video_aspect_ratio(image_aspect_ratio)
    image_prompts_path = Path(run_result["files"]["image_prompts.json"])
    payload = json.loads(image_prompts_path.read_text(encoding="utf-8"))
    items = payload.get("items") or []

    images_dir = Path(run_result["output_dir"]) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[Path] = []
    image_clips: list[dict[str, Any]] = []
    fallback_count = 0
    reused_count = 0
    prompt_cache_reuse_count = 0
    last_success_by_scene: dict[int, Path] = {}
    last_success_global: Path | None = None
    prompt_output_cache: dict[str, Path] = {}
    generation_entries: list[dict[str, Any]] = []

    for idx, item in enumerate(items, start=1):
        scene_number = int(item.get("scene") or idx)
        prompt = str(item.get("prompt") or "").strip()
        prompt_key = _prompt_cache_key(prompt)
        try:
            duration_sec = max(int(item.get("duration_sec", 5)), 1)
        except (TypeError, ValueError):
            duration_sec = 5
        out_path = images_dir / f"scene_{scene_number:03d}_shot_{idx:03d}.png"
        ok = False
        status = "generated"
        detail = ""
        attempt_total = 0
        variant_used = ""
        if prompt_key and prompt_key in prompt_output_cache:
            cache_source = prompt_output_cache[prompt_key]
            if cache_source.exists():
                try:
                    shutil.copy2(cache_source, out_path)
                    ok = True
                    prompt_cache_reuse_count += 1
                    status = "reused_prompt_cache"
                    detail = f"copied_from={cache_source.name}"
                    variant_used = "reuse_prompt_cache"
                except Exception as exc:
                    detail = f"prompt_cache_copy_error: {exc}"

        if prompt:
            if not ok:
                prompt_variants: list[tuple[str, str]] = [("original_prompt", prompt)]
                fallback_prompt = _build_image_fallback_prompt(prompt, scene_number=scene_number)
                if fallback_prompt and fallback_prompt != prompt:
                    prompt_variants.append(("fallback_no_text_logo", fallback_prompt))

                for variant_name, variant_prompt in prompt_variants:
                    max_attempts = 2 if variant_name == "original_prompt" else 1
                    ok, error_detail, attempts_used = _generate_vertex_image_with_retries(
                        variant_prompt,
                        out_path,
                        model_name=model_name,
                        image_aspect_ratio=selected_aspect_ratio,
                        max_attempts=max_attempts,
                    )
                    attempt_total += attempts_used
                    detail = error_detail
                    variant_used = variant_name
                    if ok:
                        break

        if not ok:
            reuse_source = last_success_by_scene.get(scene_number) or last_success_global
            if reuse_source and reuse_source.exists():
                try:
                    shutil.copy2(reuse_source, out_path)
                    ok = True
                    reused_count += 1
                    status = "reused_previous_image"
                    detail = f"copied_from={reuse_source.name}"
                    variant_used = "reuse_previous"
                except Exception as exc:
                    detail = f"{detail} | reuse_error: {exc}".strip(" |")

        if not ok:
            fallback_count += 1
            _create_placeholder_image(
                out_path,
                title=video_title,
                prompt=prompt or f"Cena {scene_number}",
                scene_number=scene_number,
                width=video_width,
                height=video_height,
            )
            status = "placeholder"
            if not detail:
                detail = "generation_failed"
            if not variant_used:
                variant_used = "none"
        else:
            if status == "generated":
                status = "generated"
            last_success_by_scene[scene_number] = out_path
            last_success_global = out_path
        if prompt_key and out_path.exists():
            prompt_output_cache[prompt_key] = out_path

        if out_path.exists():
            canvas_ok, canvas_detail = _enforce_image_canvas(
                out_path,
                target_width=video_width,
                target_height=video_height,
            )
            if canvas_detail:
                detail = f"{detail} | {canvas_detail}".strip(" |")
            if not canvas_ok and status == "generated":
                status = "generated_with_postprocess_warning"

        generated_files.append(out_path)
        image_clips.append(
            {
                "scene": scene_number,
                "image_path": out_path,
                "duration_sec": duration_sec,
            }
        )
        generation_entries.append(
            {
                "idx": idx,
                "scene": scene_number,
                "status": status,
                "variant_used": variant_used,
                "attempts": attempt_total,
                "detail": detail[:500],
                "output_file": out_path.name,
            }
        )

    report_path = images_dir / "image_generation_report.json"
    report_payload = {
        "image_model": model_name,
        "image_aspect_ratio": selected_aspect_ratio,
        "target_canvas": f"{video_width}x{video_height}",
        "total_items": len(items),
        "generated_images": len(items) - fallback_count,
        "reused_images": reused_count,
        "prompt_cache_reused_images": prompt_cache_reuse_count,
        "placeholder_images": fallback_count,
        "entries": generation_entries,
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "images_dir": images_dir,
        "image_files": generated_files,
        "image_clips": image_clips,
        "image_model": model_name,
        "fallback_images": fallback_count,
        "reused_images": reused_count,
        "prompt_cache_reused_images": prompt_cache_reuse_count,
        "image_generation_report_path": report_path,
    }


def _synthesize_voice_audio(
    run_result: dict[str, Any],
    *,
    tts_model_name: str | None = None,
    voice_name: str | None = None,
    language_code: str | None = None,
    audio_format: str | None = None,
    speaking_rate: float | None = None,
    pitch: float | None = None,
    volume_gain_db: float | None = None,
    sample_rate_hz: int | None = None,
) -> dict[str, Any]:
    script_path = Path(run_result["files"]["script.txt"])
    script_text = script_path.read_text(encoding="utf-8").strip()
    if not script_text:
        raise RuntimeError("Roteiro vazio; não foi possível sintetizar narração.")

    preferred_models = {row["name"] for row in PREFERRED_VERTEX_TTS_MODELS}
    selected_tts_model = (tts_model_name or TTS_MODEL_NAME).strip() or TTS_MODEL_NAME
    if selected_tts_model not in preferred_models:
        selected_tts_model = TTS_MODEL_NAME if TTS_MODEL_NAME in preferred_models else "chirp-3.0-generate-001"
    selected_voice_raw = (voice_name or TTS_VOICE_NAME).strip() or TTS_VOICE_NAME
    selected_voice = _normalize_voice_name_for_tts_model(
        tts_model_name=selected_tts_model,
        voice_name=selected_voice_raw,
    )
    selected_lang = (language_code or TTS_LANGUAGE_CODE).strip() or TTS_LANGUAGE_CODE
    selected_format = (audio_format or TTS_AUDIO_FORMAT).strip().lower() or "mp3"
    selected_speaking_rate = float(speaking_rate if speaking_rate is not None else TTS_SPEAKING_RATE)
    selected_pitch = float(pitch if pitch is not None else TTS_PITCH)
    selected_volume_gain_db = float(volume_gain_db if volume_gain_db is not None else TTS_VOLUME_GAIN_DB)
    selected_sample_rate_hz = int(sample_rate_hz if sample_rate_hz is not None else TTS_SAMPLE_RATE_HZ)

    audio_content, ext = _synthesize_tts_audio_bytes(
        text=script_text,
        tts_model_name=selected_tts_model,
        voice_name=selected_voice,
        language_code=selected_lang,
        audio_format=selected_format,
        speaking_rate=selected_speaking_rate,
        pitch=selected_pitch,
        volume_gain_db=selected_volume_gain_db,
        sample_rate_hz=selected_sample_rate_hz,
    )
    output_audio = Path(run_result["output_dir"]) / f"narration.{ext}"
    output_audio.write_bytes(audio_content)

    return {
        "audio_path": output_audio,
        "tts_model": selected_tts_model,
        "voice_name_input": selected_voice_raw,
        "voice_name": selected_voice,
        "language_code": selected_lang,
        "audio_format": ext,
        "speaking_rate": selected_speaking_rate,
        "pitch": selected_pitch,
        "volume_gain_db": selected_volume_gain_db,
        "sample_rate_hz": selected_sample_rate_hz,
    }


def _synthesize_tts_audio_bytes(
    *,
    text: str,
    tts_model_name: str,
    voice_name: str,
    language_code: str,
    audio_format: str,
    speaking_rate: float,
    pitch: float,
    volume_gain_db: float,
    sample_rate_hz: int,
) -> tuple[bytes, str]:
    try:
        from google.cloud import texttospeech
    except ImportError as exc:
        raise RuntimeError(
            "SDK google-cloud-texttospeech não instalado. Rode: pip install -r requirements.txt"
        ) from exc

    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
        model_name=tts_model_name,
    )

    if audio_format == "wav":
        encoding = texttospeech.AudioEncoding.LINEAR16
        ext = "wav"
    elif audio_format == "ogg":
        encoding = texttospeech.AudioEncoding.OGG_OPUS
        ext = "ogg"
    else:
        encoding = texttospeech.AudioEncoding.MP3
        ext = "mp3"

    sr = int(sample_rate_hz)
    if sr < 8000:
        sr = 8000
    if sr > 48000:
        sr = 48000
    audio_config = texttospeech.AudioConfig(
        audio_encoding=encoding,
        speaking_rate=float(speaking_rate),
        pitch=float(pitch),
        volume_gain_db=float(volume_gain_db),
        sample_rate_hertz=sr,
    )
    try:
        response = client.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_config,
        )
    except Exception as exc:
        message = str(exc)
        if "SERVICE_DISABLED" in message or "texttospeech.googleapis.com" in message:
            raise RuntimeError(
                "Cloud Text-to-Speech API está desativada no projeto. "
                "Ative em: https://console.developers.google.com/apis/api/texttospeech.googleapis.com/overview"
            ) from exc
        if "cannot be used with non-Gemini voices" in message:
            raise RuntimeError(
                "Voz incompatível com modelo Gemini TTS. "
                "Use voz Studio no formato de alias (ex.: Achernar, Kore, Zephyr) "
                "ou troque para Chirp 3."
            ) from exc
        raise RuntimeError(f"Falha ao sintetizar voz: {exc}") from exc

    return response.audio_content, ext


def synthesize_tts_preview(
    *,
    tts_model_name: str,
    voice_name: str,
    language_code: str,
    preview_text: str,
    audio_format: str | None = None,
    speaking_rate: float | None = None,
    pitch: float | None = None,
    volume_gain_db: float | None = None,
    sample_rate_hz: int | None = None,
) -> dict[str, Any]:
    text = preview_text.strip()
    if not text:
        raise RuntimeError("Texto de preview vazio.")

    selected_model = (tts_model_name or TTS_MODEL_NAME).strip() or TTS_MODEL_NAME
    selected_format = (audio_format or TTS_AUDIO_FORMAT).strip().lower() or "mp3"
    selected_speaking_rate = float(speaking_rate if speaking_rate is not None else TTS_SPEAKING_RATE)
    selected_pitch = float(pitch if pitch is not None else TTS_PITCH)
    selected_volume_gain_db = float(volume_gain_db if volume_gain_db is not None else TTS_VOLUME_GAIN_DB)
    selected_sample_rate_hz = int(sample_rate_hz if sample_rate_hz is not None else TTS_SAMPLE_RATE_HZ)
    selected_voice_raw = voice_name.strip()
    selected_voice = _normalize_voice_name_for_tts_model(
        tts_model_name=selected_model,
        voice_name=selected_voice_raw,
    )
    selected_lang = language_code.strip()
    audio_bytes, ext = _synthesize_tts_audio_bytes(
        text=text,
        tts_model_name=selected_model,
        voice_name=selected_voice,
        language_code=selected_lang,
        audio_format=selected_format,
        speaking_rate=selected_speaking_rate,
        pitch=selected_pitch,
        volume_gain_db=selected_volume_gain_db,
        sample_rate_hz=selected_sample_rate_hz,
    )
    return {
        "tts_model": selected_model,
        "voice_name_input": selected_voice_raw,
        "voice_name": selected_voice,
        "language_code": selected_lang,
        "audio_format": ext,
        "speaking_rate": selected_speaking_rate,
        "pitch": selected_pitch,
        "volume_gain_db": selected_volume_gain_db,
        "sample_rate_hz": selected_sample_rate_hz,
        "audio_bytes": audio_bytes,
        "preview_text": text,
    }


def _ffmpeg_exists() -> bool:
    if shutil.which(FFMPEG_BIN):
        return True
    try:
        return Path(FFMPEG_BIN).exists()
    except Exception:
        return False


def _run_ffmpeg(args: list[str], *, cwd: Path | None = None) -> None:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ffmpeg não encontrado. Configure FFMPEG_BIN no .env (atual: {FFMPEG_BIN})."
        ) from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or "sem detalhes"
        raise RuntimeError(f"Falha no ffmpeg: {detail[:1200]}")


def _ffprobe_candidates() -> list[str]:
    raw = str(FFMPEG_BIN or "").strip() or "ffmpeg"
    path = Path(raw)
    name = path.name
    candidates: list[str] = []
    if name.lower().startswith("ffmpeg"):
        probe_name = "ffprobe" + name[len("ffmpeg") :]
        probe_path = str(path.with_name(probe_name))
        candidates.append(probe_path)
    candidates.append("ffprobe")
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        key = candidate.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def _probe_audio_duration_sec(audio_path: Path) -> float | None:
    if not audio_path.exists() or not audio_path.is_file():
        return None
    suffix = audio_path.suffix.lower()
    if suffix == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                frame_rate = int(wav_file.getframerate() or 0)
                frame_count = int(wav_file.getnframes() or 0)
                if frame_rate > 0 and frame_count > 0:
                    return frame_count / frame_rate
        except Exception:
            pass

    for ffprobe_bin in _ffprobe_candidates():
        try:
            proc = subprocess.run(
                [
                    ffprobe_bin,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            continue
        if proc.returncode != 0:
            continue
        raw_value = (proc.stdout or "").strip().splitlines()
        if not raw_value:
            continue
        try:
            duration = float(raw_value[0].strip())
        except Exception:
            continue
        if duration > 0:
            return duration
    return None


def _normalize_video_aspect_ratio(value: str | None) -> str:
    ratio = (value or "16:9").strip()
    return ratio if ratio in {"16:9", "9:16"} else "16:9"


def _normalize_motion_intensity(value: str | None) -> str:
    intensity = (value or "suave").strip().lower()
    return intensity if intensity in {"suave", "medio", "forte"} else "suave"


def _resolve_zoom_profile(motion_intensity: str) -> tuple[float, float, float, float]:
    # Tuned to keep motion smooth while avoiding abrupt jumps that look like shake.
    if motion_intensity == "forte":
        return 0.0014, 1.18, 0.0020, 1.50
    if motion_intensity == "medio":
        return 0.0010, 1.12, 0.0015, 1.35
    return 0.0006, 1.08, 0.0010, 1.20


def _resolve_video_dimensions(aspect_ratio: str) -> tuple[int, int]:
    long_edge = max(VIDEO_WIDTH, VIDEO_HEIGHT)
    short_edge = int(round(long_edge * 9 / 16))
    if long_edge % 2:
        long_edge += 1
    if short_edge % 2:
        short_edge += 1
    if aspect_ratio == "9:16":
        return short_edge, long_edge
    return long_edge, short_edge


def _normalize_subtitle_mode(value: str | None) -> str:
    mode = (value or SUBTITLE_MODE_DISABLED).strip().lower()
    return mode if mode in SUBTITLE_MODES else SUBTITLE_MODE_DISABLED


def _normalize_subtitle_density(value: str | None) -> str:
    density = (value or SUBTITLE_DENSITY_BALANCED).strip().lower()
    return density if density in SUBTITLE_DENSITIES else SUBTITLE_DENSITY_BALANCED


def _normalize_subtitle_position(value: str | None) -> str:
    position = (value or SUBTITLE_POSITION_BOTTOM).strip().lower()
    return position if position in SUBTITLE_POSITIONS else SUBTITLE_POSITION_BOTTOM


def _normalize_subtitle_size(value: str | None) -> str:
    size = (value or SUBTITLE_SIZE_MEDIUM).strip().lower()
    return size if size in SUBTITLE_SIZES else SUBTITLE_SIZE_MEDIUM


def _subtitle_chars_per_line(density: str) -> int:
    normalized = _normalize_subtitle_density(density)
    if normalized == SUBTITLE_DENSITY_COMPACT:
        return 42
    if normalized == SUBTITLE_DENSITY_DETAILED:
        return 66
    return 54


def _subtitle_ass_alignment(position: str) -> int:
    normalized = _normalize_subtitle_position(position)
    if normalized == SUBTITLE_POSITION_TOP:
        return 8
    if normalized == SUBTITLE_POSITION_CENTER:
        return 5
    return 2


def _subtitle_ass_font_size(size: str) -> int:
    normalized = _normalize_subtitle_size(size)
    if normalized == SUBTITLE_SIZE_SMALL:
        return 22
    if normalized == SUBTITLE_SIZE_LARGE:
        return 30
    return 26


def _srt_timestamp(total_seconds: float) -> str:
    ms_total = max(int(round(total_seconds * 1000)), 0)
    hours = ms_total // 3_600_000
    ms_total %= 3_600_000
    minutes = ms_total // 60_000
    ms_total %= 60_000
    seconds = ms_total // 1_000
    milliseconds = ms_total % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _chunk_subtitle_line(text: str, *, max_chars: int) -> list[str]:
    words = [w for w in text.split(" ") if w]
    if not words:
        return []
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word]).strip()
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _build_subtitle_entries(
    script_text: str,
    *,
    total_duration_sec: float,
    max_chars_per_line: int = 54,
) -> list[dict[str, Any]]:
    merged_text = " ".join(line.strip() for line in script_text.splitlines() if line.strip())
    merged_text = re.sub(r"\s+", " ", merged_text).strip()
    if not merged_text:
        return []

    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", merged_text) if x.strip()]
    if not sentences:
        sentences = [merged_text]

    chunks: list[str] = []
    for sentence in sentences:
        chunks.extend(_chunk_subtitle_line(sentence, max_chars=max_chars_per_line))
    if not chunks:
        chunks = [merged_text]

    duration_budget = float(total_duration_sec)
    if duration_budget <= 0:
        duration_budget = max(6.0, len(chunks) * 2.4)

    weights = [max(len(re.sub(r"\s+", "", chunk)), 1) for chunk in chunks]
    weights_sum = sum(weights) or 1
    raw_durations = [duration_budget * (w / weights_sum) for w in weights]
    min_seg = 0.35
    durations = [max(value, min_seg) for value in raw_durations]
    total_allocated = sum(durations) or 1.0
    scale = duration_budget / total_allocated
    durations = [max(0.08, value * scale) for value in durations]

    entries: list[dict[str, Any]] = []
    cursor = 0.0
    for idx, (chunk, duration) in enumerate(zip(chunks, durations), start=1):
        start = cursor
        if idx == len(chunks):
            end = duration_budget
        else:
            end = min(duration_budget, start + duration)
            if end <= start:
                end = min(duration_budget, start + 0.08)
                if end <= start:
                    continue
        cursor = end
        entries.append(
            {
                "index": idx,
                "start_sec": start,
                "end_sec": end,
                "text": chunk,
            }
        )
    return entries


def _generate_script_subtitles(
    run_result: dict[str, Any],
    *,
    scenes: list[dict[str, Any]],
    language_code: str,
    audio_path: Path | None = None,
    subtitle_density: str | None = None,
) -> dict[str, Any]:
    script_path = Path(run_result["files"]["script.txt"])
    script_text = script_path.read_text(encoding="utf-8").strip()
    if not script_text:
        raise RuntimeError("Roteiro vazio; não foi possível gerar legendas.")

    duration_source = "scenes"
    total_duration = 0.0
    if audio_path is not None:
        measured = _probe_audio_duration_sec(audio_path)
        if measured and measured > 0:
            total_duration = measured
            duration_source = "audio_probe"
    if total_duration <= 0:
        for scene in scenes:
            total_duration += float(_safe_int(scene.get("duration_sec"), 0) or 0)
    if total_duration <= 0:
        total_duration = max(8.0, len(script_text.split()) / 2.8)
        duration_source = "fallback_words"

    selected_density = _normalize_subtitle_density(subtitle_density)
    max_chars_per_line = _subtitle_chars_per_line(selected_density)

    entries = _build_subtitle_entries(
        script_text,
        total_duration_sec=total_duration,
        max_chars_per_line=max_chars_per_line,
    )
    if not entries:
        raise RuntimeError("Não foi possível gerar entradas de legenda.")

    run_dir = Path(run_result["output_dir"])
    srt_path = run_dir / "captions.srt"
    lines: list[str] = []
    for item in entries:
        lines.append(str(item["index"]))
        lines.append(f"{_srt_timestamp(item['start_sec'])} --> {_srt_timestamp(item['end_sec'])}")
        lines.append(str(item["text"]))
        lines.append("")
    srt_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    report_path = run_dir / "subtitle_generation_report.json"
    report_payload = {
        "generator": "roteiro_proporcional",
        "language_code": language_code,
        "entries": len(entries),
        "duration_source": duration_source,
        "estimated_total_duration_sec": total_duration,
        "subtitle_density": selected_density,
        "max_chars_per_line": max_chars_per_line,
        "subtitle_file": srt_path.name,
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "srt_path": srt_path,
        "report_path": report_path,
        "entries": len(entries),
        "generator": "roteiro_proporcional",
        "duration_source": duration_source,
        "estimated_total_duration_sec": total_duration,
        "subtitle_density": selected_density,
        "max_chars_per_line": max_chars_per_line,
    }


def _ffmpeg_subtitle_path_for_filter(subtitle_path: Path) -> str:
    normalized = str(subtitle_path).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", normalized):
        normalized = normalized[0] + "\\:" + normalized[2:]
    return normalized.replace("'", "\\'")


def _build_final_video(
    run_result: dict[str, Any],
    *,
    image_clips: list[dict[str, Any]],
    audio_path: Path,
    video_width: int,
    video_height: int,
    subtle_motion: bool,
    motion_intensity: str,
    subtitle_mode: str,
    subtitle_language_code: str,
    subtitle_density: str,
    subtitle_position: str,
    subtitle_size: str,
) -> dict[str, Any]:
    if not _ffmpeg_exists():
        raise RuntimeError(
            f"ffmpeg não encontrado no sistema. Instale o ffmpeg e configure FFMPEG_BIN se necessário."
        )

    scenes_path = Path(run_result["files"]["scenes.json"])
    scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
    scenes_list = scenes if isinstance(scenes, list) else []
    durations_by_scene = _scene_duration_map(scenes_list)
    selected_subtitle_mode = _normalize_subtitle_mode(subtitle_mode)
    selected_subtitle_density = _normalize_subtitle_density(subtitle_density)
    selected_subtitle_position = _normalize_subtitle_position(subtitle_position)
    selected_subtitle_size = _normalize_subtitle_size(subtitle_size)

    run_dir = Path(run_result["output_dir"])
    concat_dir = run_dir / "_ffmpeg"
    concat_dir.mkdir(parents=True, exist_ok=True)

    subtitles: dict[str, Any] | None = None
    if selected_subtitle_mode != SUBTITLE_MODE_DISABLED:
        subtitles = _generate_script_subtitles(
            run_result,
            scenes=scenes_list,
            language_code=subtitle_language_code,
            audio_path=audio_path,
            subtitle_density=selected_subtitle_density,
        )

    clip_paths: list[Path] = []
    for idx, clip in enumerate(image_clips, start=1):
        img = Path(clip.get("image_path"))
        scene_num = _safe_int(clip.get("scene"), idx) or idx
        duration = _safe_int(clip.get("duration_sec"), durations_by_scene.get(scene_num, 5)) or 5
        duration = max(duration, 1)
        clip_path = concat_dir / f"clip_{idx:03d}.mp4"
        frames = max(duration * VIDEO_FPS, 1)
        upscale_px = 8000
        if subtle_motion:
            zoom_in_step, zoom_in_cap, zoom_out_step, zoom_out_start = _resolve_zoom_profile(
                motion_intensity
            )
            zoom_in_step_str = f"{zoom_in_step:.4f}"
            zoom_in_cap_str = f"{zoom_in_cap:.2f}"
            zoom_out_step_str = f"{zoom_out_step:.4f}"
            zoom_out_start_str = f"{zoom_out_start:.2f}"
            if scene_num % 2:
                z_expr = f"min(zoom+{zoom_in_step_str},{zoom_in_cap_str})"
            else:
                z_expr = (
                    f"if(lte(zoom,1.0),{zoom_out_start_str},"
                    f"max(1.001,zoom-{zoom_out_step_str}))"
                )
        else:
            z_expr = "1.0"
        vf = (
            f"scale={upscale_px}:-1,"
            f"zoompan=z='{z_expr}':"
            "x='iw/2-(iw/zoom/2)':"
            "y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s={video_width}x{video_height}:fps={VIDEO_FPS},"
            "format=yuv420p"
        )
        _run_ffmpeg(
            [
                FFMPEG_BIN,
                "-y",
                "-loop",
                "1",
                "-framerate",
                str(VIDEO_FPS),
                "-i",
                str(img),
                "-vf",
                vf,
                "-t",
                str(duration),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                str(clip_path),
            ]
        )
        clip_paths.append(clip_path)

    concat_list = concat_dir / "concat_list.txt"
    concat_lines: list[str] = []
    for p in clip_paths:
        safe_path = str(p).replace("'", "''")
        concat_lines.append(f"file '{safe_path}'")
    concat_list.write_text(
        "\n".join(concat_lines),
        encoding="utf-8",
    )

    video_no_audio = run_dir / "video_no_audio.mp4"
    _run_ffmpeg(
        [
            FFMPEG_BIN,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(video_no_audio),
        ]
    )

    final_video = run_dir / "final.mp4"
    if selected_subtitle_mode == SUBTITLE_MODE_BURN_IN and subtitles:
        subtitle_filter_path = _ffmpeg_subtitle_path_for_filter(subtitles["srt_path"])
        subtitle_alignment = _subtitle_ass_alignment(selected_subtitle_position)
        subtitle_font_size = _subtitle_ass_font_size(selected_subtitle_size)
        subtitle_margin_v = 42 if subtitle_alignment in {2, 8} else 20
        subtitle_style = (
            f"Alignment={subtitle_alignment},MarginV={subtitle_margin_v},FontName=Arial,FontSize={subtitle_font_size},"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0"
        )
        _run_ffmpeg(
            [
                FFMPEG_BIN,
                "-y",
                "-i",
                str(video_no_audio),
                "-i",
                str(audio_path),
                "-vf",
                f"subtitles='{subtitle_filter_path}':force_style='{subtitle_style}'",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-shortest",
                str(final_video),
            ]
        )
    else:
        _run_ffmpeg(
            [
                FFMPEG_BIN,
                "-y",
                "-i",
                str(video_no_audio),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-shortest",
                str(final_video),
            ]
        )
    return {
        "video_no_audio": video_no_audio,
        "final_video": final_video,
        "clips_dir": concat_dir,
        "clips_count": len(clip_paths),
        "subtitle_mode": selected_subtitle_mode,
        "subtitle_density": selected_subtitle_density,
        "subtitle_position": selected_subtitle_position,
        "subtitle_size": selected_subtitle_size,
        "subtitles": subtitles,
    }


def generate_vertex_video_run(
    channel: dict[str, Any],
    video_title: str,
    *,
    script_model_name: str | None = None,
    visual_prompt_mode: str | None = None,
    image_model_name: str | None = None,
    video_model_name: str | None = None,
    video_mode: str | None = None,
    tts_model_name: str | None = None,
    voice_name: str | None = None,
    language_code: str | None = None,
    audio_format: str | None = None,
    audio_speaking_rate: float | None = None,
    audio_pitch: float | None = None,
    audio_volume_gain_db: float | None = None,
    audio_sample_rate_hz: int | None = None,
    subtitle_mode: str | None = None,
    subtitle_density: str | None = None,
    subtitle_position: str | None = None,
    subtitle_size: str | None = None,
    assemble_final_video: bool = True,
    video_aspect_ratio: str | None = None,
    subtle_motion: bool = True,
    motion_intensity: str | None = None,
    video_platform_preset: str | None = None,
) -> dict[str, Any]:
    selected_aspect_ratio = _normalize_video_aspect_ratio(video_aspect_ratio)
    selected_motion_intensity = _normalize_motion_intensity(motion_intensity)
    selected_visual_mode = _normalize_visual_prompt_mode(visual_prompt_mode)
    selected_subtitle_mode = _normalize_subtitle_mode(subtitle_mode)
    if not assemble_final_video and selected_subtitle_mode == SUBTITLE_MODE_BURN_IN:
        selected_subtitle_mode = SUBTITLE_MODE_SRT_ONLY
    video_width, video_height = _resolve_video_dimensions(selected_aspect_ratio)
    base = generate_vertex_run(
        channel,
        video_title,
        model_name=script_model_name or VERTEX_SCRIPT_MODEL,
        visual_prompt_mode=selected_visual_mode,
    )

    image_result = _generate_scene_images(
        base,
        video_title=video_title,
        image_model_name=image_model_name or VERTEX_IMAGE_MODEL,
        image_aspect_ratio=selected_aspect_ratio,
        video_width=video_width,
        video_height=video_height,
    )
    voice_result = _synthesize_voice_audio(
        base,
        tts_model_name=tts_model_name or TTS_MODEL_NAME,
        voice_name=voice_name or TTS_VOICE_NAME,
        language_code=language_code or TTS_LANGUAGE_CODE,
        audio_format=audio_format or TTS_AUDIO_FORMAT,
        speaking_rate=audio_speaking_rate if audio_speaking_rate is not None else TTS_SPEAKING_RATE,
        pitch=audio_pitch if audio_pitch is not None else TTS_PITCH,
        volume_gain_db=audio_volume_gain_db if audio_volume_gain_db is not None else TTS_VOLUME_GAIN_DB,
        sample_rate_hz=audio_sample_rate_hz if audio_sample_rate_hz is not None else TTS_SAMPLE_RATE_HZ,
    )
    subtitle_result: dict[str, Any] | None = None
    video_result: dict[str, Any] | None = None
    if assemble_final_video:
        video_result = _build_final_video(
            base,
            image_clips=image_result["image_clips"],
            audio_path=voice_result["audio_path"],
            video_width=video_width,
            video_height=video_height,
            subtle_motion=subtle_motion,
            motion_intensity=selected_motion_intensity,
            subtitle_mode=selected_subtitle_mode,
            subtitle_language_code=(language_code or TTS_LANGUAGE_CODE),
            subtitle_density=subtitle_density or SUBTITLE_DENSITY_BALANCED,
            subtitle_position=subtitle_position or SUBTITLE_POSITION_BOTTOM,
            subtitle_size=subtitle_size or SUBTITLE_SIZE_MEDIUM,
        )
    elif selected_subtitle_mode != SUBTITLE_MODE_DISABLED:
        scenes_path = Path(base["files"]["scenes.json"])
        scenes_payload = json.loads(scenes_path.read_text(encoding="utf-8"))
        subtitle_result = _generate_script_subtitles(
            base,
            scenes=scenes_payload if isinstance(scenes_payload, list) else [],
            language_code=(language_code or TTS_LANGUAGE_CODE),
            audio_path=voice_result["audio_path"],
            subtitle_density=subtitle_density or SUBTITLE_DENSITY_BALANCED,
        )

    base["files"].update(
        {
            "narration_audio": voice_result["audio_path"],
            "images_dir": image_result["images_dir"],
        }
    )
    if video_result:
        base["files"]["video_no_audio.mp4"] = video_result["video_no_audio"]
        base["files"]["final.mp4"] = video_result["final_video"]
    if image_result.get("image_generation_report_path"):
        base["files"]["image_generation_report.json"] = image_result["image_generation_report_path"]
    subtitles_payload = (video_result.get("subtitles") if video_result else None) or subtitle_result
    if subtitles_payload:
        subtitles = subtitles_payload
        base["files"]["captions.srt"] = subtitles["srt_path"]
        base["files"]["subtitle_generation_report.json"] = subtitles["report_path"]
    base["media"] = {
        "script_model": base.get("model") or (script_model_name or VERTEX_SCRIPT_MODEL),
        "image_model": image_result["image_model"],
        "video_model": (video_model_name or VERTEX_VIDEO_MODEL),
        "video_mode": (video_mode or "imagens_estaticas"),
        "tts_model": voice_result["tts_model"],
        "voice_name": voice_result["voice_name"],
        "voice_language_code": voice_result["language_code"],
        "audio_format": voice_result["audio_format"],
        "audio_sample_rate_hz": voice_result["sample_rate_hz"],
        "audio_speaking_rate": voice_result["speaking_rate"],
        "audio_pitch": voice_result["pitch"],
        "audio_volume_gain_db": voice_result["volume_gain_db"],
        "visual_prompt_mode": selected_visual_mode,
        "fallback_images": image_result["fallback_images"],
        "reused_images": image_result["reused_images"],
        "prompt_cache_reused_images": image_result.get("prompt_cache_reused_images", 0),
        "delivery_mode": "video_final" if assemble_final_video else "pacote_producao",
        "final_video_generated": bool(assemble_final_video),
        "clips_count": (video_result["clips_count"] if video_result else len(image_result["image_clips"])),
        "subtitle_mode": (video_result.get("subtitle_mode") if video_result else selected_subtitle_mode),
        "subtitle_density": (
            video_result.get("subtitle_density")
            if video_result
            else (subtitle_density or SUBTITLE_DENSITY_BALANCED)
        ),
        "subtitle_position": (
            video_result.get("subtitle_position")
            if video_result
            else (subtitle_position or SUBTITLE_POSITION_BOTTOM)
        ),
        "subtitle_size": (
            video_result.get("subtitle_size")
            if video_result
            else (subtitle_size or SUBTITLE_SIZE_MEDIUM)
        ),
        "subtitle_generator": (subtitles_payload or {}).get("generator"),
        "subtitle_entries": (subtitles_payload or {}).get("entries", 0),
        "subtitle_duration_source": (subtitles_payload or {}).get("duration_source"),
        "subtitle_estimated_total_duration_sec": (subtitles_payload or {}).get("estimated_total_duration_sec"),
        "video_aspect_ratio": (selected_aspect_ratio if assemble_final_video else None),
        "video_resolution": (f"{video_width}x{video_height}" if assemble_final_video else None),
        "video_subtle_motion": (bool(subtle_motion) if assemble_final_video else None),
        "video_motion_intensity": (selected_motion_intensity if assemble_final_video else None),
        "video_platform_preset": (video_platform_preset or "custom"),
    }
    return base
