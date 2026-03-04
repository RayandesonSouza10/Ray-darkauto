from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import ValidationError

from config import (
    DB_PATH,
    FFMPEG_BIN,
    GOOGLE_APPLICATION_CREDENTIALS,
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_CLOUD_PROJECT,
    OUTPUTS_DIR,
    TTS_AUDIO_FORMAT,
    TTS_LANGUAGE_CODE,
    TTS_MODEL_NAME,
    TTS_PITCH,
    TTS_SAMPLE_RATE_HZ,
    TTS_SPEAKING_RATE,
    TTS_VOLUME_GAIN_DB,
    TTS_VOICE_NAME,
    VERTEX_IMAGE_MODEL,
    VERTEX_MODEL,
    VERTEX_SCRIPT_MODEL,
    VERTEX_VIDEO_MODEL,
)
from db import fetch_all, init_db
from models import ChannelPayload, CharacterPayload, PromptPayload, RunPayload, TemplatePayload
from services import (
    clone_channel,
    clone_template,
    create_channel,
    create_character,
    create_prompt,
    create_template,
    delete_channel,
    delete_character,
    delete_prompt,
    delete_template,
    generate_mock_run,
    generate_vertex_video_run,
    get_channel,
    get_character,
    list_gemini_studio_voice_profiles,
    list_preferred_tts_languages,
    list_preferred_tts_models,
    get_prompt,
    get_template,
    list_channels,
    list_characters,
    list_prompts,
    list_runs,
    list_templates,
    list_tts_voices_catalog,
    list_vertex_models_catalog,
    synthesize_tts_preview,
    test_vertex_connection,
    update_channel,
    update_channel_voice_defaults,
    update_character,
    update_prompt,
    update_template,
)

STATUS_OPTIONS = ["Ideia", "Teste", "Escala", "Pausado", "Abandonado"]
PROMPT_TYPES = ["roteiro", "imagem", "imagem_animada", "thumbnail", "titulo"]
MENU_ITEMS = [
    "Dashboard",
    "Gerar Conteúdo",
    "Canais",
    "Templates",
    "Personagens",
    "Prompts",
    "Modelos",
    "Outputs",
    "Configurações",
]
SUBTITLE_MODE_OPTIONS = [
    "desativada",
    "arquivo_srt",
    "embutida_no_video",
]
SUBTITLE_DENSITY_OPTIONS = ["compacta", "equilibrada", "detalhada"]
SUBTITLE_POSITION_OPTIONS = ["inferior", "centro", "superior"]
SUBTITLE_SIZE_OPTIONS = ["pequena", "media", "grande"]

st.set_page_config(
    page_title="Ray DarkAuto – Dark Channel Factory",
    page_icon="🧠",
    layout="wide",
)

init_db()


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ray-accent: #ff4b4b;
            --ray-ink: #1f2430;
            --ray-soft: #f4f6fb;
            --ray-border: #e8ebf3;
        }
        .stApp {
            background:
                radial-gradient(circle at 92% 8%, rgba(255, 75, 75, 0.06), transparent 36%),
                radial-gradient(circle at 8% 20%, rgba(19, 120, 255, 0.04), transparent 34%),
                #f7f8fc;
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--ray-border);
            background: linear-gradient(180deg, #f3f5fa 0%, #eef1f8 100%);
        }
        [data-testid="stSidebar"] .stRadio > div {
            gap: 0.2rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.92);
            border: 1px solid var(--ray-border);
            border-radius: 14px;
            padding: 0.35rem 0.5rem;
        }
        .ray-page-head {
            background: rgba(255,255,255,0.9);
            border: 1px solid var(--ray-border);
            border-radius: 16px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 8px 30px rgba(16, 24, 40, 0.03);
        }
        .ray-page-head h1 {
            margin: 0;
            font-size: 2.1rem;
            line-height: 1.05;
            color: var(--ray-ink);
        }
        .ray-page-head p {
            margin: 0.25rem 0 0;
            color: #5c6577;
            font-size: 0.95rem;
        }
        .ray-table-caption {
            color: #5c6577;
            font-size: 0.9rem;
            margin: 0.25rem 0 0.45rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--ray-border);
            border-radius: 14px;
            overflow: hidden;
            background: rgba(255,255,255,0.92);
        }
        [data-testid="stForm"] {
            border-radius: 16px;
            border: 1px solid var(--ray-border);
            background: rgba(255,255,255,0.9);
        }
        .stAlert {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=5)
def cached_templates() -> list[dict[str, Any]]:
    return list_templates()


@st.cache_data(ttl=5)
def cached_characters() -> list[dict[str, Any]]:
    return list_characters()


@st.cache_data(ttl=60)
def cached_vertex_models_catalog() -> list[dict[str, Any]]:
    return list_vertex_models_catalog()


@st.cache_data(ttl=300)
def cached_tts_voices_catalog(language_code: str = "") -> list[dict[str, Any]]:
    return list_tts_voices_catalog(language_code or None)


def extract_voice_alias(voice_name: str) -> str:
    return voice_name.split("-")[-1].strip() if voice_name else ""


def build_studio_voice_entries(
    voices_catalog: list[dict[str, Any]],
    language_hint: str = "",
    tts_model_name: str = "",
) -> list[dict[str, Any]]:
    profiles = list_gemini_studio_voice_profiles()
    catalog_by_alias: dict[str, list[dict[str, Any]]] = {}
    for row in voices_catalog:
        provider_name = str(row.get("name") or "").strip()
        if not provider_name:
            continue
        alias = extract_voice_alias(provider_name)
        if alias:
            catalog_by_alias.setdefault(alias, []).append(row)

    hint = (language_hint or "").strip().lower()
    is_gemini_model = (tts_model_name or "").strip().lower().startswith("gemini-")
    entries: list[dict[str, Any]] = []
    for profile in profiles:
        alias = profile["name"]
        style = profile.get("style", "")
        candidates = catalog_by_alias.get(alias, [])
        selected: dict[str, Any] | None = None
        if candidates:
            def score_voice(candidate: dict[str, Any]) -> tuple[int, int, str]:
                langs = [str(x).lower() for x in (candidate.get("language_codes") or [])]
                has_hint = int(bool(hint) and hint in langs)
                name = str(candidate.get("name") or "")
                is_chirp = int("chirp" in name.lower())
                chirp_priority = (1 - is_chirp) if is_gemini_model else is_chirp
                return (has_hint, chirp_priority, name)

            selected = sorted(candidates, key=score_voice, reverse=True)[0]

        provider_name = str(selected.get("name") if selected else alias).strip() or alias
        language_codes = (
            [str(x) for x in (selected.get("language_codes") or [])]
            if selected
            else []
        )
        entries.append(
            {
                "alias": alias,
                "style": style,
                "provider_name": provider_name,
                "language_codes": language_codes,
                "available_in_tts": bool(selected),
            }
        )
    return entries


def build_tts_language_label_map(voices_catalog: list[dict[str, Any]]) -> dict[str, str]:
    labels = {
        str(row.get("code") or "").strip(): f"{row['label']} ({row['code']})"
        for row in list_preferred_tts_languages()
        if str(row.get("code") or "").strip()
    }
    for row in voices_catalog:
        for code in row.get("language_codes") or []:
            normalized = str(code or "").strip()
            if normalized and normalized not in labels:
                labels[normalized] = normalized
    return labels


def infer_tts_voice_family(voice_name: str, *, is_studio: bool = False) -> str:
    lowered = str(voice_name or "").lower()
    if is_studio and "chirp" in lowered:
        return "Studio (Chirp)"
    if is_studio:
        return "Studio"
    if "chirp" in lowered:
        return "Chirp"
    if "neural2" in lowered:
        return "Neural2"
    if "wavenet" in lowered:
        return "WaveNet"
    if "standard" in lowered:
        return "Standard"
    if "journey" in lowered:
        return "Journey"
    return "Outras"


def invalidate_caches() -> None:
    cached_templates.clear()
    cached_characters.clear()


def format_options(rows: list[dict[str, Any]], label_key: str = "name") -> dict[str, int]:
    return {f"#{row['id']} - {row[label_key]}": row["id"] for row in rows}


def select_optional_id(
    label: str,
    rows: list[dict[str, Any]],
    key: str,
    default_id: int | None = None,
) -> int | None:
    options = {"Nenhum": None}
    options.update(format_options(rows))
    labels = list(options.keys())
    default_index = 0
    if default_id is not None:
        for i, item_label in enumerate(labels):
            if options[item_label] == default_id:
                default_index = i
                break
    selected_label = st.selectbox(label, labels, index=default_index, key=key)
    return options[selected_label]


def show_validation_error(exc: ValidationError) -> None:
    first_error = exc.errors()[0]
    location = ".".join(str(x) for x in first_error.get("loc", []))
    st.error(f"Validação falhou em '{location}': {first_error.get('msg')}")


def select_model_name(
    label: str,
    options: list[str],
    *,
    default_value: str,
    key: str,
    help_text: str | None = None,
) -> str:
    values = options[:] if options else []
    if default_value and default_value not in values:
        values = [default_value, *values]
    if not values:
        st.warning(f"Nenhum valor disponível para '{label}'. Usando padrão: {default_value}")
        return default_value
    default_index = values.index(default_value) if default_value in values else 0
    return st.selectbox(label, values, index=default_index, key=key, help=help_text)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def filter_vertex_models_for_task(
    models_catalog: list[dict[str, Any]],
    *,
    task: str,
) -> list[str]:
    rows = [row for row in models_catalog if str(row.get("name") or "").strip()]
    if not rows:
        return []

    def norm(value: Any) -> str:
        return str(value or "").strip().lower()

    def supports_generation(actions_value: Any) -> bool:
        actions = norm(actions_value)
        if not actions:
            return True
        return any(token in actions for token in ["generatecontent", "predict"])

    strict_matches: list[str] = []
    modality_fallback: list[str] = []

    for row in rows:
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        lowered_name = name.lower()
        modality = norm(row.get("modality"))
        actions = row.get("supported_actions")

        if task == "script":
            if modality == "text":
                modality_fallback.append(name)
            if modality != "text" or "gemini" not in lowered_name:
                continue
            if any(
                token in lowered_name
                for token in ["embed", "tts", "speech", "chirp", "image", "imagen", "video", "veo"]
            ):
                continue
            if not supports_generation(actions):
                continue
            strict_matches.append(name)
            continue

        if task == "image":
            if modality == "image":
                modality_fallback.append(name)
            if modality != "image":
                continue
            if not any(token in lowered_name for token in ["image", "imagen"]):
                continue
            if any(
                token in lowered_name
                for token in ["upscale", "edit", "caption", "embed", "tts", "speech", "video", "veo"]
            ):
                continue
            if not supports_generation(actions):
                continue
            strict_matches.append(name)
            continue

        if task == "video":
            if modality == "video":
                modality_fallback.append(name)
            if modality != "video":
                continue
            if not any(token in lowered_name for token in ["veo", "video"]):
                continue
            if not supports_generation(actions):
                continue
            strict_matches.append(name)
            continue

    return _dedupe_preserve_order(strict_matches or modality_fallback)


def render_page_header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="ray-page-head">
            <h1>{title}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_table(data: list[dict[str, Any]], empty_message: str, caption: str | None = None) -> None:
    if caption:
        st.markdown(f'<div class="ray-table-caption">{caption}</div>', unsafe_allow_html=True)
    if not data:
        st.info(empty_message)
        return
    st.dataframe(data, use_container_width=True, hide_index=True)


def render_subtitle_options_preview(
    *,
    subtitle_mode: str,
    subtitle_density: str,
    subtitle_position: str,
    subtitle_size: str,
) -> None:
    density_labels = {
        "compacta": "Compacta",
        "equilibrada": "Equilibrada",
        "detalhada": "Detalhada",
    }
    density_examples = {
        "compacta": "O mercado abriu forte.\nPreço rompeu resistência.\nVolume acima da média.",
        "equilibrada": "O mercado abriu forte e rompeu a resistência.\nO volume ficou acima da média.",
        "detalhada": (
            "O mercado abriu forte e rompeu a resistência do dia anterior, "
            "com volume acima da média e confirmação na segunda hora."
        ),
    }

    st.caption("Pré-visualização rápida das opções de legenda.")
    c1, c2, c3 = st.columns(3)
    for col, code in zip((c1, c2, c3), ("compacta", "equilibrada", "detalhada")):
        selected_suffix = " (selecionada)" if code == subtitle_density else ""
        col.markdown(f"**{density_labels[code]}{selected_suffix}**")
        col.code(density_examples[code], language="text")

    if subtitle_mode != "embutida_no_video":
        st.info("Ative 'SRT automático + legenda embutida no vídeo' para visualizar posição e tamanho.")
        return

    align_map = {"superior": "flex-start", "centro": "center", "inferior": "flex-end"}
    size_map = {"pequena": "20px", "media": "26px", "grande": "32px"}
    align_css = align_map.get(subtitle_position, "flex-end")
    size_css = size_map.get(subtitle_size, "26px")

    st.markdown("**Preview visual (posição + tamanho)**")
    st.markdown(
        f"""
        <div style="
            height:220px;
            border:1px solid #d7dbe5;
            border-radius:12px;
            background:linear-gradient(180deg,#10131a 0%,#1a2230 100%);
            display:flex;
            align-items:{align_css};
            justify-content:center;
            padding:16px;">
            <div style="
                max-width:88%;
                color:#ffffff;
                font-size:{size_css};
                line-height:1.25;
                font-weight:700;
                text-align:center;
                text-shadow:0 0 4px rgba(0,0,0,0.85), 0 0 10px rgba(0,0,0,0.7);">
                Exemplo de legenda em tempo real para pré-visualização
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    channels = list_channels()
    templates = list_templates()
    characters = list_characters()
    runs = list_runs(10)
    active_channels = [c for c in channels if c["status"] in {"Teste", "Escala"}]

    render_page_header(
        "Ray DarkAuto – Dark Channel Factory",
        "Painel local para operação e escala de canais dark",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total de canais", len(channels))
    c2.metric("Total de templates", len(templates))
    c3.metric("Total de personagens", len(characters))
    c4.metric("Canais ativos", len(active_channels))
    c5.metric("Últimas execuções", len(runs))

    st.subheader("Últimas execuções")
    if runs:
        render_table(
            [
                {
                    "ID": run["id"],
                    "Canal": run["channel_name"],
                    "Título": run["video_title"],
                    "Status": run["status"],
                    "Criado em": run["created_at"],
                    "Output": run["output_dir"],
                }
                for run in runs
            ],
            empty_message="Nenhuma execução registrada ainda.",
        )
    else:
        st.info("Nenhuma execução registrada ainda.")


def render_channels() -> None:
    render_page_header("Canais", "Gerencie seu portfólio completo de canais dark")
    templates = cached_templates()
    characters = cached_characters()

    filter_col, _ = st.columns([1, 3])
    status_filter = filter_col.selectbox("Filtrar por status", ["Todos", *STATUS_OPTIONS])
    rows = list_channels(status_filter)

    render_table(
        [
            {
                "ID": r["id"],
                "Nome": r["name"],
                "Idioma": r["language"],
                "País": r["country"],
                "Nicho": r["niche"],
                "Plataforma": r["platform"],
                "Status": r["status"],
                "Template": r.get("template_name") or "—",
                "Personagem": r.get("character_name") or "—",
                "Modelo TTS padrão": str(r.get("default_tts_model_name") or "").strip() or "—",
                "Áudio padrão": (
                    f"{str(r.get('default_tts_audio_format') or '').strip().lower() or TTS_AUDIO_FORMAT} | "
                    f"{int(r.get('default_tts_sample_rate_hz') or TTS_SAMPLE_RATE_HZ)} Hz"
                ),
                "Voz padrão": (
                    f"{str(r.get('default_voice_name') or '').strip()} "
                    f"({str(r.get('default_voice_language_code') or '').strip()})"
                    if str(r.get("default_voice_name") or "").strip()
                    and str(r.get("default_voice_language_code") or "").strip()
                    else (str(r.get("default_voice_name") or "").strip() or "—")
                ),
            }
            for r in rows
        ],
        empty_message="Nenhum canal cadastrado ainda. Use a aba 'Criar' para começar.",
        caption=f"{len(rows)} canal(is) encontrado(s)" + ("" if status_filter == "Todos" else f" com status '{status_filter}'"),
    )

    tab_create, tab_edit, tab_actions, tab_voice = st.tabs(
        ["Criar", "Editar", "Ações", "Voz padrão por canal"]
    )

    with tab_create:
        with st.form("create_channel_form", clear_on_submit=True):
            a, b, c = st.columns(3)
            name = a.text_input("Nome")
            language = b.text_input("Idioma", value="Português")
            country = c.text_input("País", value="Brasil")

            d, e, f = st.columns(3)
            niche = d.text_input("Nicho")
            subniche = e.text_input("Subnicho")
            platform = f.text_input("Plataforma", value="YouTube")

            g, h = st.columns(2)
            creation_date = g.date_input("Data de criação", value=date.today())
            status = h.selectbox("Status", STATUS_OPTIONS)

            template_id = select_optional_id("Template associado", templates, key="create_channel_template")
            character_id = select_optional_id(
                "Personagem padrão (opcional)",
                characters,
                key="create_channel_character",
            )
            strategic_description = st.text_area("Descrição estratégica", height=120)

            submitted = st.form_submit_button("Criar canal", type="primary")
            if submitted:
                try:
                    payload = ChannelPayload(
                        name=name,
                        language=language,
                        country=country,
                        niche=niche,
                        subniche=subniche,
                        platform=platform,
                        creation_date=creation_date,
                        status=status,
                        template_id=template_id,
                        default_character_id=character_id,
                        strategic_description=strategic_description,
                    )
                    create_channel(payload.model_dump())
                    st.success("Canal criado com sucesso.")
                    st.rerun()
                except ValidationError as exc:
                    show_validation_error(exc)

    with tab_edit:
        if not rows:
            st.info("Crie um canal para editar.")
        else:
            selected_id = st.selectbox(
                "Selecione o canal",
                options=[r["id"] for r in rows],
                format_func=lambda x: f"#{x} - {next(r['name'] for r in rows if r['id'] == x)}",
                key="edit_channel_select",
            )
            channel = get_channel(selected_id)
            if channel:
                with st.form("edit_channel_form"):
                    a, b, c = st.columns(3)
                    name = a.text_input("Nome", value=channel["name"])
                    language = b.text_input("Idioma", value=channel["language"])
                    country = c.text_input("País", value=channel["country"])

                    d, e, f = st.columns(3)
                    niche = d.text_input("Nicho", value=channel["niche"])
                    subniche = e.text_input("Subnicho", value=channel.get("subniche") or "")
                    platform = f.text_input("Plataforma", value=channel["platform"])

                    g, h = st.columns(2)
                    creation_date = g.date_input(
                        "Data de criação",
                        value=date.fromisoformat(channel["creation_date"]),
                    )
                    status = h.selectbox(
                        "Status",
                        STATUS_OPTIONS,
                        index=STATUS_OPTIONS.index(channel["status"])
                        if channel["status"] in STATUS_OPTIONS
                        else 0,
                    )

                    template_id = select_optional_id(
                        "Template associado",
                        templates,
                        key=f"edit_channel_template_{selected_id}",
                        default_id=channel.get("template_id"),
                    )
                    character_id = select_optional_id(
                        "Personagem padrão (opcional)",
                        characters,
                        key=f"edit_channel_character_{selected_id}",
                        default_id=channel.get("default_character_id"),
                    )
                    strategic_description = st.text_area(
                        "Descrição estratégica",
                        value=channel.get("strategic_description") or "",
                        height=120,
                    )

                    save = st.form_submit_button("Salvar alterações", type="primary")
                    if save:
                        try:
                            payload = ChannelPayload(
                                name=name,
                                language=language,
                                country=country,
                                niche=niche,
                                subniche=subniche,
                                platform=platform,
                                creation_date=creation_date,
                                status=status,
                                template_id=template_id,
                                default_character_id=character_id,
                                strategic_description=strategic_description,
                            )
                            update_channel(selected_id, payload.model_dump())
                            st.success("Canal atualizado.")
                            st.rerun()
                        except ValidationError as exc:
                            show_validation_error(exc)

    with tab_actions:
        if not rows:
            st.info("Nenhum canal disponível.")
        else:
            selected_id = st.selectbox(
                "Canal para ação",
                options=[r["id"] for r in rows],
                format_func=lambda x: f"#{x} - {next(r['name'] for r in rows if r['id'] == x)}",
                key="channel_action_select",
            )
            action_cols = st.columns(2)
            if action_cols[0].button("Clonar canal", use_container_width=True):
                clone_channel(selected_id)
                st.success("Canal clonado.")
                st.rerun()
            if action_cols[1].button("Excluir canal", use_container_width=True):
                delete_channel(selected_id)
                st.warning("Canal excluído.")
                st.rerun()

    with tab_voice:
        voice_rows = list_channels()
        if not voice_rows:
            st.info("Crie um canal para configurar voz padrão.")
        else:
            selected_voice_channel_id = st.selectbox(
                "Canal para configurar voz padrão",
                options=[r["id"] for r in voice_rows],
                format_func=lambda x: f"#{x} - {next(r['name'] for r in voice_rows if r['id'] == x)}",
                key="channel_voice_defaults_select",
            )
            channel_voice = get_channel(selected_voice_channel_id)
            if not channel_voice:
                st.warning("Canal não encontrado para configuração de voz.")
            else:
                current_tts_model = (
                    str(channel_voice.get("default_tts_model_name") or "").strip() or TTS_MODEL_NAME
                )
                current_voice_name = (
                    str(channel_voice.get("default_voice_name") or "").strip() or TTS_VOICE_NAME
                )
                current_voice_language = (
                    str(channel_voice.get("default_voice_language_code") or "").strip() or TTS_LANGUAGE_CODE
                )
                current_audio_format = (
                    str(channel_voice.get("default_tts_audio_format") or "").strip().lower() or TTS_AUDIO_FORMAT
                )
                if current_audio_format not in {"wav", "mp3", "ogg"}:
                    current_audio_format = TTS_AUDIO_FORMAT if TTS_AUDIO_FORMAT in {"wav", "mp3", "ogg"} else "wav"
                try:
                    current_speaking_rate = float(
                        channel_voice.get("default_tts_speaking_rate")
                        if channel_voice.get("default_tts_speaking_rate") is not None
                        else TTS_SPEAKING_RATE
                    )
                except (TypeError, ValueError):
                    current_speaking_rate = float(TTS_SPEAKING_RATE)
                try:
                    current_pitch = float(
                        channel_voice.get("default_tts_pitch")
                        if channel_voice.get("default_tts_pitch") is not None
                        else TTS_PITCH
                    )
                except (TypeError, ValueError):
                    current_pitch = float(TTS_PITCH)
                try:
                    current_volume_gain_db = float(
                        channel_voice.get("default_tts_volume_gain_db")
                        if channel_voice.get("default_tts_volume_gain_db") is not None
                        else TTS_VOLUME_GAIN_DB
                    )
                except (TypeError, ValueError):
                    current_volume_gain_db = float(TTS_VOLUME_GAIN_DB)
                try:
                    current_sample_rate_hz = int(
                        channel_voice.get("default_tts_sample_rate_hz")
                        if channel_voice.get("default_tts_sample_rate_hz") is not None
                        else TTS_SAMPLE_RATE_HZ
                    )
                except (TypeError, ValueError):
                    current_sample_rate_hz = int(TTS_SAMPLE_RATE_HZ)
                preferred_tts_models = list_preferred_tts_models()
                model_options = [row["name"] for row in preferred_tts_models]
                model_label_map = {
                    row["name"]: f"{row['label']} ({row['name']})"
                    for row in preferred_tts_models
                }
                if current_tts_model not in model_options:
                    model_options = [current_tts_model, *model_options]
                    model_label_map[current_tts_model] = current_tts_model
                selected_tts_model = st.selectbox(
                    "Modelo TTS padrão",
                    options=model_options,
                    index=model_options.index(current_tts_model) if current_tts_model in model_options else 0,
                    format_func=lambda name: model_label_map.get(name, name),
                    key=f"channel_default_tts_model_{selected_voice_channel_id}",
                )
                st.caption(
                    "Padrão atual salvo: "
                    f"modelo `{current_tts_model}` | voz `{current_voice_name}` ({current_voice_language}) | "
                    f"{current_audio_format} {current_sample_rate_hz}Hz | rate {current_speaking_rate} | "
                    f"pitch {current_pitch} | ganho {current_volume_gain_db} dB"
                )

                selected_voice_name = current_voice_name
                selected_voice_language = current_voice_language
                selected_audio_format = current_audio_format
                selected_speaking_rate = current_speaking_rate
                selected_pitch = current_pitch
                selected_volume_gain_db = current_volume_gain_db
                selected_sample_rate_hz = current_sample_rate_hz
                voice_catalog_error: str | None = None
                try:
                    voice_catalog = cached_tts_voices_catalog("")
                except Exception as exc:
                    voice_catalog = []
                    voice_catalog_error = str(exc)

                if voice_catalog_error:
                    st.warning(f"Falha ao carregar catálogo de vozes TTS: {voice_catalog_error}")

                if voice_catalog:
                    language_label_map = build_tts_language_label_map(voice_catalog)
                    language_options = sorted(
                        language_label_map.keys(),
                        key=lambda code: language_label_map.get(code, code).lower(),
                    )
                    if current_voice_language not in language_options:
                        language_options = [current_voice_language, *language_options]
                        language_label_map[current_voice_language] = current_voice_language

                    selected_voice_language = st.selectbox(
                        "Idioma padrão da voz",
                        options=language_options,
                        index=(
                            language_options.index(current_voice_language)
                            if current_voice_language in language_options
                            else 0
                        ),
                        format_func=lambda code: language_label_map.get(code, code),
                        key=f"channel_default_voice_lang_{selected_voice_channel_id}",
                    )
                    voice_name_options = sorted(
                        {
                            str(row.get("name") or "").strip()
                            for row in voice_catalog
                            if str(row.get("name") or "").strip()
                            and selected_voice_language.lower()
                            in [str(code or "").strip().lower() for code in (row.get("language_codes") or [])]
                        }
                    )
                    if current_voice_name not in voice_name_options:
                        voice_name_options = [current_voice_name, *voice_name_options]

                    selected_voice_name = st.selectbox(
                        "Voz padrão (nome técnico)",
                        options=voice_name_options,
                        index=(
                            voice_name_options.index(current_voice_name)
                            if current_voice_name in voice_name_options
                            else 0
                        ),
                        key=f"channel_default_voice_name_{selected_voice_channel_id}",
                    )
                else:
                    selected_voice_language = st.text_input(
                        "Idioma padrão da voz",
                        value=current_voice_language,
                        key=f"channel_default_voice_lang_fallback_{selected_voice_channel_id}",
                    ).strip() or current_voice_language
                    selected_voice_name = st.text_input(
                        "Voz padrão (nome técnico)",
                        value=current_voice_name,
                        key=f"channel_default_voice_name_fallback_{selected_voice_channel_id}",
                    ).strip() or current_voice_name

                st.markdown("**Parâmetros de áudio padrão do canal**")
                audio_col1, audio_col2 = st.columns(2)
                selected_audio_format = audio_col1.selectbox(
                    "Formato de áudio",
                    options=["wav", "mp3", "ogg"],
                    index=(["wav", "mp3", "ogg"].index(current_audio_format) if current_audio_format in {"wav", "mp3", "ogg"} else 0),
                    key=f"channel_default_audio_format_{selected_voice_channel_id}",
                )
                sample_rate_options = [16000, 22050, 24000, 32000, 44100, 48000]
                selected_sample_rate_hz = audio_col2.selectbox(
                    "Sample rate (Hz)",
                    options=sample_rate_options,
                    index=(
                        sample_rate_options.index(current_sample_rate_hz)
                        if current_sample_rate_hz in sample_rate_options
                        else sample_rate_options.index(44100)
                    ),
                    key=f"channel_default_audio_sample_rate_{selected_voice_channel_id}",
                )
                selected_speaking_rate = st.slider(
                    "Velocidade da fala",
                    min_value=0.7,
                    max_value=1.3,
                    value=min(1.3, max(0.7, float(current_speaking_rate))),
                    step=0.05,
                    key=f"channel_default_audio_speaking_rate_{selected_voice_channel_id}",
                )
                selected_pitch = st.slider(
                    "Pitch",
                    min_value=-6.0,
                    max_value=6.0,
                    value=min(6.0, max(-6.0, float(current_pitch))),
                    step=0.5,
                    key=f"channel_default_audio_pitch_{selected_voice_channel_id}",
                )
                selected_volume_gain_db = st.slider(
                    "Ganho de volume (dB)",
                    min_value=-8.0,
                    max_value=8.0,
                    value=min(8.0, max(-8.0, float(current_volume_gain_db))),
                    step=0.5,
                    key=f"channel_default_audio_gain_{selected_voice_channel_id}",
                )

                st.markdown("**Preview da voz nesta aba**")
                preview_text = st.text_area(
                    "Texto para preview da voz",
                    value="Olá, este é um teste da voz padrão deste canal.",
                    height=90,
                    key=f"channel_voice_preview_text_{selected_voice_channel_id}",
                )
                preview_audio_key = f"channel_voice_preview_audio_{selected_voice_channel_id}"
                preview_format_key = f"channel_voice_preview_format_{selected_voice_channel_id}"
                preview_error_key = f"channel_voice_preview_error_{selected_voice_channel_id}"
                if st.button(
                    "Ouvir preview da voz deste canal",
                    key=f"channel_voice_preview_btn_{selected_voice_channel_id}",
                ):
                    try:
                        preview_result = synthesize_tts_preview(
                            tts_model_name=selected_tts_model,
                            voice_name=selected_voice_name,
                            language_code=selected_voice_language,
                            preview_text=preview_text,
                            audio_format=selected_audio_format,
                            speaking_rate=float(selected_speaking_rate),
                            pitch=float(selected_pitch),
                            volume_gain_db=float(selected_volume_gain_db),
                            sample_rate_hz=int(selected_sample_rate_hz),
                        )
                        st.session_state[preview_audio_key] = preview_result["audio_bytes"]
                        st.session_state[preview_format_key] = preview_result["audio_format"]
                        st.session_state.pop(preview_error_key, None)
                    except Exception as exc:
                        st.session_state[preview_error_key] = str(exc)

                if st.session_state.get(preview_error_key):
                    st.error(f"Falha no preview: {st.session_state[preview_error_key]}")
                if st.session_state.get(preview_audio_key):
                    st.audio(
                        st.session_state[preview_audio_key],
                        format=f"audio/{st.session_state.get(preview_format_key, 'mp3')}",
                    )

                if st.button(
                    "Salvar voz padrão deste canal",
                    key=f"save_channel_default_voice_{selected_voice_channel_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    update_channel_voice_defaults(
                        selected_voice_channel_id,
                        default_tts_model_name=selected_tts_model,
                        default_voice_name=selected_voice_name,
                        default_voice_language_code=selected_voice_language,
                        default_tts_audio_format=selected_audio_format,
                        default_tts_speaking_rate=float(selected_speaking_rate),
                        default_tts_pitch=float(selected_pitch),
                        default_tts_volume_gain_db=float(selected_volume_gain_db),
                        default_tts_sample_rate_hz=int(selected_sample_rate_hz),
                    )
                    st.success("Voz padrão do canal atualizada.")
                    st.rerun()


def template_form_fields(prefix: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    data = data or {}
    a, b = st.columns(2)
    name = a.text_input("Nome do template", value=data.get("name", ""), key=f"{prefix}_name")
    niche = b.text_input("Nicho", value=data.get("niche", ""), key=f"{prefix}_niche")
    content_strategy = st.text_area(
        "Estratégia de conteúdo",
        value=data.get("content_strategy", ""),
        height=100,
        key=f"{prefix}_content_strategy",
    )
    visual_style = st.text_area(
        "Estilo visual",
        value=data.get("visual_style", ""),
        height=80,
        key=f"{prefix}_visual_style",
    )
    script_prompt_base = st.text_area(
        "Prompt base de roteiro",
        value=data.get("script_prompt_base", ""),
        height=120,
        key=f"{prefix}_script_prompt_base",
    )
    image_prompt_base = st.text_area(
        "Prompt base de imagens",
        value=data.get("image_prompt_base", ""),
        height=120,
        key=f"{prefix}_image_prompt_base",
    )
    thumbnail_prompt = st.text_area(
        "Prompt de thumbnail",
        value=data.get("thumbnail_prompt", ""),
        height=100,
        key=f"{prefix}_thumbnail_prompt",
    )
    default_persona = st.text_input(
        "Persona padrão",
        value=data.get("default_persona", ""),
        key=f"{prefix}_default_persona",
    )
    extra_settings = st.text_area(
        "Configurações adicionais",
        value=data.get("extra_settings", ""),
        height=100,
        key=f"{prefix}_extra_settings",
    )
    return {
        "name": name,
        "niche": niche,
        "content_strategy": content_strategy,
        "visual_style": visual_style,
        "script_prompt_base": script_prompt_base,
        "image_prompt_base": image_prompt_base,
        "thumbnail_prompt": thumbnail_prompt,
        "default_persona": default_persona,
        "extra_settings": extra_settings,
    }


def render_templates() -> None:
    render_page_header("Templates", "Estruturas reutilizáveis para padronizar operações")
    rows = list_templates()
    render_table(
        [
            {"ID": r["id"], "Nome": r["name"], "Nicho": r["niche"], "Criado em": r["created_at"]}
            for r in rows
        ],
        empty_message="Nenhum template cadastrado ainda.",
        caption=f"{len(rows)} template(s)",
    )

    tab_create, tab_edit, tab_actions = st.tabs(["Criar", "Editar", "Ações"])
    with tab_create:
        with st.form("create_template_form", clear_on_submit=True):
            form_data = template_form_fields("create_template")
            if st.form_submit_button("Criar template", type="primary"):
                try:
                    payload = TemplatePayload(**form_data)
                    create_template(payload.model_dump())
                    invalidate_caches()
                    st.success("Template criado com sucesso.")
                    st.rerun()
                except ValidationError as exc:
                    show_validation_error(exc)

    with tab_edit:
        if not rows:
            st.info("Crie um template para editar.")
        else:
            selected_id = st.selectbox(
                "Selecione o template",
                options=[r["id"] for r in rows],
                format_func=lambda x: f"#{x} - {next(r['name'] for r in rows if r['id'] == x)}",
                key="edit_template_select",
            )
            template = get_template(selected_id)
            if template:
                with st.form("edit_template_form"):
                    form_data = template_form_fields(f"edit_template_{selected_id}", template)
                    if st.form_submit_button("Salvar alterações", type="primary"):
                        try:
                            payload = TemplatePayload(**form_data)
                            update_template(selected_id, payload.model_dump())
                            invalidate_caches()
                            st.success("Template atualizado.")
                            st.rerun()
                        except ValidationError as exc:
                            show_validation_error(exc)

    with tab_actions:
        if not rows:
            st.info("Nenhum template disponível.")
        else:
            selected_id = st.selectbox(
                "Template para ação",
                options=[r["id"] for r in rows],
                format_func=lambda x: f"#{x} - {next(r['name'] for r in rows if r['id'] == x)}",
                key="template_action_select",
            )
            c1, c2 = st.columns(2)
            if c1.button("Clonar template", use_container_width=True):
                clone_template(selected_id)
                invalidate_caches()
                st.success("Template clonado.")
                st.rerun()
            if c2.button("Excluir template", use_container_width=True):
                delete_template(selected_id)
                invalidate_caches()
                st.warning("Template excluído.")
                st.rerun()


def character_form_fields(prefix: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    data = data or {}
    a, b = st.columns(2)
    name = a.text_input("Nome", value=data.get("name", ""), key=f"{prefix}_name")
    language = b.text_input("Idioma (opcional)", value=data.get("language", ""), key=f"{prefix}_language")
    description = st.text_area(
        "Descrição",
        value=data.get("description", ""),
        height=80,
        key=f"{prefix}_description",
    )
    visual_style = st.text_area(
        "Estilo visual",
        value=data.get("visual_style", ""),
        height=80,
        key=f"{prefix}_visual_style",
    )
    base_prompt = st.text_area(
        "Prompt base",
        value=data.get("base_prompt", ""),
        height=120,
        key=f"{prefix}_base_prompt",
    )
    uploaded_file = st.file_uploader(
        "Upload de imagem de referência (opcional)",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"{prefix}_upload",
    )
    reference_image_path = data.get("reference_image_path", "")
    if uploaded_file is not None:
        char_assets_dir = OUTPUTS_DIR / "_character_assets"
        char_assets_dir.mkdir(parents=True, exist_ok=True)
        file_path = char_assets_dir / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())
        reference_image_path = str(file_path)
        st.caption(f"Imagem salva em: {reference_image_path}")
    else:
        reference_image_path = st.text_input(
            "Caminho da imagem de referência",
            value=reference_image_path,
            key=f"{prefix}_reference_image_path",
        )
    tags = st.text_input(
        "Tags (separadas por vírgula)",
        value=data.get("tags", ""),
        key=f"{prefix}_tags",
    )
    return {
        "name": name,
        "description": description,
        "visual_style": visual_style,
        "base_prompt": base_prompt,
        "reference_image_path": reference_image_path,
        "tags": tags,
        "language": language,
    }


def render_characters() -> None:
    render_page_header("Personagens", "Biblioteca global reutilizável de personagens e visuais")
    rows = list_characters()
    render_table(
        [
            {
                "ID": r["id"],
                "Nome": r["name"],
                "Idioma": r.get("language") or "—",
                "Tags": r.get("tags") or "—",
                "Imagem ref": r.get("reference_image_path") or "—",
            }
            for r in rows
        ],
        empty_message="Nenhum personagem cadastrado ainda.",
        caption=f"{len(rows)} personagem(ns)",
    )

    tab_create, tab_edit, tab_delete = st.tabs(["Criar", "Editar", "Excluir"])

    with tab_create:
        with st.form("create_character_form", clear_on_submit=True):
            form_data = character_form_fields("create_character")
            if st.form_submit_button("Criar personagem", type="primary"):
                try:
                    payload = CharacterPayload(**form_data)
                    create_character(payload.model_dump())
                    invalidate_caches()
                    st.success("Personagem criado.")
                    st.rerun()
                except ValidationError as exc:
                    show_validation_error(exc)

    with tab_edit:
        if not rows:
            st.info("Crie um personagem para editar.")
        else:
            selected_id = st.selectbox(
                "Selecione o personagem",
                options=[r["id"] for r in rows],
                format_func=lambda x: f"#{x} - {next(r['name'] for r in rows if r['id'] == x)}",
                key="edit_character_select",
            )
            character = get_character(selected_id)
            if character:
                with st.form("edit_character_form"):
                    form_data = character_form_fields(f"edit_character_{selected_id}", character)
                    if st.form_submit_button("Salvar alterações", type="primary"):
                        try:
                            payload = CharacterPayload(**form_data)
                            update_character(selected_id, payload.model_dump())
                            invalidate_caches()
                            st.success("Personagem atualizado.")
                            st.rerun()
                        except ValidationError as exc:
                            show_validation_error(exc)

    with tab_delete:
        if not rows:
            st.info("Nenhum personagem disponível.")
        else:
            selected_id = st.selectbox(
                "Personagem para excluir",
                options=[r["id"] for r in rows],
                format_func=lambda x: f"#{x} - {next(r['name'] for r in rows if r['id'] == x)}",
                key="delete_character_select",
            )
            if st.button("Excluir personagem", type="secondary"):
                delete_character(selected_id)
                invalidate_caches()
                st.warning("Personagem excluído.")
                st.rerun()


def prompt_form_fields(
    prefix: str,
    channels: list[dict[str, Any]],
    templates: list[dict[str, Any]],
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = data or {}
    a, b, c = st.columns(3)
    name = a.text_input("Nome", value=data.get("name", ""), key=f"{prefix}_name")
    type_value = b.selectbox(
        "Tipo",
        PROMPT_TYPES,
        index=PROMPT_TYPES.index(data.get("type")) if data.get("type") in PROMPT_TYPES else 0,
        key=f"{prefix}_type",
    )
    if type_value == "imagem_animada":
        b.caption("Use para prompts híbridos (imagem + animação); o modo visual decide como interpretar.")
    language = c.text_input("Idioma", value=data.get("language", ""), key=f"{prefix}_language")

    full_text = st.text_area(
        "Texto completo",
        value=data.get("full_text", ""),
        height=160,
        key=f"{prefix}_full_text",
    )
    channel_id = select_optional_id(
        "Canal associado (opcional)",
        channels,
        key=f"{prefix}_channel_id",
        default_id=data.get("channel_id"),
    )
    if type_value == "imagem_animada" and channel_id is None:
        st.caption("Para 'imagem_animada', selecione um canal (ex.: #3 LA JEFA FINANCE).")
    template_id = select_optional_id(
        "Template associado (opcional)",
        templates,
        key=f"{prefix}_template_id",
        default_id=data.get("template_id"),
    )
    return {
        "name": name,
        "type": type_value,
        "language": language,
        "full_text": full_text,
        "channel_id": channel_id,
        "template_id": template_id,
    }


def render_prompts() -> None:
    render_page_header(
        "Prompts",
        "Biblioteca global de prompts para roteiro, imagem, imagem animada, thumbnail e título",
    )
    channels = list_channels()
    templates = list_templates()
    rows = list_prompts()

    render_table(
        [
            {
                "ID": r["id"],
                "Nome": r["name"],
                "Tipo": r["type"],
                "Idioma": r.get("language") or "—",
                "Canal": r.get("channel_name") or "Global",
                "Template": r.get("template_name") or "Global",
            }
            for r in rows
        ],
        empty_message="Nenhum prompt cadastrado ainda.",
        caption=f"{len(rows)} prompt(s)",
    )

    tab_create, tab_edit, tab_delete = st.tabs(["Criar", "Editar", "Excluir"])
    with tab_create:
        with st.form("create_prompt_form", clear_on_submit=True):
            form_data = prompt_form_fields("create_prompt", channels, templates)
            if st.form_submit_button("Criar prompt", type="primary"):
                try:
                    payload = PromptPayload(**form_data)
                    create_prompt(payload.model_dump())
                    st.success("Prompt criado.")
                    st.rerun()
                except ValidationError as exc:
                    show_validation_error(exc)

    with tab_edit:
        if not rows:
            st.info("Crie um prompt para editar.")
        else:
            selected_id = st.selectbox(
                "Selecione o prompt",
                options=[r["id"] for r in rows],
                format_func=lambda x: f"#{x} - {next(r['name'] for r in rows if r['id'] == x)}",
                key="edit_prompt_select",
            )
            prompt = get_prompt(selected_id)
            if prompt:
                with st.form("edit_prompt_form"):
                    form_data = prompt_form_fields(
                        f"edit_prompt_{selected_id}",
                        channels,
                        templates,
                        prompt,
                    )
                    if st.form_submit_button("Salvar alterações", type="primary"):
                        try:
                            payload = PromptPayload(**form_data)
                            update_prompt(selected_id, payload.model_dump())
                            st.success("Prompt atualizado.")
                            st.rerun()
                        except ValidationError as exc:
                            show_validation_error(exc)

    with tab_delete:
        if not rows:
            st.info("Nenhum prompt disponível.")
        else:
            selected_id = st.selectbox(
                "Prompt para excluir",
                options=[r["id"] for r in rows],
                format_func=lambda x: f"#{x} - {next(r['name'] for r in rows if r['id'] == x)}",
                key="delete_prompt_select",
            )
            if st.button("Excluir prompt"):
                delete_prompt(selected_id)
                st.warning("Prompt excluído.")
                st.rerun()


def render_models() -> None:
    render_page_header("Modelos", "Catálogo dinâmico de modelos Vertex e vozes para o pipeline")
    st.link_button(
        "Abrir Model Garden no Google Cloud",
        f"https://console.cloud.google.com/vertex-ai/model-garden?project={GOOGLE_CLOUD_PROJECT or ''}",
    )

    c1, c2 = st.columns(2)
    if c1.button("Atualizar catálogo de modelos Vertex", use_container_width=True):
        cached_vertex_models_catalog.clear()
    if c2.button("Atualizar catálogo de vozes TTS", use_container_width=True):
        cached_tts_voices_catalog.clear()

    tab_vertex, tab_voices = st.tabs(["Modelos Vertex", "Vozes TTS"])

    with tab_vertex:
        try:
            vertex_models = cached_vertex_models_catalog()
            modality_filter = st.selectbox(
                "Filtrar modalidade",
                ["Todos", "text", "image", "video", "voice", "embedding"],
                key="vertex_models_modality_filter",
            )
            filtered = [
                row
                for row in vertex_models
                if modality_filter == "Todos" or row.get("modality") == modality_filter
            ]
            render_table(
                [
                    {
                        "Modelo": row["name"],
                        "Nome completo": row["full_name"],
                        "Display": row["display_name"],
                        "Modalidade": row["modality"],
                        "Ações": row["supported_actions"] or "—",
                    }
                    for row in filtered
                ],
                empty_message="Nenhum modelo retornado pelo Vertex nesta consulta.",
                caption=f"{len(filtered)} modelo(s) exibido(s) de {len(vertex_models)}",
            )
        except Exception as exc:
            st.error(f"Falha ao carregar catálogo de modelos Vertex: {exc}")

    with tab_voices:
        preferred_tts_models = list_preferred_tts_models()
        model_labels = [f"{row['label']} ({row['name']})" for row in preferred_tts_models]
        model_by_label = {f"{row['label']} ({row['name']})": row["name"] for row in preferred_tts_models}
        default_model_index = 0
        for idx, row in enumerate(preferred_tts_models):
            if row["name"] == TTS_MODEL_NAME:
                default_model_index = idx
                break
        selected_model_label = st.selectbox(
            "Modelo TTS recomendado",
            options=model_labels,
            index=default_model_index if model_labels else 0,
            key="models_voice_tts_model",
        )
        preview_tts_model = model_by_label[selected_model_label]

        st.caption(
            "Modelos habilitados nesta aplicação: "
            + ", ".join(row["name"] for row in preferred_tts_models)
        )
        tts_error: str | None = None
        try:
            voices = cached_tts_voices_catalog("")
        except Exception as exc:
            voices = []
            tts_error = str(exc)
        if tts_error:
            st.warning(f"Falha ao carregar catálogo completo do TTS: {tts_error}")

        if not voices:
            st.info("Nenhuma voz retornada pelo catálogo TTS.")
            return

        studio_profiles = list_gemini_studio_voice_profiles()
        studio_aliases = {row["name"] for row in studio_profiles}
        language_label_map = build_tts_language_label_map(voices)
        available_language_codes = sorted(
            language_label_map.keys(),
            key=lambda code: language_label_map.get(code, code).lower(),
        )
        language_options = ["__all__"] + available_language_codes
        default_language_option = (
            TTS_LANGUAGE_CODE if TTS_LANGUAGE_CODE in available_language_codes else "__all__"
        )
        f1, f2, f3 = st.columns([1.2, 1.8, 1.0])
        selected_language_option = f1.selectbox(
            "Idioma",
            options=language_options,
            index=(
                language_options.index(default_language_option)
                if default_language_option in language_options
                else 0
            ),
            format_func=lambda code: (
                "Todos os idiomas"
                if code == "__all__"
                else language_label_map.get(code, code)
            ),
            key="tts_voices_language_filter_select",
        )
        voice_language_filter = (
            "" if selected_language_option == "__all__" else selected_language_option
        )
        voice_search_query = f2.text_input(
            "Buscar voz (nome/alias)",
            value="",
            placeholder="Ex.: Achernar, en-US-Neural2-F, chirp",
            key="tts_voices_search_query",
        ).strip().lower()
        only_studio = f3.checkbox(
            "Somente Studio",
            value=False,
            key="tts_voices_only_studio",
            help="Mostra apenas vozes com alias Gemini Studio detectado.",
        )

        filtered_voices: list[dict[str, Any]] = []
        for row in voices:
            provider_name = str(row.get("name") or "").strip()
            if not provider_name:
                continue
            language_codes = [str(x).strip() for x in (row.get("language_codes") or []) if str(x).strip()]
            alias = extract_voice_alias(provider_name)
            is_studio = alias in studio_aliases
            if (
                voice_language_filter
                and voice_language_filter.lower() not in [code.lower() for code in language_codes]
            ):
                continue
            if only_studio and not is_studio:
                continue
            searchable_text = f"{provider_name} {alias} {' '.join(language_codes)}".lower()
            if voice_search_query and voice_search_query not in searchable_text:
                continue
            filtered_voices.append(
                {
                    "name": provider_name,
                    "alias": alias,
                    "language_codes": language_codes,
                    "ssml_gender": str(row.get("ssml_gender") or ""),
                    "natural_sample_rate_hertz": int(row.get("natural_sample_rate_hertz") or 0),
                    "is_studio": is_studio,
                    "family": infer_tts_voice_family(provider_name, is_studio=is_studio),
                }
            )
        filtered_voices.sort(key=lambda item: (item["family"], item["name"]))

        language_counts: dict[str, int] = {}
        for row in filtered_voices:
            for code in set(row["language_codes"]):
                language_counts[code] = language_counts.get(code, 0) + 1
        language_distribution_rows = [
            {
                "Idioma": language_label_map.get(code, code),
                "Código": code,
                "Qtd de vozes": count,
            }
            for code, count in sorted(language_counts.items(), key=lambda item: (-item[1], item[0]))
        ]

        tab_catalog, tab_studio, tab_preview = st.tabs(
            ["Catálogo completo", "Vozes Studio", "Preview"]
        )

        with tab_catalog:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Vozes no catálogo", len(voices))
            m2.metric("Idiomas disponíveis", len(available_language_codes))
            m3.metric("Vozes exibidas", len(filtered_voices))
            m4.metric("Vozes Studio detectadas", sum(1 for row in voices if extract_voice_alias(str(row.get("name") or "")) in studio_aliases))

            render_table(
                [
                    {
                        "Nome técnico (TTS)": row["name"],
                        "Alias": row["alias"] or "—",
                        "Família": row["family"],
                        "Idiomas": ", ".join(row["language_codes"]) or "—",
                        "Gênero": row["ssml_gender"] or "—",
                        "Sample rate (Hz)": row["natural_sample_rate_hertz"] or "—",
                        "Studio": "Sim" if row["is_studio"] else "Não",
                    }
                    for row in filtered_voices
                ],
                empty_message="Nenhuma voz encontrada com os filtros atuais.",
                caption=(
                    f"{len(filtered_voices)} voz(es) exibida(s) de {len(voices)} "
                    f"| filtro idioma: {voice_language_filter or 'todos'}"
                ),
            )

            st.markdown("**Distribuição por idioma (filtro atual)**")
            render_table(
                language_distribution_rows,
                empty_message="Nenhum idioma encontrado para os filtros atuais.",
            )

        with tab_studio:
            studio_entries = build_studio_voice_entries(
                voices,
                voice_language_filter,
                tts_model_name=preview_tts_model,
            )
            filtered_entries = [
                row
                for row in studio_entries
                if (
                    (not voice_language_filter)
                    or voice_language_filter.lower() in [x.lower() for x in row["language_codes"]]
                )
                and (
                    not voice_search_query
                    or voice_search_query in f"{row['alias']} {row['provider_name']} {row['style']}".lower()
                )
            ]
            render_table(
                [
                    {
                        "Voz Studio": row["alias"],
                        "Estilo": row["style"] or "—",
                        "Nome técnico (TTS)": row["provider_name"],
                        "Idiomas": ", ".join(row["language_codes"]) or "—",
                        "Status": "Disponível" if row["available_in_tts"] else "Não encontrada no catálogo TTS",
                    }
                    for row in filtered_entries
                ],
                empty_message="Nenhuma voz Studio para os filtros atuais.",
                caption=f"{len(filtered_entries)} voz(es) Studio exibida(s) de {len(studio_entries)}",
            )

        with tab_preview:
            preview_voice_pool = filtered_voices or [
                {
                    "name": str(row.get("name") or ""),
                    "alias": extract_voice_alias(str(row.get("name") or "")),
                    "language_codes": [str(x).strip() for x in (row.get("language_codes") or []) if str(x).strip()],
                    "family": infer_tts_voice_family(
                        str(row.get("name") or ""),
                        is_studio=extract_voice_alias(str(row.get("name") or "")) in studio_aliases,
                    ),
                }
                for row in voices
                if str(row.get("name") or "").strip()
            ]
            if not preview_voice_pool:
                st.info("Nenhuma voz disponível para preview.")
            else:
                preview_names = [row["name"] for row in preview_voice_pool]
                preview_default_idx = (
                    preview_names.index(TTS_VOICE_NAME)
                    if TTS_VOICE_NAME in preview_names
                    else 0
                )
                preview_voice_name = st.selectbox(
                    "Escolha a voz para preview",
                    options=preview_names,
                    index=preview_default_idx,
                    key="models_voice_preview_select",
                    format_func=lambda name: (
                        f"{name} "
                        f"({', '.join(next((row['language_codes'] for row in preview_voice_pool if row['name'] == name), [])) or 'sem idioma'})"
                    ),
                )
                selected_preview_entry = next(
                    row for row in preview_voice_pool if row["name"] == preview_voice_name
                )
                preview_default_lang = (
                    selected_preview_entry["language_codes"][0]
                    if selected_preview_entry["language_codes"]
                    else (voice_language_filter or TTS_LANGUAGE_CODE)
                )
                preview_lang = st.text_input(
                    "Idioma para preview",
                    value=preview_default_lang,
                    key="models_voice_preview_language",
                ).strip() or TTS_LANGUAGE_CODE
                st.caption(
                    f"Família: **{selected_preview_entry['family']}** | Nome técnico no TTS: `{preview_voice_name}`"
                )
                preview_audio_format = st.selectbox(
                    "Formato de áudio (preview)",
                    options=["wav", "mp3", "ogg"],
                    index=0,
                    key="models_voice_preview_audio_format",
                )
                preview_speaking_rate = st.slider(
                    "Velocidade (preview)",
                    min_value=0.7,
                    max_value=1.3,
                    value=float(TTS_SPEAKING_RATE),
                    step=0.05,
                    key="models_voice_preview_speaking_rate",
                )
                preview_pitch = st.slider(
                    "Pitch (preview)",
                    min_value=-6.0,
                    max_value=6.0,
                    value=float(TTS_PITCH),
                    step=0.5,
                    key="models_voice_preview_pitch",
                )
                preview_text = st.text_area(
                    "Texto de preview",
                    value="Este é um teste de voz para escolher a melhor narração do canal.",
                    height=90,
                    key="models_voice_preview_text",
                )
                if st.button("Ouvir preview (aba Modelos)", key="models_voice_preview_btn"):
                    try:
                        result = synthesize_tts_preview(
                            tts_model_name=preview_tts_model,
                            voice_name=preview_voice_name,
                            language_code=preview_lang,
                            preview_text=preview_text,
                            audio_format=preview_audio_format,
                            speaking_rate=preview_speaking_rate,
                            pitch=preview_pitch,
                            volume_gain_db=0.0,
                            sample_rate_hz=44100,
                        )
                        st.session_state["models_voice_preview_audio"] = result["audio_bytes"]
                        st.session_state["models_voice_preview_format"] = result["audio_format"]
                        st.session_state.pop("models_voice_preview_error", None)
                    except Exception as exc:
                        st.session_state["models_voice_preview_error"] = str(exc)

                if st.session_state.get("models_voice_preview_error"):
                    st.error(f"Falha no preview: {st.session_state['models_voice_preview_error']}")
                if st.session_state.get("models_voice_preview_audio"):
                    fmt = st.session_state.get("models_voice_preview_format", "mp3")
                    st.audio(st.session_state["models_voice_preview_audio"], format=f"audio/{fmt}")


def render_generate_content() -> None:
    render_page_header("Gerar Conteúdo", "Pipeline local com geração via Vertex AI ou Mock")
    channels = list_channels()
    if not channels:
        st.info("Crie ao menos um canal antes de iniciar a geração.")
        return

    vertex_status = st.session_state.get("vertex_connection_status")
    if vertex_status == "ok":
        model = st.session_state.get("vertex_connection_model", VERTEX_MODEL)
        latency = st.session_state.get("vertex_connection_latency_ms")
        latency_text = f" | {latency} ms" if isinstance(latency, int) else ""
        st.success(f"Vertex AI conectado ({model}){latency_text}")
    elif vertex_status == "error":
        error_msg = st.session_state.get("vertex_connection_error", "Falha desconhecida")
        st.warning(f"Último teste Vertex AI falhou: {error_msg}")
    else:
        st.info("Dica: use a aba 'Configurações' para testar a conexão com Vertex AI antes de gerar.")

    channel_map = {f"#{c['id']} - {c['name']} ({c['platform']})": c["id"] for c in channels}
    selected_label = st.selectbox("1) Selecionar canal", list(channel_map.keys()))
    selected_channel_id = channel_map[selected_label]
    channel = get_channel(selected_channel_id)
    if not channel:
        st.error("Canal selecionado não encontrado.")
        return

    channel_default_voice_name = str(channel.get("default_voice_name") or "").strip()
    channel_default_voice_language = str(channel.get("default_voice_language_code") or "").strip()
    channel_default_tts_model = str(channel.get("default_tts_model_name") or "").strip()
    channel_default_audio_format = str(channel.get("default_tts_audio_format") or "").strip().lower()
    if channel_default_audio_format not in {"wav", "mp3", "ogg"}:
        channel_default_audio_format = (
            TTS_AUDIO_FORMAT if TTS_AUDIO_FORMAT in {"wav", "mp3", "ogg"} else "wav"
        )
    try:
        channel_default_speaking_rate = float(
            channel.get("default_tts_speaking_rate")
            if channel.get("default_tts_speaking_rate") is not None
            else TTS_SPEAKING_RATE
        )
    except (TypeError, ValueError):
        channel_default_speaking_rate = float(TTS_SPEAKING_RATE)
    try:
        channel_default_pitch = float(
            channel.get("default_tts_pitch")
            if channel.get("default_tts_pitch") is not None
            else TTS_PITCH
        )
    except (TypeError, ValueError):
        channel_default_pitch = float(TTS_PITCH)
    try:
        channel_default_volume_gain_db = float(
            channel.get("default_tts_volume_gain_db")
            if channel.get("default_tts_volume_gain_db") is not None
            else TTS_VOLUME_GAIN_DB
        )
    except (TypeError, ValueError):
        channel_default_volume_gain_db = float(TTS_VOLUME_GAIN_DB)
    try:
        channel_default_sample_rate_hz = int(
            channel.get("default_tts_sample_rate_hz")
            if channel.get("default_tts_sample_rate_hz") is not None
            else TTS_SAMPLE_RATE_HZ
        )
    except (TypeError, ValueError):
        channel_default_sample_rate_hz = int(TTS_SAMPLE_RATE_HZ)
    effective_channel_tts_model = channel_default_tts_model or TTS_MODEL_NAME
    effective_channel_voice_name = channel_default_voice_name or TTS_VOICE_NAME
    effective_channel_voice_language = channel_default_voice_language or TTS_LANGUAGE_CODE
    st.caption(
        "Padrão de áudio do canal: "
        f"modelo `{effective_channel_tts_model}` | "
        f"voz `{effective_channel_voice_name}` ({effective_channel_voice_language}) | "
        f"{channel_default_audio_format} {channel_default_sample_rate_hz}Hz | "
        f"rate {channel_default_speaking_rate} | pitch {channel_default_pitch} | "
        f"ganho {channel_default_volume_gain_db} dB"
    )

    raw_video_title = st.text_input(
        "2) Título do vídeo",
        placeholder="Ex.: 7 mistérios que ninguém explica",
    )
    video_title = raw_video_title.strip()
    if not video_title:
        st.caption("Preencha o título para habilitar o START.")
    vertex_ready = bool(GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION)
    generation_mode = st.radio(
        "Modo de geração",
        ["Vertex AI", "Mock local"],
        index=0 if vertex_ready else 1,
        horizontal=True,
        help="Use Vertex AI para gerar conteúdo real, ou Mock local para testes sem custo.",
    )
    model_name = VERTEX_MODEL
    script_model_name = VERTEX_SCRIPT_MODEL
    image_model_name = VERTEX_IMAGE_MODEL
    video_model_name = VERTEX_VIDEO_MODEL
    video_mode = "imagens_estaticas"
    visual_prompt_mode = "somente_imagens"
    video_platform_preset = "youtube"
    video_aspect_ratio = "16:9"
    video_subtle_motion = True
    video_motion_intensity = "suave"
    tts_model_name = effective_channel_tts_model
    voice_name = effective_channel_voice_name
    voice_language_code = effective_channel_voice_language
    audio_format = channel_default_audio_format
    audio_speaking_rate = channel_default_speaking_rate
    audio_pitch = channel_default_pitch
    audio_volume_gain_db = channel_default_volume_gain_db
    audio_sample_rate_hz = channel_default_sample_rate_hz
    subtitle_mode = "desativada"
    subtitle_density = "equilibrada"
    subtitle_position = "inferior"
    subtitle_size = "media"
    build_final_mp4 = True
    if generation_mode == "Vertex AI":
        vertex_models_catalog: list[dict[str, Any]] = []
        voices_catalog: list[dict[str, Any]] = []
        catalog_error_msgs: list[str] = []
        try:
            vertex_models_catalog = cached_vertex_models_catalog()
        except Exception as exc:
            catalog_error_msgs.append(f"Modelos Vertex: {exc}")
        try:
            voices_catalog = cached_tts_voices_catalog("")
        except Exception as exc:
            catalog_error_msgs.append(f"Vozes TTS: {exc}")

        text_models = filter_vertex_models_for_task(vertex_models_catalog, task="script")
        image_models = filter_vertex_models_for_task(vertex_models_catalog, task="image")
        video_models = filter_vertex_models_for_task(vertex_models_catalog, task="video")
        st.caption(
            f"Modelos filtrados para seu pipeline: roteiro={len(text_models)} "
            f"| imagem={len(image_models)} | vídeo={len(video_models)}"
        )

        st.markdown("**Configuração do pipeline**")
        cfg_tab_script, cfg_tab_image, cfg_tab_delivery, cfg_tab_audio = st.tabs(
            ["Roteiro", "Imagem", "Entrega", "Áudio e Legenda"]
        )
        with cfg_tab_script:
            script_model_name = select_model_name(
                "Modelo de roteiro (texto)",
                text_models,
                default_value=VERTEX_SCRIPT_MODEL,
                key="gen_script_model",
                help_text="Modelo Vertex para gerar roteiro, cenas e prompts.",
            )
        with cfg_tab_image:
            image_model_name = select_model_name(
                "Modelo de imagens",
                image_models,
                default_value=VERTEX_IMAGE_MODEL,
                key="gen_image_model",
                help_text="Modelo Vertex para gerar imagens por cena.",
            )
        with cfg_tab_delivery:
            delivery_mode = st.radio(
                "Saída do pipeline",
                options=[
                    "Pacote de produção (roteiro + áudio + imagens)",
                    "Vídeo final (.mp4)",
                ],
                index=1,
                horizontal=True,
                key="gen_delivery_mode",
                help="Escolha se quer só os assets de produção ou o vídeo final renderizado.",
            )
            build_final_mp4 = delivery_mode == "Vídeo final (.mp4)"
            if build_final_mp4:
                video_mode = st.selectbox(
                    "Estratégia de vídeo por cena",
                    [
                        "imagens_estaticas",
                        "animar_imagens",
                        "gerar_videos_por_cena",
                    ],
                    format_func=lambda x: {
                        "imagens_estaticas": "Imagens estáticas (pipeline atual)",
                        "animar_imagens": "Animar imagens (planejado)",
                        "gerar_videos_por_cena": "Gerar vídeos por cena (planejado)",
                    }[x],
                    key="gen_video_mode",
                    help="A UI já permite escolher a estratégia; o pipeline atual ainda usa imagens estáticas + ffmpeg.",
                )
            else:
                video_mode = "imagens_estaticas"
                st.info(
                    "Modo pacote ativo: vou gerar e salvar roteiro, imagens, narração e arquivos auxiliares sem renderizar final.mp4."
                )
            visual_prompt_mode = st.selectbox(
                "Uso de prompts visuais",
                options=["somente_imagens", "imagens_e_animacao"],
                index=0,
                key="gen_visual_prompt_mode",
                format_func=lambda x: {
                    "somente_imagens": "Somente prompts de imagem",
                    "imagens_e_animacao": "Prompts de imagem + imagem animada",
                }[x],
                help=(
                    "Controla como o modelo interpreta prompts visuais que têm partes de "
                    "imagem e de animação."
                ),
            )
            if visual_prompt_mode == "somente_imagens":
                st.caption(
                    "Modo ativo: o modelo vai ignorar instruções de animação e focar apenas em prompts de imagem."
                )
            else:
                st.caption(
                    "Modo ativo: o modelo pode aproveitar instruções de animação para enriquecer o visual."
                )
            if build_final_mp4:
                video_platform_preset = st.selectbox(
                    "Preset de plataforma",
                    options=["youtube", "shorts_reels_tiktok", "custom"],
                    index=0,
                    key="gen_video_platform_preset",
                    format_func=lambda x: {
                        "youtube": "YouTube (horizontal)",
                        "shorts_reels_tiktok": "Shorts/Reels/TikTok (vertical)",
                        "custom": "Personalizado",
                    }[x],
                )
                selected_aspect_ratio = st.selectbox(
                    "Aspect ratio do vídeo final",
                    options=["16:9", "9:16"],
                    index=0,
                    key="gen_video_aspect_ratio",
                    format_func=lambda x: "16:9 (YouTube horizontal)" if x == "16:9" else "9:16 (Shorts/Reels)",
                )
                if video_platform_preset == "youtube":
                    video_aspect_ratio = "16:9"
                    st.caption("Preset aplicado: YouTube usa 16:9.")
                elif video_platform_preset == "shorts_reels_tiktok":
                    video_aspect_ratio = "9:16"
                    st.caption("Preset aplicado: Shorts/Reels/TikTok usa 9:16.")
                else:
                    video_aspect_ratio = selected_aspect_ratio
                video_subtle_motion = st.checkbox(
                    "Aplicar movimento leve nas cenas (zoom in/out)",
                    value=True,
                    key="gen_video_subtle_motion",
                    help="Aplica um efeito suave tipo Ken Burns, alternando zoom in e zoom out por cena.",
                )
                if video_subtle_motion:
                    video_motion_intensity = st.selectbox(
                        "Intensidade do movimento",
                        options=["suave", "medio", "forte"],
                        index=0,
                        key="gen_video_motion_intensity",
                        format_func=lambda x: {
                            "suave": "Suave",
                            "medio": "Médio",
                            "forte": "Forte",
                        }[x],
                    )
                video_model_name = select_model_name(
                    "Modelo de vídeo",
                    video_models,
                    default_value=VERTEX_VIDEO_MODEL,
                    key="gen_video_model",
                    help_text="Modelo Vertex de vídeo (ex.: Veo). Será usado na evolução do pipeline.",
                )
                if video_mode != "imagens_estaticas":
                    st.info(
                        "A estratégia escolhida está preparada na UI, mas o pipeline atual ainda monta o MP4 "
                        "a partir de imagens por cena. Vou adaptar a geração de vídeo nativa na próxima etapa."
                    )
            else:
                video_platform_preset = "custom"
                video_aspect_ratio = "16:9"
                video_subtle_motion = True
                video_motion_intensity = "suave"
                video_model_name = VERTEX_VIDEO_MODEL
        with cfg_tab_audio:
            st.markdown("**Configuração de voz**")
            preferred_tts_models = list_preferred_tts_models()
            preferred_tts_languages = list_preferred_tts_languages()
            model_labels = [f"{row['label']} ({row['name']})" for row in preferred_tts_models]
            model_by_label = {f"{row['label']} ({row['name']})": row["name"] for row in preferred_tts_models}
            if tts_model_name and tts_model_name not in model_by_label.values():
                custom_model_label = f"Canal salvo ({tts_model_name})"
                model_labels = [custom_model_label, *model_labels]
                model_by_label[custom_model_label] = tts_model_name
            default_tts_model_label = model_labels[0] if model_labels else ""
            for row in preferred_tts_models:
                label = f"{row['label']} ({row['name']})"
                if row["name"] == tts_model_name:
                    default_tts_model_label = label
                    break
            selected_tts_model_label = st.selectbox(
                "Modelo TTS",
                options=model_labels,
                index=model_labels.index(default_tts_model_label) if default_tts_model_label in model_labels else 0,
                key="gen_tts_model_name",
            )
            tts_model_name = model_by_label[selected_tts_model_label]

            language_codes = [row["code"] for row in preferred_tts_languages]
            language_label_map = {row["code"]: f"{row['label']} ({row['code']})" for row in preferred_tts_languages}
            default_lang_code = (
                voice_language_code
                if voice_language_code in language_codes
                else (language_codes[0] if language_codes else "pt-BR")
            )
            selected_lang_code = st.selectbox(
                "Idioma da voz",
                options=language_codes,
                index=language_codes.index(default_lang_code) if default_lang_code in language_codes else 0,
                format_func=lambda code: language_label_map.get(code, code),
                key="gen_voice_language_select",
            )
            voice_language_code = selected_lang_code

            studio_entries = build_studio_voice_entries(
                voices_catalog,
                voice_language_code,
                tts_model_name=tts_model_name,
            )
            visible_studio_entries = [
                row
                for row in studio_entries
                if not voice_language_code
                or voice_language_code.lower() in [x.lower() for x in row["language_codes"]]
            ]
            if not visible_studio_entries:
                visible_studio_entries = studio_entries

            entry_by_alias = {row["alias"]: row for row in visible_studio_entries}
            voice_alias_options = list(entry_by_alias.keys())
            default_alias = extract_voice_alias(voice_name)
            if default_alias not in voice_alias_options and voice_alias_options:
                default_alias = voice_alias_options[0]
            if voice_alias_options:
                voice_alias = st.selectbox(
                    "Voz (Google Studio)",
                    options=voice_alias_options,
                    index=voice_alias_options.index(default_alias) if default_alias in voice_alias_options else 0,
                    key="gen_voice_name",
                    format_func=lambda x: f"{x} — {entry_by_alias[x]['style']}",
                    help="Vozes Studio de alta qualidade.",
                )
                selected_voice_entry = entry_by_alias[voice_alias]
                voice_name = selected_voice_entry["provider_name"]
            else:
                voice_name = st.text_input(
                    "Voz (nome técnico)",
                    value=voice_name,
                    key="gen_voice_name_fallback",
                ).strip() or voice_name

            st.caption(
                f"Vozes Studio: {len(studio_entries)} total"
                f" | exibindo {len(visible_studio_entries)}"
                + (f" para {voice_language_code}" if voice_language_code else "")
            )
            st.caption(f"Nome técnico usado no TTS: `{voice_name}`")

            with st.expander("Parâmetros de áudio", expanded=True):
                audio_format = st.selectbox(
                    "Formato",
                    options=["wav", "mp3", "ogg"],
                    index=(["wav", "mp3", "ogg"].index(audio_format) if audio_format in {"wav", "mp3", "ogg"} else 0),
                    key="gen_audio_format",
                    help="WAV (LINEAR16) tende a preservar mais qualidade para edição e mix.",
                )
                sample_rate_options = [16000, 22050, 24000, 32000, 44100, 48000]
                default_sample_rate = (
                    int(audio_sample_rate_hz)
                    if int(audio_sample_rate_hz) in sample_rate_options
                    else 44100
                )
                audio_sample_rate_hz = st.selectbox(
                    "Sample rate (Hz)",
                    options=sample_rate_options,
                    index=sample_rate_options.index(default_sample_rate),
                    key="gen_audio_sample_rate_hz",
                )
                audio_speaking_rate = st.slider(
                    "Velocidade da fala",
                    min_value=0.7,
                    max_value=1.3,
                    value=min(1.3, max(0.7, float(audio_speaking_rate))),
                    step=0.05,
                    key="gen_audio_speaking_rate",
                )
                audio_pitch = st.slider(
                    "Pitch",
                    min_value=-6.0,
                    max_value=6.0,
                    value=min(6.0, max(-6.0, float(audio_pitch))),
                    step=0.5,
                    key="gen_audio_pitch",
                )
                audio_volume_gain_db = st.slider(
                    "Ganho de volume (dB)",
                    min_value=-8.0,
                    max_value=8.0,
                    value=min(8.0, max(-8.0, float(audio_volume_gain_db))),
                    step=0.5,
                    key="gen_audio_volume_gain_db",
                )

            st.markdown("**Legenda**")
            subtitle_mode = st.selectbox(
                "Gerador de legenda",
                options=SUBTITLE_MODE_OPTIONS,
                index=0,
                key="gen_subtitle_mode",
                format_func=lambda x: {
                    "desativada": "Desativada",
                    "arquivo_srt": "SRT automático (arquivo .srt)",
                    "embutida_no_video": "SRT automático + legenda embutida no vídeo",
                }[x],
                help=(
                    "No momento, a legenda é gerada automaticamente a partir do roteiro "
                    "com sincronização proporcional."
                ),
            )
            subtitle_density = st.selectbox(
                "Densidade da legenda",
                options=SUBTITLE_DENSITY_OPTIONS,
                index=SUBTITLE_DENSITY_OPTIONS.index("equilibrada"),
                key="gen_subtitle_density",
                format_func=lambda x: {
                    "compacta": "Compacta (mais cortes, frases curtas)",
                    "equilibrada": "Equilibrada",
                    "detalhada": "Detalhada (blocos mais longos)",
                }[x],
                help="Controla o tamanho médio das frases em cada bloco da legenda.",
            )
            if not build_final_mp4 and subtitle_mode == "embutida_no_video":
                st.caption("No modo pacote não existe renderização de vídeo. Ajustando legenda para arquivo SRT.")
                subtitle_mode = "arquivo_srt"
            if subtitle_mode == "embutida_no_video":
                c_sub1, c_sub2 = st.columns(2)
                subtitle_position = c_sub1.selectbox(
                    "Posição da legenda",
                    options=SUBTITLE_POSITION_OPTIONS,
                    index=SUBTITLE_POSITION_OPTIONS.index("inferior"),
                    key="gen_subtitle_position",
                    format_func=lambda x: {
                        "inferior": "Inferior",
                        "centro": "Centro",
                        "superior": "Superior",
                    }[x],
                )
                subtitle_size = c_sub2.selectbox(
                    "Tamanho da legenda",
                    options=SUBTITLE_SIZE_OPTIONS,
                    index=SUBTITLE_SIZE_OPTIONS.index("media"),
                    key="gen_subtitle_size",
                    format_func=lambda x: {
                        "pequena": "Pequena",
                        "media": "Média",
                        "grande": "Grande",
                    }[x],
                )
                st.caption("Legenda embutida exige reencode do vídeo final (mais lento, resultado mais limpo no player).")
            elif subtitle_mode == "arquivo_srt":
                st.caption("Gera `captions.srt` para upload na plataforma ou edição externa.")
            with st.expander("Visualizar opções de legenda"):
                render_subtitle_options_preview(
                    subtitle_mode=subtitle_mode,
                    subtitle_density=subtitle_density,
                    subtitle_position=subtitle_position,
                    subtitle_size=subtitle_size,
                )

            preview_text = st.text_area(
                "Texto para preview da voz",
                value="Olá, eu sou a voz do seu canal dark. Este é um teste de narração.",
                height=100,
                key="gen_voice_preview_text",
            )
            if st.button("Ouvir preview da voz selecionada", key="btn_voice_preview"):
                try:
                    with st.spinner("Gerando preview de voz..."):
                        preview = synthesize_tts_preview(
                            tts_model_name=tts_model_name,
                            voice_name=voice_name,
                            language_code=voice_language_code,
                            preview_text=preview_text,
                            audio_format=audio_format,
                            speaking_rate=audio_speaking_rate,
                            pitch=audio_pitch,
                            volume_gain_db=audio_volume_gain_db,
                            sample_rate_hz=audio_sample_rate_hz,
                        )
                    st.session_state["voice_preview_audio_bytes"] = preview["audio_bytes"]
                    st.session_state["voice_preview_audio_format"] = preview["audio_format"]
                    st.session_state["voice_preview_voice_name"] = preview["voice_name"]
                    st.session_state["voice_preview_language_code"] = preview["language_code"]
                    st.session_state["voice_preview_tts_model"] = preview["tts_model"]
                    st.session_state.pop("voice_preview_error", None)
                except Exception as exc:
                    st.session_state["voice_preview_error"] = str(exc)

            if st.session_state.get("voice_preview_error"):
                st.error(f"Falha no preview da voz: {st.session_state['voice_preview_error']}")
            if st.session_state.get("voice_preview_audio_bytes"):
                preview_fmt = st.session_state.get("voice_preview_audio_format", "mp3")
                st.success(
                    "Preview pronto: "
                    f"{st.session_state.get('voice_preview_voice_name', voice_name)} "
                    f"({st.session_state.get('voice_preview_language_code', voice_language_code)})"
                    f" | modelo: {st.session_state.get('voice_preview_tts_model', tts_model_name)}"
                )
                st.audio(
                    st.session_state["voice_preview_audio_bytes"],
                    format=f"audio/{preview_fmt}",
                )
        if vertex_ready:
            st.success(
                f"Vertex configurado: project={GOOGLE_CLOUD_PROJECT} | location={GOOGLE_CLOUD_LOCATION}"
            )
            if GOOGLE_APPLICATION_CREDENTIALS:
                st.caption(f"ADC via service account: `{GOOGLE_APPLICATION_CREDENTIALS}`")
            else:
                st.caption("ADC via ambiente/sessão local (ex.: gcloud auth application-default login)")
        else:
            st.warning(
                "Vertex não configurado (faltando GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION). "
                "Configure no .env ou use 'Mock local'."
            )
        for msg in catalog_error_msgs:
            st.warning(msg)

    if channel:
        template_name = "Nenhum"
        character_name = "Nenhum"
        if channel.get("template_id"):
            template = get_template(channel["template_id"])
            template_name = template["name"] if template else "Nenhum"
        if channel.get("default_character_id"):
            character = get_character(channel["default_character_id"])
            character_name = character["name"] if character else "Nenhum"
        st.caption(f"Template: {template_name} | Personagem padrão: {character_name}")

    if st.button("3) START", type="primary", use_container_width=True, disabled=not bool(video_title)):
        try:
            payload = RunPayload(channel_id=selected_channel_id, video_title=video_title)
            with st.spinner("Executando pipeline..."):
                if generation_mode == "Vertex AI":
                    result = generate_vertex_video_run(
                        channel,
                        payload.video_title,
                        script_model_name=script_model_name,
                        visual_prompt_mode=visual_prompt_mode,
                        image_model_name=image_model_name,
                        video_model_name=video_model_name,
                        video_mode=video_mode,
                        tts_model_name=tts_model_name,
                        voice_name=voice_name,
                        language_code=voice_language_code,
                        audio_format=audio_format,
                        audio_speaking_rate=audio_speaking_rate,
                        audio_pitch=audio_pitch,
                        audio_volume_gain_db=audio_volume_gain_db,
                        audio_sample_rate_hz=audio_sample_rate_hz,
                        subtitle_mode=subtitle_mode,
                        subtitle_density=subtitle_density,
                        subtitle_position=subtitle_position,
                        subtitle_size=subtitle_size,
                        assemble_final_video=build_final_mp4,
                        video_aspect_ratio=video_aspect_ratio,
                        subtle_motion=video_subtle_motion,
                        motion_intensity=video_motion_intensity,
                        video_platform_preset=video_platform_preset,
                    )
                else:
                    result = generate_mock_run(channel, payload.video_title)
            provider_label = "Vertex AI" if result.get("provider") == "vertex_ai" else "Mock local"
            st.success(f"Execução concluída ({provider_label}).")

            st.write("Pipeline executado:")
            st.write(f"- Provider: {provider_label}")
            if result.get("model"):
                st.write(f"- Modelo: {result['model']}")
            if result.get("media"):
                media = result["media"]
                final_video_generated = bool(media.get("final_video_generated"))
                delivery_label = (
                    "Vídeo final (.mp4)"
                    if final_video_generated
                    else "Pacote de produção (roteiro + áudio + imagens, sem final.mp4)"
                )
                st.write(f"- Saída: {delivery_label}")
                st.write(f"- Modelo de imagens: {media.get('image_model')}")
                st.write(f"- Modelo de vídeo: {media.get('video_model')} ({media.get('video_mode')})")
                if final_video_generated and media.get("video_aspect_ratio"):
                    st.write(
                        f"- Formato final: {media.get('video_aspect_ratio')} ({media.get('video_resolution', '-')})"
                    )
                if final_video_generated and media.get("video_platform_preset"):
                    st.write(f"- Preset de plataforma: {media.get('video_platform_preset')}")
                if final_video_generated and "video_subtle_motion" in media:
                    st.write(
                        f"- Movimento de câmera suave: {'Sim' if media.get('video_subtle_motion') else 'Não'}"
                    )
                if final_video_generated and media.get("video_subtle_motion"):
                    st.write(f"- Intensidade do movimento: {media.get('video_motion_intensity', 'suave')}")
                st.write(f"- Modelo TTS: {media.get('tts_model')}")
                st.write(f"- Voz: {media.get('voice_name')} ({media.get('voice_language_code')})")
                st.write(
                    f"- Áudio: {media.get('audio_format')} | {media.get('audio_sample_rate_hz')} Hz"
                    f" | velocidade {media.get('audio_speaking_rate')}"
                    f" | pitch {media.get('audio_pitch')}"
                    f" | ganho {media.get('audio_volume_gain_db')} dB"
                )
                subtitle_mode_label = {
                    "desativada": "Desativada",
                    "arquivo_srt": "SRT automático",
                    "embutida_no_video": "SRT automático + embutida no vídeo",
                }.get(media.get("subtitle_mode", "desativada"), str(media.get("subtitle_mode")))
                st.write(f"- Legenda: {subtitle_mode_label}")
                if media.get("subtitle_mode") != "desativada":
                    density_label = {
                        "compacta": "Compacta",
                        "equilibrada": "Equilibrada",
                        "detalhada": "Detalhada",
                    }.get(media.get("subtitle_density"), str(media.get("subtitle_density")))
                    st.write(f"- Densidade da legenda: {density_label}")
                if media.get("subtitle_mode") == "embutida_no_video":
                    position_label = {
                        "inferior": "Inferior",
                        "centro": "Centro",
                        "superior": "Superior",
                    }.get(media.get("subtitle_position"), str(media.get("subtitle_position")))
                    size_label = {
                        "pequena": "Pequena",
                        "media": "Média",
                        "grande": "Grande",
                    }.get(media.get("subtitle_size"), str(media.get("subtitle_size")))
                    st.write(f"- Estilo da legenda embutida: posição {position_label} | tamanho {size_label}")
                if media.get("subtitle_entries"):
                    st.write(
                        f"- Gerador de legenda: {media.get('subtitle_generator')} "
                        f"({media.get('subtitle_entries')} blocos)"
                    )
                    if media.get("subtitle_duration_source"):
                        subtitle_duration_val = media.get("subtitle_estimated_total_duration_sec")
                        subtitle_duration_text = (
                            f"{float(subtitle_duration_val):.2f}s"
                            if subtitle_duration_val is not None
                            else "n/d"
                        )
                        st.write(
                            f"- Sincronização da legenda: {media.get('subtitle_duration_source')} "
                            f"({subtitle_duration_text})"
                        )
                if media.get("visual_prompt_mode"):
                    mode_label = (
                        "Somente prompts de imagem"
                        if media.get("visual_prompt_mode") == "somente_imagens"
                        else "Prompts de imagem + imagem animada"
                    )
                    st.write(f"- Modo de prompts visuais: {mode_label}")
                st.write(f"- Imagens reutilizadas por contingência: {media.get('reused_images', 0)}")
                st.write(f"- Imagens reutilizadas por prompt repetido (cache): {media.get('prompt_cache_reused_images', 0)}")
                st.write(f"- Imagens placeholder (fallback): {media.get('fallback_images', 0)}")
            elif result.get("prompt_strategy"):
                prompt_strategy = result.get("prompt_strategy") or {}
                mode = prompt_strategy.get("visual_prompt_mode")
                if mode:
                    mode_label = (
                        "Somente prompts de imagem"
                        if mode == "somente_imagens"
                        else "Prompts de imagem + imagem animada"
                    )
                    st.write(f"- Modo de prompts visuais: {mode_label}")
            st.write(f"- Carregar template do canal: {result['template']['name'] if result['template'] else 'Nenhum'}")
            st.write(f"- Carregar prompts associados: {len(result['channel_prompts']) + len(result['template_prompts'])} prompt(s)")
            st.write(f"- Carregar personagem: {result['character']['name'] if result['character'] else 'Nenhum'}")
            st.write(f"- Criar pasta outputs/<canal>/<timestamp>: `{result['output_dir']}`")

            for filename, path in result["files"].items():
                if not isinstance(path, Path):
                    continue
                if path.is_dir():
                    continue
                with st.expander(filename):
                    if path.suffix == ".mp4":
                        st.video(str(path))
                    elif path.suffix in {".mp3", ".wav", ".ogg"}:
                        st.audio(str(path))
                    elif path.suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                        st.image(str(path), use_container_width=True)
                    elif path.suffix == ".json":
                        st.json(json.loads(path.read_text(encoding="utf-8")))
                    else:
                        st.code(path.read_text(encoding="utf-8"), language="text")
        except ValidationError as exc:
            show_validation_error(exc)
        except Exception as exc:
            st.error(f"Falha na geração: {exc}")


def render_outputs() -> None:
    render_page_header("Outputs", "Histórico de execuções e arquivos gerados")
    runs = list_runs(50)
    if not runs:
        st.info("Nenhum output gerado ainda.")
        return

    render_table(
        [
            {
                "Run": r["id"],
                "Canal": r["channel_name"],
                "Título": r["video_title"],
                "Status": r["status"],
                "Criado em": r["created_at"],
                "Pasta": r["output_dir"],
            }
            for r in runs
        ],
        empty_message="Nenhum output gerado ainda.",
        caption=f"{len(runs)} execução(ões)",
    )

    selected_run_id = st.selectbox(
        "Visualizar execução",
        options=[r["id"] for r in runs],
        format_func=lambda x: f"Run #{x} - {next(r['video_title'] for r in runs if r['id'] == x)}",
    )
    run = next(r for r in runs if r["id"] == selected_run_id)
    run_dir = Path(run["output_dir"])
    st.caption(f"Diretório: {run_dir}")

    if run_dir.exists():
        images_dir = run_dir / "images"
        if images_dir.exists() and images_dir.is_dir():
            image_files = sorted([p for p in images_dir.iterdir() if p.is_file()])
            if image_files:
                st.subheader("Imagens geradas")
                selected_preview = st.selectbox(
                    "Preview de imagem",
                    options=image_files,
                    format_func=lambda p: p.name,
                    key=f"output_image_preview_{selected_run_id}",
                )
                st.image(str(selected_preview), use_container_width=True)
                with st.expander("Galeria de imagens (nomes)"):
                    st.write([p.name for p in image_files])

        files = sorted([p for p in run_dir.iterdir() if p.is_file()])
        for file_path in files:
            with st.expander(file_path.name):
                if file_path.suffix == ".mp4":
                    st.video(str(file_path))
                    continue
                if file_path.suffix in {".mp3", ".wav", ".ogg"}:
                    st.audio(str(file_path))
                    continue
                content = file_path.read_text(encoding="utf-8")
                if file_path.suffix == ".json":
                    st.json(json.loads(content))
                else:
                    st.code(content, language="text")
    else:
        st.warning("Pasta de output não encontrada no disco.")


def render_settings() -> None:
    render_page_header("Configurações", "Ambiente local e estado atual do banco")
    st.subheader("Ambiente local")
    st.write(f"Banco SQLite: `{DB_PATH}`")
    st.write(f"Diretório de outputs: `{OUTPUTS_DIR}`")
    st.write(f"Google Cloud Project: `{GOOGLE_CLOUD_PROJECT or 'Não configurado'}`")
    st.write(f"Google Cloud Location: `{GOOGLE_CLOUD_LOCATION or 'Não configurado'}`")
    st.write(f"ADC (GOOGLE_APPLICATION_CREDENTIALS): `{GOOGLE_APPLICATION_CREDENTIALS or 'Usando ADC da sessão / não definido'}`")
    st.write(f"Modelo Vertex padrão: `{VERTEX_MODEL}`")
    st.write(f"Modelo de roteiro (Vertex): `{VERTEX_SCRIPT_MODEL}`")
    st.write(f"Modelo de imagem (Vertex): `{VERTEX_IMAGE_MODEL}`")
    st.write(f"Modelo de vídeo (Vertex): `{VERTEX_VIDEO_MODEL}`")
    st.write(f"Modelo TTS padrão: `{TTS_MODEL_NAME}`")
    st.write(f"Voz TTS padrão: `{TTS_VOICE_NAME}` ({TTS_LANGUAGE_CODE})")
    st.write(
        f"Parâmetros TTS padrão: format={TTS_AUDIO_FORMAT} | rate={TTS_SPEAKING_RATE} | "
        f"pitch={TTS_PITCH} | gain_db={TTS_VOLUME_GAIN_DB} | sample_rate={TTS_SAMPLE_RATE_HZ}"
    )
    st.write(f"FFMPEG_BIN: `{FFMPEG_BIN}`")

    st.subheader("Observações")
    st.markdown(
        """
- Projeto preparado para futuras integrações com IA (OpenAI, geração de imagem, TTS etc.)
- A etapa de geração suporta Mock local e Vertex AI (com ADC / service account)
- Você pode configurar caminhos em `.env`
        """
    )

    st.subheader("Teste de conexão Vertex AI")
    st.caption("Valida autenticação, projeto, região e acesso ao modelo sem gerar outputs.")
    test_model = st.text_input(
        "Modelo para teste",
        value=VERTEX_MODEL,
        key="vertex_test_model",
        help="Use um modelo Gemini disponível no Vertex AI na região configurada.",
    ).strip() or VERTEX_MODEL
    if st.button("Testar conexão Vertex AI", type="primary", use_container_width=True):
        try:
            with st.spinner("Testando conexão com Vertex AI..."):
                test_result = test_vertex_connection(model_name=test_model)
            st.session_state["vertex_connection_status"] = "ok"
            st.session_state["vertex_connection_model"] = test_result.get("model", test_model)
            st.session_state["vertex_connection_latency_ms"] = test_result.get("latency_ms")
            st.session_state.pop("vertex_connection_error", None)
            st.success(f"Conexão OK ({test_result['model']}) em {test_result['latency_ms']} ms")
            st.json(test_result)
        except Exception as exc:
            st.session_state["vertex_connection_status"] = "error"
            st.session_state["vertex_connection_model"] = test_model
            st.session_state["vertex_connection_error"] = str(exc)
            st.session_state.pop("vertex_connection_latency_ms", None)
            st.error(f"Falha no teste de conexão Vertex AI: {exc}")
            st.caption(
                "Verifique ADC (`gcloud auth application-default login` ou service account), "
                "permissões e disponibilidade do modelo na região."
            )

    st.subheader("Resumo do banco")
    counts = {
        "channels": fetch_all("SELECT COUNT(*) AS total FROM channels")[0]["total"],
        "templates": fetch_all("SELECT COUNT(*) AS total FROM templates")[0]["total"],
        "characters": fetch_all("SELECT COUNT(*) AS total FROM characters")[0]["total"],
        "prompts": fetch_all("SELECT COUNT(*) AS total FROM prompts")[0]["total"],
        "runs": fetch_all("SELECT COUNT(*) AS total FROM runs")[0]["total"],
    }
    st.json(counts)


def main() -> None:
    inject_custom_css()
    with st.sidebar:
        st.title("Ray DarkAuto")
        st.caption("Dark Channel Factory")
        st.caption(f"DB: {DB_PATH.name}")
        page = st.radio("Navegação", MENU_ITEMS)

    if page == "Dashboard":
        render_dashboard()
    elif page == "Gerar Conteúdo":
        render_generate_content()
    elif page == "Canais":
        render_channels()
    elif page == "Templates":
        render_templates()
    elif page == "Personagens":
        render_characters()
    elif page == "Prompts":
        render_prompts()
    elif page == "Modelos":
        render_models()
    elif page == "Outputs":
        render_outputs()
    elif page == "Configurações":
        render_settings()


if __name__ == "__main__":
    main()
