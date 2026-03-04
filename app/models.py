from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


ChannelStatus = Literal["Ideia", "Teste", "Escala", "Pausado", "Abandonado"]
PromptType = Literal["roteiro", "imagem", "imagem_animada", "thumbnail", "titulo"]


class ChannelPayload(BaseModel):
    name: str = Field(min_length=1)
    language: str = Field(min_length=1)
    country: str = Field(min_length=1)
    niche: str = Field(min_length=1)
    subniche: str = ""
    platform: str = Field(min_length=1)
    creation_date: date
    status: ChannelStatus
    template_id: int | None = None
    default_character_id: int | None = None
    strategic_description: str = ""

    @field_validator("name", "language", "country", "niche", "platform")
    @classmethod
    def strip_required(cls, value: str) -> str:
        return value.strip()


class TemplatePayload(BaseModel):
    name: str = Field(min_length=1)
    niche: str = Field(min_length=1)
    content_strategy: str = ""
    visual_style: str = ""
    script_prompt_base: str = ""
    image_prompt_base: str = ""
    thumbnail_prompt: str = ""
    default_persona: str = ""
    extra_settings: str = ""

    @field_validator("name", "niche")
    @classmethod
    def strip_required(cls, value: str) -> str:
        return value.strip()


class CharacterPayload(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""
    visual_style: str = ""
    base_prompt: str = ""
    reference_image_path: str = ""
    tags: str = ""
    language: str = ""

    @field_validator("name")
    @classmethod
    def strip_required(cls, value: str) -> str:
        return value.strip()


class PromptPayload(BaseModel):
    name: str = Field(min_length=1)
    type: PromptType
    language: str = ""
    full_text: str = Field(min_length=1)
    channel_id: int | None = None
    template_id: int | None = None

    @field_validator("name", "full_text")
    @classmethod
    def strip_required(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_image_animation_scope(self) -> "PromptPayload":
        if self.type == "imagem_animada" and self.channel_id is None:
            raise ValueError(
                "Prompt do tipo 'imagem_animada' deve estar associado a um canal específico."
            )
        return self


class RunPayload(BaseModel):
    channel_id: int
    video_title: str = Field(min_length=1)

    @field_validator("video_title")
    @classmethod
    def strip_title(cls, value: str) -> str:
        return value.strip()
