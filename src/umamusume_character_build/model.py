"""
Local character config models used by character builder.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Pronouns(BaseModel):
    self: str = "我"
    user: str = "训练员桑"


class Personality(BaseModel):
    traits: List[str] = Field(default_factory=list)
    speaking_style: str = ""
    pronouns: Pronouns = Field(default_factory=Pronouns)
    catchphrases: List[str] = Field(default_factory=list)


class VoiceConfig(BaseModel):
    model: str = "IndexTTS2"
    ref_audio_path: Optional[str] = None
    ref_text_path: Optional[str] = None
    no_voice: str = "false"
    language_code: str = "ja-JP"
    sample_rate: int = 22050


class Metadata(BaseModel):
    cv: Optional[str] = None
    birthday: Optional[str] = None
    height: Optional[str] = None
    source_wiki: Optional[str] = None
    retrieved_at: datetime = Field(default_factory=datetime.now)


class CharacterConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name_zh: str
    name_en: str
    name_jp: str
    version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    system_prompt: str
    personality: Personality = Field(default_factory=Personality)
    voice_config: VoiceConfig
    metadata: Metadata = Field(default_factory=Metadata)
    character_dir: Path = Field(default_factory=Path, exclude=True)
