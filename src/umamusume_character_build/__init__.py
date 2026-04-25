"""
Character Builder - 角色构建器模块
使用本地 prompt/voice 结果构建角色配置。
"""

from .character_builder import CharacterBuilder, build_characters, package_existing_character_images
from .model import CharacterConfig, Metadata, Personality, VoiceConfig

__all__ = [
    "CharacterBuilder",
    "CharacterConfig",
    "Metadata",
    "Personality",
    "VoiceConfig",
    "build_characters",
    "package_existing_character_images",
]
