"""
Character Builder - local build using prompt/voice results.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .model import CharacterConfig, Metadata, Personality, VoiceConfig
from .quality_filter import (
    AudioFilterConfig,
    AudioQuality,
    analyze_audio,
    character_adaptive_audio_config,
    get_audio_analysis_device,
    is_valid_japanese_text,
    quality_reason,
    score_audio,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioSelection:
    audio_path: Path
    text_jp: Optional[Path]
    text_zh: Optional[Path]
    quality: AudioQuality
    audio_issue: Optional[str]
    size_bytes: int
    text_score: float


def _normalize_name(value: str) -> str:
    cleaned = value.strip().replace("_", " ").replace("\u3000", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned.lower()


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[\\/]+", " ", value)
    cleaned = re.sub(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u9fff._ -]+", " ", cleaned)
    slug = _normalize_name(cleaned).replace(" ", "_").strip("._")
    return slug or "unknown"


def _hyphenated_filename_name(value: str) -> str:
    cleaned = value.strip().replace("_", " ").replace("\u3000", " ")
    cleaned = " ".join(cleaned.split())
    return re.sub(r"[^A-Za-z0-9]+", "-", cleaned).strip("-") or "unknown"


def _load_character_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return { _normalize_name(en): zh for zh, en in data.items() if isinstance(zh, str) and isinstance(en, str) }


def _iter_audio_files(voice_dir: Path) -> List[Path]:
    return sorted(
        [p for p in voice_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp3", ".wav"}],
        key=lambda p: p.stat().st_size,
        reverse=True,
    )


def _find_text_files(audio_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    stem = audio_path.stem
    jp = audio_path.with_name(f"{stem}_jp.txt")
    zh = audio_path.with_name(f"{stem}_zh.txt")
    return (jp if jp.exists() else None, zh if zh.exists() else None)


def _load_text(text_path: Path) -> str:
    return text_path.read_text(encoding="utf-8").strip()


def _reference_text_style_score(text: str) -> float:
    """
    Prefer neutral spoken references over highly emotional, stuttered, or acted lines.
    The text side is only a proxy, so this is a ranking penalty instead of a hard filter.
    """
    if not text:
        return 0.0

    score = 0.0
    exclamations = text.count("！") + text.count("!")
    questions = text.count("？") + text.count("?")
    ellipses = text.count("…") + text.count("...")

    score -= min(exclamations, 3) * 1.0
    score -= min(questions, 2) * 0.35
    score -= min(ellipses, 4) * 0.25

    # Stuttered kana separated by punctuation, e.g. お、お姉さま / い、い、い.
    stutter_matches = re.findall(r"([ぁ-んァ-ン])(?:[、,・]\1){1,}", text)
    score -= min(len(stutter_matches), 4) * 2.5

    emotional_patterns = [
        r"うう",
        r"うぅ",
        r"ふえ",
        r"うぇ",
        r"はう",
        r"はふ",
        r"ふふ",
        r"えへ",
        r"はわ",
        r"あう",
        r"泣",
        r"ドキドキ",
        r"いたずら",
        r"ハロウィン",
        r"ヴァンパイア",
        r"オバケ",
        r"コウモリ",
        r"衣装",
        r"お菓子",
        r"びゅーん",
    ]
    for pattern in emotional_patterns:
        if re.search(pattern, text):
            score -= 1.0

    neutral_patterns = [
        r"してたら",
        r"届いた",
        r"会った",
        r"見つけた",
        r"拾って",
        r"準備",
        r"読んで",
        r"走って",
        r"思った",
    ]
    for pattern in neutral_patterns:
        if re.search(pattern, text):
            score += 0.5

    if exclamations == 0 and len(stutter_matches) == 0 and not re.search(r"うう|うぅ|ふえ|うぇ|はう|あう", text):
        score += 1.0

    return score


def _evaluate_candidate(
    audio_path_str: str,
    text_jp_str: Optional[str],
    text_zh_str: Optional[str],
) -> Tuple[str, Optional[str], Optional[str], AudioQuality, int, float, Optional[str], List[Tuple[str, str]]]:
    audio_path = Path(audio_path_str)
    issues: List[Tuple[str, str]] = []
    text_score = 0.0

    penalties = {
        "humming_or_singing": 6.0,
        "repetitive_kana": 4.0,
        "too_many_ellipsis": 4.0,
        "too_many_symbols": 3.0,
        "too_short": 2.0,
        "too_long": 2.0,
        "no_kanji": 2.0,
    }

    if text_jp_str:
        try:
            jp_text = Path(text_jp_str).read_text(encoding="utf-8").strip()
        except Exception:
            jp_text = ""
            issues.append(("jp", "read_error"))
        if jp_text:
            valid, reason = is_valid_japanese_text(jp_text)
            if valid:
                text_score += 0.6
            else:
                text_score -= penalties.get(reason, 1.5)
                issues.append(("jp", reason))
            text_score += _reference_text_style_score(jp_text)
        else:
            text_score -= 0.5
            issues.append(("jp", "empty"))

    if text_zh_str:
        try:
            zh_text = Path(text_zh_str).read_text(encoding="utf-8").strip()
        except Exception:
            zh_text = ""
            issues.append(("zh", "read_error"))
        if zh_text:
            text_score += 0.2
        else:
            text_score -= 0.5
            issues.append(("zh", "empty"))

    quality, audio_reason = analyze_audio(audio_path_str)
    if audio_reason:
        issues.append(("audio", audio_reason))

    size_bytes = audio_path.stat().st_size
    return (audio_path_str, text_jp_str, text_zh_str, quality, size_bytes, text_score, audio_reason, issues)


def _iter_candidates(
    audio_paths: Iterable[Path],
) -> Iterable[Tuple[str, Optional[str], Optional[str]]]:
    for audio_path in audio_paths:
        text_jp, text_zh = _find_text_files(audio_path)
        yield (
            audio_path.as_posix(),
            text_jp.as_posix() if text_jp else None,
            text_zh.as_posix() if text_zh else None,
        )


def _select_best_audio(
    voice_dir: Path,
    max_candidates: Optional[int] = None,
    workers: int = 1,
) -> Optional[AudioSelection]:
    audio_files = _iter_audio_files(voice_dir)
    if not audio_files:
        return None

    config = AudioFilterConfig()
    candidates = audio_files if max_candidates is None else audio_files[:max_candidates]
    selections: List[AudioSelection] = []
    worker_count = max(1, int(workers or 1))
    analysis_device = get_audio_analysis_device()
    if analysis_device != "cpu" and worker_count > 1:
        logger.info(
            "Audio analysis device is %s, forcing workers=1 to avoid multi-process GPU contention.",
            analysis_device,
        )
        worker_count = 1
    candidate_args = list(_iter_candidates(candidates))

    def _handle_result(
        result: Tuple[str, Optional[str], Optional[str], AudioQuality, int, float, Optional[str], List[Tuple[str, str]]],
    ) -> None:
        audio_path_str, text_jp_str, text_zh_str, quality, size_bytes, text_score, audio_reason, issues = result
        display_path = audio_path_str
        for issue_type, reason in issues:
            if issue_type == "jp":
                logger.info('Flag "%s": jp text issue (%s)', display_path, reason)
            elif issue_type == "zh":
                logger.info('Flag "%s": zh text issue (%s)', display_path, reason)
            elif issue_type == "audio":
                # Audio issue is re-evaluated with per-character adaptive thresholds later.
                continue
            else:
                logger.info('Flag "%s": issue (%s)', display_path, reason)

        selections.append(
            AudioSelection(
                audio_path=Path(audio_path_str),
                text_jp=Path(text_jp_str) if text_jp_str else None,
                text_zh=Path(text_zh_str) if text_zh_str else None,
                quality=quality,
                audio_issue=audio_reason,
                size_bytes=size_bytes,
                text_score=text_score,
            )
        )

    if worker_count > 1 and len(candidate_args) > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_evaluate_candidate, *args): args[0] for args in candidate_args
            }
            for future in as_completed(future_map):
                try:
                    _handle_result(future.result())
                except Exception as exc:
                    logger.warning('Flag "%s": analysis exception (%s)', future_map[future], exc)
    else:
        for args in candidate_args:
            _handle_result(_evaluate_candidate(*args))

    if not selections:
        return None

    pitch_backend = "torch" if analysis_device != "cpu" else "librosa"
    config = character_adaptive_audio_config(
        [s.quality for s in selections],
        config=config,
        pitch_backend=pitch_backend,
    )
    min_f0_std = config.torch_min_f0_std if pitch_backend == "torch" else config.min_f0_std
    max_pitch_range = config.torch_max_pitch_range if pitch_backend == "torch" else config.max_pitch_range
    logger.info(
        "Adaptive thresholds for %s (%s): min_centroid=%.1f min_f0_std=%.1f max_pitch_range=%.1f",
        voice_dir.name,
        pitch_backend,
        config.min_centroid,
        min_f0_std,
        max_pitch_range,
    )

    adjusted: List[AudioSelection] = []
    for selection in selections:
        reason = quality_reason(selection.quality, config=config, pitch_backend=pitch_backend)
        if reason:
            logger.info('Flag "%s": audio issue (%s)', selection.audio_path, reason)
        adjusted.append(
            AudioSelection(
                audio_path=selection.audio_path,
                text_jp=selection.text_jp,
                text_zh=selection.text_zh,
                quality=selection.quality,
                audio_issue=reason,
                size_bytes=selection.size_bytes,
                text_score=selection.text_score,
            )
        )
    selections = adjusted

    def _score(selection: AudioSelection) -> float:
        size_bonus = min(selection.size_bytes / 1_000_000.0, 5.0) * 0.1
        return score_audio(selection.quality, config, pitch_backend=pitch_backend) + selection.text_score + size_bonus

    valid_audio = [s for s in selections if s.audio_issue is None]
    pool = valid_audio if valid_audio else selections
    pool.sort(key=_score, reverse=True)

    selected = pool[0]
    if selected.audio_issue:
        logger.warning(
            "No fully valid clone audio found in %s, fallback selected %s (%s)",
            voice_dir,
            selected.audio_path,
            selected.audio_issue,
        )
    logger.info(
        'Selected "%s" as clone reference (score basis: quality + text + size)',
        selected.audio_path,
    )
    return selected


def _load_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


def _copy_prompt(prompt_path: Path, character_dir: Path) -> Path:
    target = character_dir / "prompt.md"
    shutil.copy2(prompt_path, target)
    return target


def _copy_audio(selection: AudioSelection, character_dir: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    audio_suffix = selection.audio_path.suffix.lower()
    audio_target = character_dir / f"reference{audio_suffix}"
    shutil.copy2(selection.audio_path, audio_target)

    text_jp_target = None
    if selection.text_jp:
        text_jp_target = character_dir / "reference_jp.txt"
        shutil.copy2(selection.text_jp, text_jp_target)

    text_zh_target = None
    if selection.text_zh:
        text_zh_target = character_dir / "reference_zh.txt"
        shutil.copy2(selection.text_zh, text_zh_target)

    return audio_target, text_jp_target, text_zh_target


def _build_character_config(
    character_dir: Path,
    prompt_text: str,
    name_info: Dict[str, str],
    audio_target: Optional[Path],
    text_jp_target: Optional[Path],
    text_zh_target: Optional[Path],
    quality: Optional[AudioQuality],
    no_voice: bool = False,
) -> CharacterConfig:
    ref_text_path = None
    if text_jp_target:
        ref_text_path = text_jp_target.name
    elif text_zh_target:
        ref_text_path = text_zh_target.name

    sample_rate = quality.sample_rate if quality and quality.sample_rate else 22050
    ref_audio_path = audio_target.name if audio_target else None

    return CharacterConfig(
        id=f"uma_{_slugify(name_info['name_en'])}",
        name_zh=name_info.get("name_zh") or name_info["name_en"],
        name_en=name_info["name_en"],
        name_jp=name_info.get("name_jp") or name_info["name_en"],
        version="1.0",
        created_at=datetime.now(),
        last_updated=datetime.now(),
        system_prompt=prompt_text,
        personality=Personality(),
        voice_config=VoiceConfig(
            model="IndexTTS2",
            ref_audio_path=ref_audio_path,
            ref_text_path=ref_text_path,
            no_voice="true" if no_voice else "false",
            language_code="ja-JP",
            sample_rate=sample_rate,
        ),
        metadata=Metadata(
            cv=name_info.get("cv"),
            birthday=name_info.get("birthday"),
            height=name_info.get("height"),
            source_wiki=name_info.get("source_wiki"),
            retrieved_at=datetime.now(),
        ),
        character_dir=character_dir,
    )


def _cleanup_reference_assets(character_dir: Path) -> None:
    for filename in (
        "reference.mp3",
        "reference.wav",
        "reference.flac",
        "reference.m4a",
        "reference.ogg",
        "reference_jp.txt",
        "reference_zh.txt",
    ):
        path = character_dir / filename
        if path.exists() and path.is_file():
            path.unlink()


def _save_character_config(character: CharacterConfig) -> None:
    config_path = character.character_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            character.model_dump(mode="json", exclude={"character_dir"}),
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _iter_image_dirs(image_dir: Path) -> Dict[str, Path]:
    aliases = {
        "result-images": "results-images",
        "results-images": "result-images",
    }
    if not image_dir.exists() and image_dir.name in aliases:
        fallback = image_dir.with_name(aliases[image_dir.name])
        if fallback.exists():
            logger.warning("Image directory %s not found, using %s instead.", image_dir, fallback)
            image_dir = fallback

    if not image_dir.exists():
        logger.warning("Image directory not found: %s. Character images will be skipped.", image_dir)
        return {}
    return {_normalize_name(p.name): p for p in image_dir.iterdir() if p.is_dir()}


def _select_character_image(source_dir: Path, prefix: str) -> Optional[Path]:
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    prefix_lower = prefix.lower()
    candidates = [
        path
        for path in source_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in allowed_suffixes
        and path.name.lower().startswith(f"{prefix_lower}_")
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (path.stat().st_size, path.name), reverse=True)[0]


def package_character_images(
    name_en: str,
    character_dir: Path,
    image_dirs: Dict[str, Path],
    dry_run: bool = False,
) -> List[Path]:
    source_dir = image_dirs.get(_normalize_name(name_en))
    if not source_dir:
        logger.warning("Skip images for %s: image directory not found", name_en)
        return []

    filename_name = _hyphenated_filename_name(name_en)
    copied: List[Path] = []
    for source_prefix, target_prefix in (("Jsf", "JSF"), ("Zf", "ZF")):
        source = _select_character_image(source_dir, source_prefix)
        if not source:
            logger.warning(
                "Skip %s image for %s: no %s_* file in %s",
                target_prefix,
                name_en,
                source_prefix,
                source_dir,
            )
            continue

        target = character_dir / f"{target_prefix}_{filename_name}{source.suffix.lower()}"
        if dry_run:
            logger.info('Would copy "%s" -> "%s"', source, target)
        else:
            character_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            logger.info('Copied "%s" -> "%s"', source, target)
        copied.append(target)
    return copied


def _split_prompt_name_aliases(value: str) -> Tuple[Optional[str], Optional[str]]:
    aliases = [part.strip() for part in re.split(r"[/／|｜]", value) if part.strip()]
    if not aliases:
        return None, None

    english = None
    japanese = None
    for alias in aliases:
        if english is None and re.search(r"[A-Za-z]", alias):
            english = alias
        if japanese is None and re.search(r"[\u3040-\u30ff]", alias):
            japanese = alias

    return english or aliases[0], japanese


def _extract_names_from_prompt(prompt_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Match title patterns like: # 角色扮演提示词：爱慕织姬（Admire Vega）
    patterns = [
        re.compile(r"角色扮演提示词[:：]\s*([^\(\n（]+?)\s*[（(]\s*([^\)）\n]+?)\s*[)）]"),
        re.compile(r"角色提示词[:：]\s*([^\(\n（]+?)\s*[（(]\s*([^\)）\n]+?)\s*[)）]"),
    ]
    for line in prompt_text.splitlines()[:24]:
        candidate = line.strip().lstrip("#").strip()
        if not candidate:
            continue
        for pattern in patterns:
            match = pattern.search(candidate)
            if match:
                name_zh = match.group(1).strip().strip("*")
                name_en, name_jp = _split_prompt_name_aliases(match.group(2).strip().strip("*"))
                if name_zh or name_en or name_jp:
                    return name_zh or None, name_en or None, name_jp or None
    return None, None, None


def _resolve_name_info(prompt_en: str, en_to_zh: Dict[str, str], prompt_text: str) -> Dict[str, str]:
    normalized = _normalize_name(prompt_en)
    prompt_name_zh, prompt_name_en, prompt_name_jp = _extract_names_from_prompt(prompt_text)
    name_zh = en_to_zh.get(normalized) or prompt_name_zh or prompt_en
    name_en = prompt_name_en or prompt_en
    return {
        "name_en": name_en,
        "name_zh": name_zh,
        "name_jp": prompt_name_jp or name_en,
        "source_wiki": f"https://wiki.biligame.com/umamusume/{name_zh}",
    }


def build_characters(
    prompt_dir: Path,
    voice_dir: Path,
    image_dir: Path,
    output_dir: Path,
    character_map_path: Path,
    workers: int = 1,
    dry_run: bool = False,
    target_characters: Optional[Iterable[str]] = None,
    max_candidates: Optional[int] = None,
    include_images: bool = False,
) -> List[str]:
    prompt_files = sorted([p for p in prompt_dir.iterdir() if p.is_file() and p.suffix.lower() in {".md", ".txt"}])
    if voice_dir.exists():
        voice_dirs = { _normalize_name(p.name): p for p in voice_dir.iterdir() if p.is_dir() }
    else:
        logger.warning(
            "Voice directory not found: %s. Characters will still be built with no_voice=true.",
            voice_dir,
        )
        voice_dirs = {}

    image_dirs = _iter_image_dirs(image_dir) if include_images else {}
    en_to_zh = _load_character_map(character_map_path)
    target_set: Optional[Set[str]] = None
    if target_characters:
        target_set = {_normalize_name(name) for name in target_characters if name and name.strip()}

    built: List[str] = []

    for prompt_path in prompt_files:
        prompt_en = prompt_path.stem.replace("_", " ").strip()
        normalized = _normalize_name(prompt_en)
        if target_set and normalized not in target_set:
            continue
        voice_path = voice_dirs.get(normalized)

        prompt_text = _load_prompt(prompt_path)
        if not prompt_text:
            logger.warning("Skip %s: prompt file is empty", prompt_en)
            continue

        selection: Optional[AudioSelection] = None
        no_voice = False
        if not voice_path:
            no_voice = True
            logger.warning("Build %s without voice: voice directory not found", prompt_en)
        else:
            selection = _select_best_audio(voice_path, workers=workers, max_candidates=max_candidates)
            if not selection:
                no_voice = True
                logger.warning("Build %s without voice: no valid audio after filtering in %s", prompt_en, voice_path)

        name_info = _resolve_name_info(prompt_en, en_to_zh, prompt_text)
        character_slug = _slugify(name_info["name_en"])
        character_dir = output_dir / character_slug

        logger.info("Building character %s -> %s", prompt_en, character_dir)

        if dry_run:
            if include_images:
                package_character_images(name_info["name_en"], character_dir, image_dirs, dry_run=True)
            built.append(character_slug)
            continue

        character_dir.mkdir(parents=True, exist_ok=True)

        _copy_prompt(prompt_path, character_dir)
        if selection:
            audio_target, text_jp_target, text_zh_target = _copy_audio(selection, character_dir)
        else:
            _cleanup_reference_assets(character_dir)
            audio_target, text_jp_target, text_zh_target = (None, None, None)

        character_config = _build_character_config(
            character_dir=character_dir,
            prompt_text=prompt_text,
            name_info=name_info,
            audio_target=audio_target,
            text_jp_target=text_jp_target,
            text_zh_target=text_zh_target,
            quality=selection.quality if selection else None,
            no_voice=no_voice,
        )
        _save_character_config(character_config)
        if include_images:
            package_character_images(name_info["name_en"], character_dir, image_dirs)

        built.append(character_slug)

    return built


def package_existing_character_images(
    image_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    target_characters: Optional[Iterable[str]] = None,
) -> List[str]:
    if not output_dir.exists():
        logger.warning("Characters directory not found: %s", output_dir)
        return []

    image_dirs = _iter_image_dirs(image_dir)
    target_set: Optional[Set[str]] = None
    if target_characters:
        target_set = {_normalize_name(name) for name in target_characters if name and name.strip()}

    processed: List[str] = []
    for character_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        config_path = character_dir / "config.json"
        if not config_path.exists():
            continue
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skip %s: failed to read config.json (%s)", character_dir, exc)
            continue

        name_en = str(data.get("name_en") or character_dir.name.replace("_", " ")).strip()
        if (
            target_set
            and _normalize_name(name_en) not in target_set
            and _normalize_name(character_dir.name) not in target_set
        ):
            continue

        copied = package_character_images(name_en, character_dir, image_dirs, dry_run=dry_run)
        if copied:
            processed.append(character_dir.name)
    return processed


class CharacterBuilder:
    """构建角色配置（基于本地 prompt + voice 结果）"""

    def __init__(
        self,
        prompt_dir: str = "result-prompts",
        voice_dir: str = "result-voices",
        image_dir: str = "result-images",
        characters_dir: str = "characters",
        character_map_path: str = "umamusume_characters.json",
    ) -> None:
        self.prompt_dir = Path(prompt_dir)
        self.voice_dir = Path(voice_dir)
        self.image_dir = Path(image_dir)
        self.characters_dir = Path(characters_dir)
        self.character_map_path = Path(character_map_path)

    def build_all(
        self,
        dry_run: bool = False,
        workers: int = 1,
        target_characters: Optional[Iterable[str]] = None,
        max_candidates: Optional[int] = None,
        include_images: bool = False,
    ) -> List[str]:
        return build_characters(
            prompt_dir=self.prompt_dir,
            voice_dir=self.voice_dir,
            image_dir=self.image_dir,
            output_dir=self.characters_dir,
            character_map_path=self.character_map_path,
            workers=workers,
            dry_run=dry_run,
            target_characters=target_characters,
            max_candidates=max_candidates,
            include_images=include_images,
        )

    def package_images(
        self,
        dry_run: bool = False,
        target_characters: Optional[Iterable[str]] = None,
    ) -> List[str]:
        return package_existing_character_images(
            image_dir=self.image_dir,
            output_dir=self.characters_dir,
            dry_run=dry_run,
            target_characters=target_characters,
        )

    async def build_character(
        self,
        character_name: str,
        workers: int = 1,
        max_candidates: Optional[int] = None,
        include_images: bool = False,
    ) -> CharacterConfig:
        self.build_all(
            dry_run=False,
            workers=workers,
            target_characters=[character_name],
            max_candidates=max_candidates,
            include_images=include_images,
        )
        config_path = self.characters_dir / _slugify(character_name) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Character config not found: {config_path}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        character = CharacterConfig(**data)
        character.character_dir = config_path.parent
        return character
