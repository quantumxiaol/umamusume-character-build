"""
Quality filters for prompt text and voice samples.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextFilterConfig:
    min_length: int = 5
    max_length: int = 50
    max_symbol_ratio: float = 0.3
    max_ellipsis_ratio: float = 0.12


@dataclass(frozen=True)
class AudioFilterConfig:
    min_duration: float = 3.0
    max_duration: float = 35.0
    min_centroid: float = 1200.0
    min_f0_std: float = 20.0
    max_pitch_range: float = 340.0
    max_silence_ratio: float = 0.4
    max_silence_sec: float = 2.0
    top_db: float = 30.0


@dataclass(frozen=True)
class AudioQuality:
    duration: Optional[float]
    effective_duration: Optional[float]
    sample_rate: Optional[int]
    spectral_centroid: Optional[float]
    f0_std: Optional[float]
    pitch_range: Optional[float]
    silence_ratio: Optional[float]
    max_silence_sec: Optional[float]
    nasal_ratio: Optional[float]


_BAD_TEXT_PATTERNS = [
    r"ふんふん",
    r"ふふ",
    r"ラララ",
    r"ルルル",
    r"はぁ",
    r"はあ",
    r"はっ",
    r"すぅ",
    r"ふー",
    r"ぁっ",
    r"んっ",
    r"えーと",
    r"あの",
]


def _empty_quality() -> AudioQuality:
    return AudioQuality(None, None, None, None, None, None, None, None, None)


@lru_cache(maxsize=1)
def _requested_device() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except Exception:
        pass

    raw = (
        os.getenv("device")
        or os.getenv("DEVICE")
        or os.getenv("audio_device")
        or os.getenv("AUDIO_DEVICE")
        or "cpu"
    )
    return str(raw).strip().lower() or "cpu"


@lru_cache(maxsize=1)
def _resolve_audio_device() -> str:
    requested = _requested_device()
    if requested.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return requested
        except Exception:
            pass
        logger.warning('Requested audio device "%s" not available, fallback to cpu', requested)
        return "cpu"

    if requested.startswith("mps"):
        try:
            import torch

            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        logger.warning('Requested audio device "%s" not available, fallback to cpu', requested)
        return "cpu"

    if requested in {"cpu", "auto"}:
        return "cpu"

    logger.warning('Unknown audio device "%s", fallback to cpu', requested)
    return "cpu"


def get_audio_analysis_device() -> str:
    """
    Return effective device used by analyze_audio:
    - "cpu"
    - "mps"
    - "cuda" / "cuda:N"
    """
    return _resolve_audio_device()


def is_valid_japanese_text(text: str, config: TextFilterConfig | None = None) -> Tuple[bool, str]:
    cfg = config or TextFilterConfig()
    cleaned = text.strip()
    if not cleaned:
        return False, "empty"

    if len(cleaned) < cfg.min_length:
        return False, "too_short"

    if len(cleaned) > cfg.max_length:
        return False, "too_long"

    noise_chars = re.findall(r"[…—～\.。、]", cleaned)
    if cleaned and (len(noise_chars) / len(cleaned)) > cfg.max_symbol_ratio:
        return False, "too_many_symbols"

    ellipsis_count = cleaned.count("…") + cleaned.count("...")
    if cleaned and ellipsis_count / len(cleaned) > cfg.max_ellipsis_ratio:
        return False, "too_many_ellipsis"

    for pattern in _BAD_TEXT_PATTERNS:
        if re.search(pattern, cleaned):
            if "ふん" in pattern or "ララ" in pattern or "ルル" in pattern:
                return False, "humming_or_singing"

    if re.search(r"([ぁ-んァ-ン])\1{2,}", cleaned):
        return False, "repetitive_kana"

    has_kanji = bool(re.search(r"[\u4e00-\u9fff]", cleaned))
    if not has_kanji and len(cleaned) > 10:
        return False, "no_kanji"

    return True, "ok"


def _quality_reason(quality: AudioQuality, cfg: AudioFilterConfig) -> Optional[str]:
    if quality.effective_duration is not None:
        if quality.effective_duration < cfg.min_duration or quality.effective_duration > cfg.max_duration:
            return "duration_out_of_range"

    if quality.silence_ratio is not None and quality.silence_ratio > cfg.max_silence_ratio:
        return "too_much_silence"

    if quality.max_silence_sec is not None and quality.max_silence_sec > cfg.max_silence_sec:
        return "long_silence_gap"

    if quality.spectral_centroid is not None and quality.spectral_centroid < cfg.min_centroid:
        return "too_muddy"

    if quality.f0_std is not None and quality.f0_std < cfg.min_f0_std:
        return "monotone_or_humming"

    if quality.pitch_range is not None and quality.pitch_range > cfg.max_pitch_range:
        return "singing_pitch_range"

    return None


def _silence_stats(samples, sample_rate: int, top_db: float) -> Tuple[float, float, float]:
    import librosa
    import numpy as np

    total_duration = float(samples.shape[0] / sample_rate) if sample_rate else 0.0
    if total_duration <= 0.0:
        return 0.0, 1.0, 0.0

    intervals = librosa.effects.split(samples, top_db=top_db)
    if intervals.size == 0:
        return 0.0, 1.0, total_duration

    voiced = np.sum((intervals[:, 1] - intervals[:, 0]) / sample_rate)
    silence_ratio = max(0.0, min(1.0, 1.0 - voiced / total_duration))

    max_gap = intervals[0][0] / sample_rate
    for (start, end), (next_start, _next_end) in zip(intervals[:-1], intervals[1:]):
        gap = (next_start - end) / sample_rate
        if gap > max_gap:
            max_gap = gap
    tail_gap = (samples.shape[0] - intervals[-1][1]) / sample_rate
    if tail_gap > max_gap:
        max_gap = tail_gap

    return float(voiced), float(silence_ratio), float(max_gap)


def _spectral_centroid(samples, sample_rate: int) -> Optional[float]:
    import librosa
    import numpy as np

    if samples.size == 0:
        return None
    centroid = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)
    if centroid.size == 0:
        return None
    return float(np.mean(centroid))


def _pitch_stats(samples, sample_rate: int) -> Tuple[Optional[float], Optional[float]]:
    import librosa
    import numpy as np

    if samples.size == 0:
        return None, None

    f0, voiced_flag, _voiced_prob = librosa.pyin(
        samples,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    if f0 is None or voiced_flag is None:
        return None, None
    valid_f0 = f0[voiced_flag]
    if valid_f0.size < 5:
        return None, None
    p10 = np.quantile(valid_f0, 0.1)
    p90 = np.quantile(valid_f0, 0.9)
    pitch_range = float(p90 - p10)
    pitch_std = float(np.std(valid_f0))
    return pitch_std, pitch_range


def _nasal_ratio(samples, sample_rate: int) -> Optional[float]:
    import librosa
    import numpy as np

    if samples.size == 0:
        return None

    max_samples = min(samples.shape[0], int(sample_rate * 12))
    short_samples = samples[:max_samples]
    stft = librosa.stft(short_samples, n_fft=2048, hop_length=512)
    power = np.abs(stft) ** 2
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)

    def band_energy(low_hz: float, high_hz: float) -> float:
        mask = (freqs >= low_hz) & (freqs < high_hz)
        if not np.any(mask):
            return 0.0
        return float(np.sum(power[mask]))

    low_band = band_energy(200.0, 800.0)
    nasal_band = band_energy(1000.0, 2500.0)
    if low_band <= 0:
        return None
    return nasal_band / low_band


def _analyze_audio_cpu(
    audio_path: str,
    cfg: AudioFilterConfig,
) -> Tuple[AudioQuality, Optional[str]]:
    try:
        import librosa

        samples, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        if samples.size == 0:
            return _empty_quality(), "empty_audio"

        duration = float(samples.shape[0] / sample_rate)
        voiced_duration, silence_ratio, max_silence_sec = _silence_stats(samples, sample_rate, cfg.top_db)
        effective_duration = float(voiced_duration)
        trimmed, _ = librosa.effects.trim(samples, top_db=cfg.top_db)

        quality = AudioQuality(
            duration=duration,
            effective_duration=effective_duration,
            sample_rate=sample_rate,
            spectral_centroid=_spectral_centroid(trimmed, sample_rate),
            f0_std=None,
            pitch_range=None,
            silence_ratio=silence_ratio,
            max_silence_sec=max_silence_sec,
            nasal_ratio=_nasal_ratio(trimmed, sample_rate),
        )
        pitch_std, pitch_range = _pitch_stats(trimmed, sample_rate)
        quality = AudioQuality(
            duration=quality.duration,
            effective_duration=quality.effective_duration,
            sample_rate=quality.sample_rate,
            spectral_centroid=quality.spectral_centroid,
            f0_std=pitch_std,
            pitch_range=pitch_range,
            silence_ratio=quality.silence_ratio,
            max_silence_sec=quality.max_silence_sec,
            nasal_ratio=quality.nasal_ratio,
        )
        return quality, _quality_reason(quality, cfg)
    except Exception:
        return _empty_quality(), "analysis_failed"


def _trim_by_amplitude_torch(samples, top_db: float):
    import torch

    if samples.numel() == 0:
        return samples
    peak = torch.max(torch.abs(samples))
    peak_value = float(peak.detach().cpu().item())
    if peak_value <= 0:
        return samples[:0]

    amp_threshold = peak * (10.0 ** (-top_db / 20.0))
    active_idx = torch.nonzero(torch.abs(samples) >= amp_threshold, as_tuple=False).flatten()
    if active_idx.numel() == 0:
        return samples[:0]
    start = int(active_idx[0].item())
    end = int(active_idx[-1].item()) + 1
    return samples[start:end]


def _intervals_from_voiced_mask(mask, frame_length: int, hop_length: int, total_samples: int):
    intervals = []
    start_idx = None
    mask_list = mask.detach().to("cpu").tolist()
    for idx, flag in enumerate(mask_list):
        if flag and start_idx is None:
            start_idx = idx
        elif (not flag) and start_idx is not None:
            end_idx = idx - 1
            start = start_idx * hop_length
            end = min(total_samples, end_idx * hop_length + frame_length)
            if end > start:
                intervals.append((start, end))
            start_idx = None

    if start_idx is not None:
        end_idx = len(mask_list) - 1
        start = start_idx * hop_length
        end = min(total_samples, end_idx * hop_length + frame_length)
        if end > start:
            intervals.append((start, end))

    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _silence_stats_torch(samples, sample_rate: int, top_db: float) -> Tuple[float, float, float]:
    import torch
    import torch.nn.functional as F

    total_samples = int(samples.shape[0])
    total_duration = float(total_samples / sample_rate) if sample_rate else 0.0
    if total_duration <= 0.0:
        return 0.0, 1.0, 0.0

    frame_length = 2048
    hop_length = 512
    work = samples
    if total_samples < frame_length:
        work = F.pad(samples, (0, frame_length - total_samples))
    frames = work.unfold(0, frame_length, hop_length)
    if frames.numel() == 0:
        return 0.0, 1.0, total_duration

    rms = torch.sqrt(torch.clamp(torch.mean(frames * frames, dim=-1), min=1e-12))
    db = 20.0 * torch.log10(torch.clamp(rms, min=1e-12))
    threshold = torch.max(db) - top_db
    voiced_mask = db > threshold

    intervals = _intervals_from_voiced_mask(voiced_mask, frame_length, hop_length, total_samples)
    if not intervals:
        return 0.0, 1.0, total_duration

    voiced_samples = sum(end - start for start, end in intervals)
    voiced_duration = float(voiced_samples / sample_rate)
    silence_ratio = max(0.0, min(1.0, 1.0 - voiced_duration / total_duration))

    max_gap = intervals[0][0] / sample_rate
    for (_start, end), (next_start, _next_end) in zip(intervals[:-1], intervals[1:]):
        gap = (next_start - end) / sample_rate
        if gap > max_gap:
            max_gap = gap
    tail_gap = (total_samples - intervals[-1][1]) / sample_rate
    if tail_gap > max_gap:
        max_gap = tail_gap

    return voiced_duration, silence_ratio, float(max_gap)


def _spectral_centroid_torch(samples, sample_rate: int) -> Optional[float]:
    import torch

    if samples.numel() == 0:
        return None

    n_fft = 2048
    hop_length = 512
    work = samples
    if work.shape[0] < n_fft:
        work = torch.nn.functional.pad(work, (0, n_fft - work.shape[0]))
    stft = torch.stft(work, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    power = stft.abs() ** 2
    if power.numel() == 0:
        return None
    denom = torch.sum(power)
    if float(denom.detach().cpu().item()) <= 0.0:
        return None
    freqs = torch.linspace(0.0, sample_rate / 2.0, power.shape[0], device=power.device, dtype=power.dtype)
    centroid = torch.sum(power * freqs.unsqueeze(1)) / denom
    return float(centroid.detach().cpu().item())


def _pitch_stats_torch(samples, sample_rate: int) -> Tuple[Optional[float], Optional[float]]:
    import torch

    if samples.numel() == 0:
        return None, None

    frame_length = max(512, int(sample_rate * 0.04))
    hop_length = max(160, int(sample_rate * 0.01))
    if samples.shape[0] < frame_length:
        return None, None

    frames = samples.unfold(0, frame_length, hop_length)
    if frames.numel() == 0:
        return None, None

    window = torch.hann_window(frame_length, device=samples.device, dtype=samples.dtype)
    frames = frames * window.unsqueeze(0)

    fft_size = 1
    while fft_size < frame_length * 2:
        fft_size *= 2

    spectrum = torch.fft.rfft(frames, n=fft_size, dim=1)
    power = torch.abs(spectrum) ** 2
    autocorr = torch.fft.irfft(power, n=fft_size, dim=1)[:, :frame_length]

    min_lag = max(1, int(sample_rate / 2093.0))
    max_lag = min(frame_length - 1, int(sample_rate / 65.4))
    if max_lag <= min_lag:
        return None, None

    candidate = autocorr[:, min_lag : max_lag + 1]
    peak_values, peak_indices = torch.max(candidate, dim=1)
    base_energy = torch.clamp(autocorr[:, 0], min=1e-9)
    confidence = peak_values / base_energy
    valid_mask = confidence > 0.2
    if int(valid_mask.sum().detach().cpu().item()) < 5:
        return None, None

    lags = (peak_indices + min_lag).to(dtype=torch.float32)
    f0 = sample_rate / torch.clamp(lags, min=1.0)
    valid = f0[valid_mask]
    if valid.numel() < 5:
        return None, None

    p10 = torch.quantile(valid, 0.1)
    p90 = torch.quantile(valid, 0.9)
    pitch_range = p90 - p10
    pitch_std = torch.std(valid, unbiased=False)
    return float(pitch_std.detach().cpu().item()), float(pitch_range.detach().cpu().item())


def _nasal_ratio_torch(samples, sample_rate: int) -> Optional[float]:
    import torch

    if samples.numel() == 0:
        return None

    max_samples = min(int(samples.shape[0]), int(sample_rate * 12))
    if max_samples <= 0:
        return None

    short_samples = samples[:max_samples]
    n_fft = 2048
    hop_length = 512
    if short_samples.shape[0] < n_fft:
        short_samples = torch.nn.functional.pad(short_samples, (0, n_fft - short_samples.shape[0]))
    stft = torch.stft(short_samples, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    power = stft.abs() ** 2
    freqs = torch.linspace(0.0, sample_rate / 2.0, power.shape[0], device=power.device, dtype=power.dtype)

    def band_energy(low_hz: float, high_hz: float) -> float:
        mask = (freqs >= low_hz) & (freqs < high_hz)
        if not bool(torch.any(mask).detach().cpu().item()):
            return 0.0
        return float(torch.sum(power[mask]).detach().cpu().item())

    low_band = band_energy(200.0, 800.0)
    nasal_band = band_energy(1000.0, 2500.0)
    if low_band <= 0.0:
        return None
    return nasal_band / low_band


def _analyze_audio_accelerated(
    audio_path: str,
    cfg: AudioFilterConfig,
    device: str,
) -> Tuple[AudioQuality, Optional[str]]:
    try:
        import librosa
        import torch

        samples_np, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        if samples_np.size == 0:
            return _empty_quality(), "empty_audio"

        samples = torch.as_tensor(samples_np, dtype=torch.float32, device=torch.device(device))

        duration = float(samples.shape[0] / sample_rate)
        voiced_duration, silence_ratio, max_silence_sec = _silence_stats_torch(samples, sample_rate, cfg.top_db)
        effective_duration = float(voiced_duration)
        trimmed_np, _ = librosa.effects.trim(samples_np, top_db=cfg.top_db)

        # Keep textural/acoustic features identical to legacy CPU path for stable scoring.
        centroid = _spectral_centroid(trimmed_np, sample_rate)
        pitch_std, pitch_range = _pitch_stats(trimmed_np, sample_rate)
        nasal_ratio = _nasal_ratio(trimmed_np, sample_rate)

        quality = AudioQuality(
            duration=duration,
            effective_duration=effective_duration,
            sample_rate=int(sample_rate),
            spectral_centroid=centroid,
            f0_std=pitch_std,
            pitch_range=pitch_range,
            silence_ratio=silence_ratio,
            max_silence_sec=max_silence_sec,
            nasal_ratio=nasal_ratio,
        )
        return quality, _quality_reason(quality, cfg)
    except Exception as exc:
        logger.warning(
            'Accelerated audio analysis failed on "%s" with device=%s (%s), fallback to cpu',
            audio_path,
            device,
            exc,
        )
        return _analyze_audio_cpu(audio_path, cfg)


def analyze_audio(
    audio_path: str,
    config: AudioFilterConfig | None = None,
) -> Tuple[AudioQuality, Optional[str]]:
    cfg = config or AudioFilterConfig()
    device = get_audio_analysis_device()
    if device == "cpu":
        return _analyze_audio_cpu(audio_path, cfg)
    return _analyze_audio_accelerated(audio_path, cfg, device)


def score_audio(quality: AudioQuality, config: AudioFilterConfig | None = None) -> float:
    cfg = config or AudioFilterConfig()
    score = 0.0

    if quality.effective_duration:
        score += quality.effective_duration
        if 10.0 <= quality.effective_duration <= 15.0:
            score += 3.0
        elif cfg.min_duration <= quality.effective_duration < 10.0:
            score += 2.0
        elif 15.0 < quality.effective_duration <= cfg.max_duration:
            score += 1.0
        elif quality.effective_duration < cfg.min_duration:
            score -= (cfg.min_duration - quality.effective_duration) * 2.0
        elif quality.effective_duration > cfg.max_duration:
            score -= (quality.effective_duration - cfg.max_duration) * 0.5

    if quality.spectral_centroid:
        score += min(quality.spectral_centroid / 1000.0, 3.0)
        if quality.spectral_centroid < cfg.min_centroid:
            score -= (cfg.min_centroid - quality.spectral_centroid) / 1000.0 * 4.0

    if quality.silence_ratio:
        score -= quality.silence_ratio * 4.0

    if quality.max_silence_sec:
        score -= quality.max_silence_sec * 1.0

    if quality.nasal_ratio:
        score -= max(quality.nasal_ratio - 1.0, 0) * 10.0
        if quality.nasal_ratio > 1.6:
            score -= (quality.nasal_ratio - 1.6) * 8.0

    if quality.f0_std is not None:
        if quality.f0_std < cfg.min_f0_std:
            score -= (cfg.min_f0_std - quality.f0_std) / max(cfg.min_f0_std, 1.0) * 6.0
        else:
            score += min(quality.f0_std / 50.0, 2.0)

    if quality.pitch_range is not None and quality.pitch_range > cfg.max_pitch_range:
        score -= (quality.pitch_range - cfg.max_pitch_range) / max(cfg.max_pitch_range, 1.0) * 5.0

    return score
