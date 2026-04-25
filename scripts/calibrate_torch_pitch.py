#!/usr/bin/env python3
"""
Calibrate torch pitch thresholds against librosa baseline labels.

Usage:
  uv run python scripts/calibrate_torch_pitch.py --sample-size 300 --objective balanced
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate torch pitch thresholds from local result-voices data.")
    parser.add_argument("--voice-root", default="result-voices", help="Root directory of voice dataset")
    parser.add_argument("--sample-size", type=int, default=120, help="Number of mp3 files to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--freq-low", type=float, default=70.0, help="Torch detect_pitch_frequency freq_low")
    parser.add_argument("--freq-high", type=float, default=400.0, help="Torch detect_pitch_frequency freq_high")
    parser.add_argument(
        "--objective",
        choices=("balanced", "joint"),
        default="balanced",
        help="Threshold search objective: balanced class accuracy or joint accuracy",
    )
    return parser.parse_args()


def _load_rows(args: argparse.Namespace):
    voice_root = Path(args.voice_root)
    all_files = sorted(voice_root.glob("*/*.mp3"))
    random.seed(args.seed)
    files = random.sample(all_files, min(args.sample_size, len(all_files)))

    rows = []
    skipped_too_short = 0
    skipped_librosa = 0
    skipped_torch = 0
    for path in files:
        samples, sample_rate = librosa.load(path, sr=None, mono=True)
        samples, _ = librosa.effects.trim(samples, top_db=30)
        if sample_rate <= 0 or len(samples) < int(sample_rate * 0.35):
            skipped_too_short += 1
            continue

        try:
            f0, voiced_flag, _ = librosa.pyin(
                samples,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
            )
        except Exception:
            skipped_librosa += 1
            continue
        if f0 is None or voiced_flag is None:
            skipped_librosa += 1
            continue
        valid_lib = f0[voiced_flag]
        if valid_lib.size < 5:
            skipped_librosa += 1
            continue

        lib_std = float(np.std(valid_lib))
        lib_range = float(np.quantile(valid_lib, 0.9) - np.quantile(valid_lib, 0.1))
        lib_mono = lib_std < 20.0
        lib_sing = lib_range > 340.0

        wave = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
        try:
            f0_torch = torchaudio.functional.detect_pitch_frequency(
                wave,
                sample_rate=sample_rate,
                frame_time=0.01,
                freq_low=args.freq_low,
                freq_high=args.freq_high,
            )
        except RuntimeError:
            skipped_torch += 1
            continue
        valid_torch = f0_torch[f0_torch > 0]
        if valid_torch.numel() < 5:
            skipped_torch += 1
            continue

        torch_std = float(torch.std(valid_torch, unbiased=False).item())
        torch_range = float((torch.quantile(valid_torch, 0.9) - torch.quantile(valid_torch, 0.1)).item())
        rows.append((lib_mono, lib_sing, torch_std, torch_range, lib_std, lib_range, path.name))

    stats = {
        "sampled": len(files),
        "rows": len(rows),
        "skipped_too_short": skipped_too_short,
        "skipped_librosa": skipped_librosa,
        "skipped_torch": skipped_torch,
    }
    return rows, stats


def _balanced_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    tpr = tp / (tp + fn) if (tp + fn) else 1.0
    tnr = tn / (tn + fp) if (tn + fp) else 1.0
    return (tpr + tnr) / 2.0


def _find_best_thresholds(rows, objective: str):
    best = None
    for min_std in range(40, 181, 1):
        for max_range in range(240, 501, 1):
            joint = mono = sing = 0
            mono_tp = mono_tn = mono_fp = mono_fn = 0
            sing_tp = sing_tn = sing_fp = sing_fn = 0
            for lib_mono, lib_sing, torch_std, torch_range, *_ in rows:
                torch_mono = torch_std < min_std
                torch_sing = torch_range > max_range
                mono_tp += int(torch_mono and lib_mono)
                mono_tn += int((not torch_mono) and (not lib_mono))
                mono_fp += int(torch_mono and (not lib_mono))
                mono_fn += int((not torch_mono) and lib_mono)
                sing_tp += int(torch_sing and lib_sing)
                sing_tn += int((not torch_sing) and (not lib_sing))
                sing_fp += int(torch_sing and (not lib_sing))
                sing_fn += int((not torch_sing) and lib_sing)
                mono += int(torch_mono == lib_mono)
                sing += int(torch_sing == lib_sing)
                joint += int((torch_mono == lib_mono) and (torch_sing == lib_sing))
            n = len(rows)
            joint_acc = joint / n
            mono_acc = mono / n
            sing_acc = sing / n
            mono_bal = _balanced_accuracy(mono_tp, mono_tn, mono_fp, mono_fn)
            sing_bal = _balanced_accuracy(sing_tp, sing_tn, sing_fp, sing_fn)
            objective_score = (mono_bal + sing_bal) / 2.0 if objective == "balanced" else joint_acc

            cand = (
                objective_score,
                joint_acc,
                mono_bal,
                sing_bal,
                mono_acc,
                sing_acc,
                float(min_std),
                float(max_range),
            )
            if best is None or cand[:4] > best[:4]:
                best = cand
    return best


def main() -> None:
    args = _parse_args()
    rows, stats = _load_rows(args)
    if not rows:
        raise SystemExit("No valid samples found for calibration.")

    best = _find_best_thresholds(rows, args.objective)
    score, joint, mono_bal, sing_bal, mono_acc, sing_acc, min_std, max_range = best

    print("stats", stats)
    print(f"lib_mono_true={sum(1 for r in rows if r[0])} lib_sing_true={sum(1 for r in rows if r[1])}")
    print(
        "best_thresholds",
        {
            "torch_min_f0_std": min_std,
            "torch_max_pitch_range": max_range,
            "objective": args.objective,
            "objective_score": round(score, 4),
            "joint_acc": round(joint, 4),
            "mono_bal_acc": round(mono_bal, 4),
            "sing_bal_acc": round(sing_bal, 4),
            "mono_acc": round(mono_acc, 4),
            "sing_acc": round(sing_acc, 4),
            "freq_low": args.freq_low,
            "freq_high": args.freq_high,
        },
    )
    print(
        "means",
        {
            "torch_std": round(float(np.mean([r[2] for r in rows])), 3),
            "torch_range": round(float(np.mean([r[3] for r in rows])), 3),
            "lib_std": round(float(np.mean([r[4] for r in rows])), 3),
            "lib_range": round(float(np.mean([r[5] for r in rows])), 3),
        },
    )


if __name__ == "__main__":
    main()
