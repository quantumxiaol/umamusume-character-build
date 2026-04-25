from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

try:
    from umamusume_character_build import CharacterBuilder
    from umamusume_character_build.quality_filter import get_audio_analysis_device
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    from umamusume_character_build import CharacterBuilder
    from umamusume_character_build.quality_filter import get_audio_analysis_device


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Umamusume character cards from local prompt/voice results.")
    parser.add_argument(
        "--character",
        "-c",
        dest="characters",
        action="append",
        help="Build only this character (can be passed multiple times). Example: -c 'Admire Vega'",
    )
    parser.add_argument("--prompt-dir", default="result-prompts", help="Prompt directory path")
    parser.add_argument("--voice-dir", default="result-voices", help="Voice directory path")
    parser.add_argument("--image-dir", default="result-images", help="Image directory path")
    parser.add_argument("--characters-dir", default="characters", help="Character output directory path")
    parser.add_argument("--character-map-path", default="umamusume_characters.json", help="Character map json path")
    parser.add_argument("--workers", type=int, default=1, help="Worker count for audio analysis")
    parser.add_argument("--max-candidates", type=int, default=None, help="Max audio candidates per character")
    parser.add_argument("--include-images", action="store_true", help="Package Jsf/Zf character images during build")
    parser.add_argument("--images-only", action="store_true", help="Only package images for already built characters")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be built")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs")
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def _run_build(args: argparse.Namespace) -> List[str]:
    builder = CharacterBuilder(
        prompt_dir=args.prompt_dir,
        voice_dir=args.voice_dir,
        image_dir=args.image_dir,
        characters_dir=args.characters_dir,
        character_map_path=args.character_map_path,
    )
    target_characters: Optional[List[str]] = args.characters if args.characters else None
    if args.images_only:
        return builder.package_images(
            dry_run=args.dry_run,
            target_characters=target_characters,
        )
    return builder.build_all(
        dry_run=args.dry_run,
        workers=args.workers,
        target_characters=target_characters,
        max_candidates=args.max_candidates,
        include_images=args.include_images,
    )


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)
    if args.verbose:
        logging.getLogger(__name__).info("Audio analysis device: %s", get_audio_analysis_device())
    results = _run_build(args)
    if results:
        action = "Packaged images for" if args.images_only else "Built"
        print(f"{action} {len(results)} character(s): {', '.join(results)}")
    else:
        print("No characters were processed.")


if __name__ == "__main__":
    main()
