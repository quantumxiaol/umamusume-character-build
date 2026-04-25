# umamusume-character-build

用于把上游项目产出的 `prompt`、`voice` 与可选 `image` 数据，构建成 `umamusume-agent` 可直接消费的角色卡目录（`characters/<character_slug>/`）。

## 前置数据来源

本项目**不负责抓取原始数据**，依赖以下两个上游项目先产出结果文件：

1. `umamusume-agent-prompt`
2. `umamusume-voice-data`

构建前请保证当前仓库根目录已有：

- `result-prompts/`：来自 `umamusume-agent-prompt` 的角色提示词结果（如 `Admire_Vega.md`）
- `result-voices/`：来自 `umamusume-voice-data` 的角色语音结果（如 `Admire Vega/*.mp3 + *_jp.txt + *_zh.txt`）
- `result-images/`：可选，角色图片结果（如 `Admire Vega/Jsf_103301.png + Zf_103301.png`）

## 输出目录

构建输出到 `characters/`，并保持角色目录结构不变：

```text
characters/<character_slug>/
├── config.json
├── prompt.md
├── reference.mp3
├── reference_jp.txt
├── reference_zh.txt
├── JSF_<character-name>.png
└── ZF_<character-name>.png
```

## 安装依赖

```bash
uv sync
```

## 使用方式

只构建单个角色（推荐）：

```bash
uv run python main.py -c "Admire Vega" --workers 4
```

构建所有可匹配角色：

```bash
uv run python main.py --workers 4
```

构建时同时打包角色图片：

```bash
uv run python main.py -c "Admire Vega" --workers 4 --include-images
```

只给已构建角色补充图片：

```bash
uv run python main.py --images-only
uv run python main.py --images-only -c "Admire Vega"
```

预览不落盘：

```bash
uv run python main.py -c "Admire Vega" --dry-run --workers 4 --verbose
```

## 说明

- 音频筛选目标是给 Index TTS / Qwen3-TTS 等做音色克隆的单条高质量参考音频，默认要求有效语音时长 5-25 秒，并过滤静音过多或中间空白过长的音频。
- 可通过 `.env` 指定音频分析设备：`device=cpu` / `device=mps` / `device=cuda`（不可用时自动回退 CPU）。
- 当 `device` 为 `mps/cuda` 时，音频分析会启用 Torch 路径，并使用校准后的 Torch 音高阈值。
- 候选音频会先做全局质量分析，再按**每个角色**的候选分布自适应阈值（避免低沉声线被全局阈值误伤）。
- 构建以 `result-prompts/` 为主：只要 prompt 存在，就会构建角色目录。
- 如果 `result-voices/` 中缺少该角色音频（或音频全不合格），也会继续构建，并在 `voice_config` 写入 `"no_voice": "true"`。
- 角色名优先通过 `result-prompts/*.md` 与 `result-voices/<角色英文名>/` 自动匹配。
- 若存在 `umamusume_characters.json`，会用于英文名到中文名映射；没有时会从 prompt 标题自动提取中文名。
- 可通过 `--include-images` 打包 `result-images/<角色英文名>/` 下的 `Jsf_*` 与 `Zf_*` 图片到角色目录，文件名会重命名为 `JSF_<角色英文名连字符>.png` / `ZF_<角色英文名连字符>.png`（保留原始图片扩展名）。
- 可通过 `--images-only` 给 `characters/` 中已构建的角色补充图片，不会重建 prompt 或重新筛选音频。

## Torch 阈值校准

可用脚本重新校准 Torch 音高阈值（以 librosa 基线为教师，默认使用 `balanced` 目标）：

```bash
uv run python scripts/calibrate_torch_pitch.py --sample-size 300 --objective balanced
```

当前默认 Torch 阈值（`src/umamusume_character_build/quality_filter.py`，作为全局基线）：

- `torch_min_f0_std = 54`
- `torch_max_pitch_range = 308`
