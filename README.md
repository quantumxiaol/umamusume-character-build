# umamusume-character-build

用于把上游项目产出的 `prompt` 与 `voice` 数据，构建成 `umamusume-agent` 可直接消费的角色卡目录（`characters/<character_slug>/`）。

## 前置数据来源

本项目**不负责抓取原始数据**，依赖以下两个上游项目先产出结果文件：

1. `umamusume-agent-prompt`
2. `umamusume-voice-data`

构建前请保证当前仓库根目录已有：

- `result-prompts/`：来自 `umamusume-agent-prompt` 的角色提示词结果（如 `Admire_Vega.md`）
- `result-voices/`：来自 `umamusume-voice-data` 的角色语音结果（如 `Admire Vega/*.mp3 + *_jp.txt + *_zh.txt`）

## 输出目录

构建输出到 `characters/`，并保持角色目录结构不变：

```text
characters/<character_slug>/
├── config.json
├── prompt.md
├── reference.mp3
├── reference_jp.txt
└── reference_zh.txt
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

预览不落盘：

```bash
uv run python main.py -c "Admire Vega" --dry-run --workers 4 --verbose
```

## 说明

- 音频筛选目标是给 Index TTS 做音色克隆的单条高质量参考音频。
- 可通过 `.env` 指定音频分析设备：`device=cpu` / `device=mps` / `device=cuda`（不可用时自动回退 CPU）。
- 构建以 `result-prompts/` 为主：只要 prompt 存在，就会构建角色目录。
- 如果 `result-voices/` 中缺少该角色音频（或音频全不合格），也会继续构建，并在 `voice_config` 写入 `"no_voice": "true"`。
- 角色名优先通过 `result-prompts/*.md` 与 `result-voices/<角色英文名>/` 自动匹配。
- 若存在 `umamusume_characters.json`，会用于英文名到中文名映射；没有时会从 prompt 标题自动提取中文名。
