# Breeze TTS 2 本地演示（MLX）

Apple Silicon + **mlx-audio** 常驻服务：网页演示**声音设计 / 克隆 / 方向**，并可批量跑三段「声音设计」拍片素材。

## 环境

- macOS，Apple Silicon（M 系列）
- Python **≥ 3.10**
- 已安装：`pip install "mlx-audio[server]"`（含 FastAPI / Uvicorn）
- 模型：`mlx-community/Breeze-TTS-2-mlx-4bit`（首次由 Hugging Face 自动下载/读缓存）

## 本目录文件

| 文件 | 作用 |
|------|------|
| `breeze_demo_server.py` | 启动时加载模型；提供 Web UI 与 API |
| `breeze_demo_ui.html` | 演示页面（由服务挂载在 `/ui`） |
| `breeze_design_showcase.py` | 调用 API：3 种音色各一句 → 分段 WAV + 合并 `showcase_merged.wav` |
| `demo_presets.json` | 克隆演示用参考稿；需与参考音频逐字一致 |
| `1.mp3` | **自备**：克隆参考音，放在本目录（与 `demo_presets.json` 中 `filename` 一致） |

## 快速开始

```bash
# 建议先激活你的 Python 虚拟环境
python breeze_demo_server.py --warmup --open
```

- 浏览器：**http://127.0.0.1:8765/ui**
- 服务端生成的 WAV 默认在 **`./audio_test/`**（不会堆在本目录根下）
- 结束服务：终端 `Ctrl+C`

## 批量：三段声音设计 + 合并

需先保持 `breeze_demo_server.py` 在运行：

```bash
python breeze_design_showcase.py --out ./showcase_out
```

输出：`showcase_01/02/03_*.wav`、`showcase_merged.wav`、`showcase_manifest.json`。  
脚本用 `persist=false`，不会在 `audio_test/` 再写一份重复文件。

## 克隆注意

1. 将 `1.mp3` 放入本目录。  
2. `demo_presets.json` 里 `refText` 必须与 `1.mp3` **完全一致**。  
3. 修改 `demo_presets.json` 后需**重启**服务。

## 常用参数

| 参数 | 说明 |
|------|------|
| `--warmup` | 启动后预热，减少首条请求延迟 |
| `--open` | 自动打开 `/ui` |
| `--output-dir` | 默认 `audio_test`，服务端落盘目录 |
| `--port` | 默认 `8765` |

## 许可

Breeze 权重与自托管输出为研究与非商业用途；详见 [BreezeBlue 模型卡](https://huggingface.co/BreezeBlue/Breeze-TTS-2)。
