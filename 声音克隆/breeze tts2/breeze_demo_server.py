#!/usr/bin/env python3
"""Breeze TTS 2 演示服务：启动时加载模型并常驻内存，避免每次 CLI 冷启动。

依赖：mlx-audio 0.6+，以及 server 可选依赖（FastAPI / Uvicorn）：
  pip install "mlx-audio[server]"

示例：
  python breeze_demo_server.py --warmup
  python breeze_demo_client.py design --text "[笑] 好的呀。" --instruct "温柔女声。"
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

_SCRIPT_DIR = Path(__file__).resolve().parent
_UI_PATH = _SCRIPT_DIR / "breeze_demo_ui.html"
_DEMO_ASSETS = {"1.mp3"}
_DEMO_PRESETS_PATH = _SCRIPT_DIR / "demo_presets.json"

DEFAULT_MODEL = "mlx-community/Breeze-TTS-2-mlx-4bit"

_state: dict[str, Any] = {
    "model": None,
    "model_id": None,
    "load_seconds": None,
    "output_dir": Path("audio_test"),
    "request_count": 0,
}


def _get_model():
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return model


def _synthesize(
    *,
    text: str,
    instruct: str | None = None,
    ref_audio: str | Path | None = None,
    ref_text: str | None = None,
    cfg_scale: float = 4.0,
    voice: str | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    model = _get_model()
    t0 = time.perf_counter()
    chunks: list[np.ndarray] = []
    sample_rate: int | None = None
    last_meta: dict[str, Any] = {}

    gen_kw: dict[str, Any] = {
        "text": text,
        "cfg_scale": cfg_scale,
        "verbose": verbose,
    }
    if instruct is not None:
        gen_kw["instruct"] = instruct
    if ref_audio is not None:
        gen_kw["ref_audio"] = str(ref_audio) if isinstance(ref_audio, Path) else ref_audio
    if ref_text is not None:
        gen_kw["ref_text"] = ref_text
    if voice is not None:
        gen_kw["voice"] = voice

    for result in model.generate(**gen_kw):
        chunks.append(np.array(result.audio))
        sample_rate = int(result.sample_rate)
        last_meta = {
            "processing_time_seconds": result.processing_time_seconds,
            "real_time_factor": result.real_time_factor,
            "peak_memory_usage_gb": result.peak_memory_usage,
            "audio_duration": result.audio_duration,
        }

    if not chunks or sample_rate is None:
        raise HTTPException(status_code=500, detail="No audio generated.")

    audio = np.concatenate(chunks, axis=0)
    wall = time.perf_counter() - t0
    last_meta["inference_wall_seconds"] = round(wall, 3)
    return audio, sample_rate, last_meta


def _wav_response(path: Path, meta: dict[str, Any]) -> FileResponse:
    headers = {}
    if meta.get("processing_time_seconds") is not None:
        headers["X-Processing-Time-S"] = f"{meta['processing_time_seconds']:.2f}"
    if meta.get("inference_wall_seconds") is not None:
        headers["X-Inference-Wall-S"] = f"{meta['inference_wall_seconds']:.3f}"
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=path.name,
        headers=headers,
    )


def _save_wav(audio: np.ndarray, sample_rate: int, file_prefix: str) -> Path:
    from mlx_audio.audio_io import write as audio_write

    out_dir: Path = _state["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{file_prefix}.wav"
    audio_write(str(path), audio, sample_rate, format="wav")
    return path


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    model_id = _state["model_id"]
    print(f"Loading model (one-time): {model_id} …")
    t0 = time.perf_counter()
    from mlx_audio.tts import load

    _state["model"] = load(model_id)
    _state["load_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"✓ Model loaded in {_state['load_seconds']}s")

    if _state.get("warmup"):
        print("Warming up (first Metal graph compile) …")
        tw = time.perf_counter()
        _synthesize(text="。", instruct="清晰女声。", cfg_scale=1.0, verbose=False)
        print(f"✓ Warmup done in {time.perf_counter() - tw:.2f}s")

    ui_url = _state.get("ui_url")
    print(f"✓ Ready — output dir: {_state['output_dir'].resolve()}")
    if ui_url:
        print(f"✓ Web UI: {ui_url}")
    yield
    _state["model"] = None


app = FastAPI(title="Breeze TTS 2 Demo Server", lifespan=lifespan)


@app.get("/ui", response_class=HTMLResponse)
def demo_ui():
    if not _UI_PATH.is_file():
        raise HTTPException(status_code=500, detail=f"UI file missing: {_UI_PATH}")
    return HTMLResponse(_UI_PATH.read_text(encoding="utf-8"))


@app.get("/")
def root():
    return {
        "service": "breeze-demo-server",
        "ui": "/ui",
        "model": _state.get("model_id"),
        "loaded": _state.get("model") is not None,
        "load_seconds": _state.get("load_seconds"),
        "output_dir": str(_state["output_dir"].resolve()),
        "endpoints": {
            "health": "GET /health",
            "design": "POST /api/design  (JSON)",
            "clone": "POST /api/clone  (multipart: ref_audio file + forms)",
            "direction": "POST /api/direction  (JSON, server path)",
            "direction_form": "POST /api/direction/form  (multipart, for Web UI)",
        },
    }


def _load_demo_presets() -> dict[str, Any]:
    import json

    if _DEMO_PRESETS_PATH.is_file():
        try:
            return json.loads(_DEMO_PRESETS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"Warning: demo_presets.json invalid: {e}")
    ref_file = _SCRIPT_DIR / "1.mp3"
    return {
        "demoRef": {
            "filename": "1.mp3",
            "refText": "你好，这是我的声音采样，用于测试语音克隆效果。",
        },
        "cases": {},
    }


@app.get("/api/demo-config")
def demo_config():
    """Web UI：演示参考音路径与预设案例（可选 demo_presets.json）。"""
    data = _load_demo_presets()
    ref_name = (data.get("demoRef") or {}).get("filename") or "1.mp3"
    ref_path = _SCRIPT_DIR / ref_name
    demo_ref = {
        "filename": ref_name,
        "url": f"/demo/{ref_name}" if ref_name in _DEMO_ASSETS else None,
        "serverPath": str(ref_path.resolve()) if ref_path.is_file() else None,
        "refText": (data.get("demoRef") or {}).get("refText") or "",
    }
    return {"demoRef": demo_ref, "cases": data.get("cases") or {}}


@app.get("/demo/{filename}")
def demo_asset(filename: str):
    if filename not in _DEMO_ASSETS:
        raise HTTPException(status_code=404, detail="Unknown demo asset")
    path = _SCRIPT_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Missing demo file: {filename}")
    media = "audio/mpeg" if filename.endswith(".mp3") else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=filename)


@app.get("/health")
def health():
    return {
        "ok": _state.get("model") is not None,
        "model": _state.get("model_id"),
        "load_seconds": _state.get("load_seconds"),
        "requests": _state.get("request_count", 0),
    }


class DesignRequest(BaseModel):
    text: str = Field(..., description="要合成的正文")
    instruct: str = Field(..., description="声音设计：自然语言描述音色与风格")
    cfg_scale: float = Field(4.0, description="CFG 强度，常用 4")
    voice: str | None = Field(None, description="Breeze 说话人标签，可选 S0")
    file_prefix: str = Field("breeze_design", description="输出 WAV 文件名（不含扩展名）")
    return_file: bool = Field(True, description="True 直接返回 WAV；False 只 JSON 路径")
    persist: bool = Field(
        True,
        description="True 同时写入服务端 --output-dir；False 仅返回音频（不落盘到 output-dir）",
    )


@app.post("/api/design")
async def api_design(body: DesignRequest):
    """声音设计：无参考音频，仅 instruct + text。"""

    def sync_run():
        audio, sr, meta = _synthesize(
            text=body.text,
            instruct=body.instruct,
            cfg_scale=body.cfg_scale,
            voice=body.voice,
            verbose=True,
        )
        if body.persist:
            p = _save_wav(audio, sr, body.file_prefix)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            p = Path(tmp.name)
            from mlx_audio.audio_io import write as audio_write

            audio_write(str(p), audio, sr, format="wav")
        _state["request_count"] = _state.get("request_count", 0) + 1
        return p, meta, body.persist

    path, meta, persisted = await asyncio.to_thread(sync_run)
    if body.return_file:
        return _wav_response(path, meta)
    payload = {"meta": meta}
    if persisted:
        payload["saved"] = str(path.resolve())
    else:
        payload["saved"] = None
        payload["note"] = "persist=false，未写入 output-dir"
    return payload


class DirectionRequest(BaseModel):
    text: str
    instruct: str
    ref_audio_path: str = Field(..., description="服务器本机可读的参考 WAV 路径")
    ref_text: str
    cfg_scale: float = 4.0
    voice: str | None = "S0"
    file_prefix: str = "breeze_direction"
    return_file: bool = True


@app.post("/api/direction")
async def api_direction(body: DirectionRequest):
    """声音方向：参考音 + instruct。"""

    ref = Path(body.ref_audio_path).expanduser()
    if not ref.is_file():
        raise HTTPException(status_code=400, detail=f"ref_audio_path not found: {ref}")

    def sync_run():
        audio, sr, meta = _synthesize(
            text=body.text,
            instruct=body.instruct,
            ref_audio=str(ref.resolve()),
            ref_text=body.ref_text,
            cfg_scale=body.cfg_scale,
            voice=body.voice,
            verbose=True,
        )
        p = _save_wav(audio, sr, body.file_prefix)
        _state["request_count"] = _state.get("request_count", 0) + 1
        return p, meta

    path, meta = await asyncio.to_thread(sync_run)
    if body.return_file:
        return _wav_response(path, meta)
    return {"saved": str(path.resolve()), "meta": meta}


async def _resolve_uploaded_ref(
    ref_audio: UploadFile | None,
    ref_audio_path: str | None,
) -> tuple[Path, tempfile.NamedTemporaryFile | None]:
    temp_file: tempfile.NamedTemporaryFile | None = None
    if ref_audio_path:
        ref_path = Path(ref_audio_path).expanduser()
        if not ref_path.is_file():
            raise HTTPException(status_code=400, detail=f"ref_audio_path not found: {ref_path}")
        return ref_path, None
    if ref_audio and ref_audio.filename:
        suffix = Path(ref_audio.filename).suffix or ".wav"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        data = await ref_audio.read()
        temp_file.write(data)
        temp_file.close()
        return Path(temp_file.name), temp_file
    raise HTTPException(status_code=400, detail="Provide ref_audio upload or ref_audio_path.")


@app.post("/api/direction/form")
async def api_direction_form(
    text: str = Form(...),
    instruct: str = Form(...),
    ref_text: str = Form(...),
    ref_audio: UploadFile | None = File(None),
    ref_audio_path: str | None = Form(None),
    cfg_scale: float = Form(4.0),
    voice: str = Form("S0"),
    file_prefix: str = Form("breeze_direction"),
    return_file: bool = Form(True),
):
    """声音方向（网页上传参考音或 ref_audio_path）。"""

    ref_path, temp_file = await _resolve_uploaded_ref(ref_audio, ref_audio_path)

    def sync_run():
        try:
            audio, sr, meta = _synthesize(
                text=text,
                instruct=instruct,
                ref_audio=ref_path,
                ref_text=ref_text,
                cfg_scale=cfg_scale,
                voice=voice,
                verbose=True,
            )
            p = _save_wav(audio, sr, file_prefix)
            _state["request_count"] = _state.get("request_count", 0) + 1
            return p, meta
        finally:
            if temp_file is not None:
                Path(temp_file.name).unlink(missing_ok=True)

    path, meta = await asyncio.to_thread(sync_run)
    if return_file:
        return _wav_response(path, meta)
    return {"saved": str(path.resolve()), "meta": meta}


@app.post("/api/clone")
async def api_clone(
    text: str = Form(...),
    ref_text: str = Form(...),
    ref_audio: UploadFile | None = File(None),
    ref_audio_path: str | None = Form(None),
    voice: str = Form("S0"),
    file_prefix: str = Form("breeze_clone"),
    return_file: bool = Form(True),
):
    """声音克隆：上传 ref_audio 或提供 ref_audio_path（服务器本地路径）。"""

    ref_path, temp_file = await _resolve_uploaded_ref(ref_audio, ref_audio_path)

    def sync_run():
        try:
            audio, sr, meta = _synthesize(
                text=text,
                instruct=None,
                ref_audio=ref_path,
                ref_text=ref_text,
                voice=voice,
                verbose=True,
            )
            p = _save_wav(audio, sr, file_prefix)
            _state["request_count"] = _state.get("request_count", 0) + 1
            return p, meta
        finally:
            if temp_file is not None:
                Path(temp_file.name).unlink(missing_ok=True)

    path, meta = await asyncio.to_thread(sync_run)
    if return_file:
        return _wav_response(path, meta)
    return {"saved": str(path.resolve()), "meta": meta}


def main():
    parser = argparse.ArgumentParser(description="Breeze TTS 2 demo server (model kept in RAM)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("audio_test"),
        help="Directory for saved WAV files (default: ./audio_test under cwd)",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one short synthesis after load to reduce first-request latency",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open /ui in the default browser after startup",
    )
    args = parser.parse_args()

    _state["model_id"] = args.model
    _state["output_dir"] = args.output_dir.resolve()
    _state["output_dir"].mkdir(parents=True, exist_ok=True)
    _state["warmup"] = args.warmup
    _state["ui_url"] = f"http://{args.host}:{args.port}/ui"

    if args.open:
        import threading
        import webbrowser

        def _open():
            time.sleep(1.2)
            webbrowser.open(_state["ui_url"])

        threading.Thread(target=_open, daemon=True).start()

    print(f"Starting server http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
