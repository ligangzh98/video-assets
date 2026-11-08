#!/usr/bin/env python3
"""通过 breeze_demo_server 的 /api/design 生成 3 段「声音设计」演示音频。

前提：服务已启动且模型已加载，例如：
  python breeze_demo_server.py --warmup

用法：
  python breeze_design_showcase.py
  python breeze_design_showcase.py --out ./showcase_out

输出：保留 showcase_01/02/03.wav，并额外生成 showcase_merged.wav（三段顺序拼接）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

DEFAULT_BASE = "http://127.0.0.1:8765"

# 同一段议论拆成三句接力；连读 text 应语义不断。每句只用一种声音设计（instruct）。
# 全文：我们做本地 TTS 演示，会先让模型常驻内存；再靠 instruct 凭空设计音色；最后把三段连播，听成一条议论、三个角色。
SHOWCASE: list[dict] = [
    {
        "id": "anchor",
        "name": "沉稳男播",
        "instruct": (
            "成熟男性新闻播报腔，中低音，语速平稳，吐字清楚，"
            "语气克制、有权威感但不咄咄逼人。"
        ),
        "text": "我们做本地 TTS 演示，图的就是成本可控，信息不外泄；",
        "file_prefix": "showcase_01_anchor",
    },
    {
        "id": "guide",
        "name": "轻快女声",
        "instruct": (
            "二十多岁女性，声音明亮、有亲和力，语速略快，"
            "句尾自然，适合面向初学者的科普讲解。"
        ),
        "text": "[笑] 再靠指令凭空设计音色或者克隆已有音色",
        "file_prefix": "showcase_02_guide",
    },
    {
        "id": "podcast",
        "name": "深夜播客",
        "instruct": (
            "五岁左右小女孩的声音，声线温暖偏低，语速偏慢"
        ),
        "text": "[惊讶] 最后把各个片段拼接在一起，就是一个完整的对话了",
        "file_prefix": "showcase_03_podcast",
    },
]


def concat_wavs(segments: list[Path], merged_path: Path) -> None:
    """按顺序拼接 WAV（要求采样率/声道/位深一致）。"""
    if not segments:
        raise ValueError("没有可合并的分段音频")
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(segments[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]

    for path in segments[1:]:
        with wave.open(str(path), "rb") as w:
            if (
                w.getnchannels(),
                w.getsampwidth(),
                w.getframerate(),
                w.getcomptype(),
                w.getcompname(),
            ) != (
                params.nchannels,
                params.sampwidth,
                params.framerate,
                params.comptype,
                params.compname,
            ):
                raise ValueError(f"音频格式与首段不一致，无法合并：{path}")
            frames.append(w.readframes(w.getnframes()))

    with wave.open(str(merged_path), "wb") as out:
        out.setparams(params)
        for chunk in frames:
            out.writeframes(chunk)


def get_health(base: str) -> dict:
    url = f"{base.rstrip('/')}/health"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def design_speech(
    base: str,
    *,
    text: str,
    instruct: str,
    file_prefix: str,
    cfg_scale: float,
    out_path: Path,
) -> dict:
    url = f"{base.rstrip('/')}/api/design"
    payload = {
        "text": text,
        "instruct": instruct,
        "cfg_scale": cfg_scale,
        "file_prefix": file_prefix,
        "return_file": True,
        "persist": False,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = resp.read()
        wall = time.perf_counter() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(body)
    return {
        "wall_seconds": round(wall, 2),
        "processing_time_s": resp.headers.get("X-Processing-Time-S"),
        "inference_wall_s": resp.headers.get("X-Inference-Wall-S"),
        "bytes": len(body),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Breeze 声音设计三连演示（各音色仅用一次）")
    parser.add_argument("--base", default=DEFAULT_BASE, help="breeze_demo_server 地址")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("showcase_out"),
        help="保存 WAV 的目录（脚本落盘；默认 persist=false 不在服务端 output-dir 重复保存）",
    )
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument(
        "--merged-name",
        default="showcase_merged.wav",
        help="合并后的文件名（保存在 --out 目录，分段 WAV 保留不删）",
    )
    args = parser.parse_args()

    try:
        health = get_health(args.base)
    except urllib.error.URLError as e:
        print(f"无法连接服务 {args.base}：{e}", file=sys.stderr)
        print("请先运行：python breeze_demo_server.py --warmup", file=sys.stderr)
        return 1

    if not health.get("ok"):
        print("服务未就绪，模型仍在加载。", file=sys.stderr)
        return 1

    print(f"服务就绪 · 已合成 {health.get('requests', 0)} 次 · 输出目录 {args.out.resolve()}\n")

    manifest: list[dict] = []
    segment_paths: list[Path] = []
    for i, item in enumerate(SHOWCASE, start=1):
        wav = args.out / f"{item['file_prefix']}.wav"
        print(f"[{i}/3] {item['name']} ({item['id']}) …")
        try:
            meta = design_speech(
                args.base,
                text=item["text"],
                instruct=item["instruct"],
                file_prefix=item["file_prefix"],
                cfg_scale=args.cfg_scale,
                out_path=wav,
            )
        except urllib.error.HTTPError as e:
            print(e.read().decode(), file=sys.stderr)
            return 1

        print(f"      → {wav.name}  ({meta['bytes'] // 1024} KB, 请求 {meta['wall_seconds']}s)")
        segment_paths.append(wav)
        manifest.append(
            {
                **{k: item[k] for k in ("id", "name", "instruct", "text", "file_prefix")},
                "wav": str(wav.resolve()),
                **meta,
            }
        )

    merged_path = args.out / args.merged_name
    try:
        concat_wavs(segment_paths, merged_path)
    except ValueError as e:
        print(f"合并失败：{e}", file=sys.stderr)
        return 1
    merged_kb = merged_path.stat().st_size // 1024
    print(f"\n合并 → {merged_path.name}  ({merged_kb} KB，含 {len(segment_paths)} 段)")

    manifest_path = args.out / "showcase_manifest.json"
    doc = {
        "segments": manifest,
        "merged_wav": str(merged_path.resolve()),
        "full_text": "".join(item["text"] for item in SHOWCASE),
    }
    manifest_path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"清单：{manifest_path.resolve()}")
    print(f"连读正文：{doc['full_text']}")
    print("分段文件已保留；合并文件为按 01→02→03 顺序无缝拼接。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
