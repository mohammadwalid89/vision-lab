#!/usr/bin/env python3
"""Fetch pinned image and language tensors while keeping root free above 8 GiB."""

import hashlib
import json
import os
import struct
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REV = "f9972ac22e696a105c482202646979e33e4b5a1e"
BASE = f"https://huggingface.co/google/embeddinggemma-2/resolve/{REV}/"
FLOOR = 8 * 1024**3
MODEL = HERE / "model"


def free():
    s = os.statvfs(HERE)
    return s.f_bavail * s.f_frsize


def get_range(url, start, end):
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
    r.raise_for_status()
    assert r.status_code == 206 and len(r.content) == end - start + 1
    return r.content


def main():
    MODEL.mkdir(exist_ok=True)
    url = BASE + "model.safetensors"
    n = struct.unpack("<Q", get_range(url, 0, 7))[0]
    original = json.loads(get_range(url, 8, 7 + n))
    keep = {k: v for k, v in original.items() if k.startswith(("language_model.", "vision_tower.", "embed_vision."))}
    start = min(v["data_offsets"][0] for v in keep.values())
    end = max(v["data_offsets"][1] for v in keep.values())
    assert start == 611223040 and end == 1488743984
    for v in keep.values():
        v["data_offsets"] = [x - start for x in v["data_offsets"]]
    header = json.dumps({"__metadata__": {"format": "pt"}, **keep}, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    size = 8 + len(header) + end - start
    if free() - size < FLOOR:
        raise SystemExit(f"Insufficient headroom: free={free()} need={size} floor={FLOOR}")
    out = MODEL / "model.safetensors"
    if not out.exists():
        tmp = MODEL / "model.safetensors.part"
        digest = hashlib.sha256()
        with tmp.open("wb") as f:
            head = struct.pack("<Q", len(header)) + header
            f.write(head)
            digest.update(head)
            for offset in range(start, end, 8 * 1024 * 1024):
                chunk = get_range(url, 8 + offset, 8 + min(offset + 8 * 1024 * 1024, end) - 1)
                f.write(chunk)
                digest.update(chunk)
                if free() < FLOOR:
                    raise SystemExit("Disk floor reached; partial file retained")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(out)
        print("subset_sha256", digest.hexdigest(), flush=True)
    for name in [
        "config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]:
        r = requests.get(BASE + name, timeout=60)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        (MODEL / name).write_bytes(r.content)
    config = json.loads((MODEL / "config.json").read_text())
    config["audio_config"] = None
    (MODEL / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    from safetensors import safe_open

    with safe_open(out, framework="pt", device="cpu") as f:
        assert len(f.keys()) == len(keep)
    digest = hashlib.sha256()
    with out.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    (HERE / "model_provenance.json").write_text(
        json.dumps(
            {
                "source": BASE,
                "revision": REV,
                "source_safetensors_bytes": 1488915288,
                "included_prefixes": ["language_model", "vision_tower", "embed_vision"],
                "subset_sha256": digest.hexdigest(),
                "subset_bytes": out.stat().st_size,
            },
            indent=2,
        )
        + "\n"
    )
    print("ready", out.stat().st_size, "free", free(), flush=True)


if __name__ == "__main__":
    main()
