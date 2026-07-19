#!/usr/bin/env python3
"""Verify the two deployment tracks and record the observed local result."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

import torch

from nano_gpt import GPTConfig, GPTMini


ROOT = Path(__file__).resolve().parent


def verify_track_a() -> dict:
    torch.manual_seed(7)
    cfg = GPTConfig(vocab_size=128, n_layer=2, n_head=4, d_model=64, max_len=16)
    model = GPTMini(cfg).eval()
    sample = torch.tensor([[1, 5, 9, 2, 7]], dtype=torch.long)
    with torch.inference_mode():
        expected = model(sample)

    with tempfile.TemporaryDirectory(prefix="gptmini-track-a-") as directory:
        path = Path(directory)
        (path / "config.json").write_text(json.dumps(asdict(cfg)))
        torch.save({"model": model.state_dict()}, path / "final.pt")

        loaded_cfg = GPTConfig(**json.loads((path / "config.json").read_text()))
        loaded = GPTMini(loaded_cfg).eval()
        checkpoint = torch.load(path / "final.pt", map_location="cpu", weights_only=True)
        loaded.load_state_dict(checkpoint["model"])
        with torch.inference_mode():
            actual = loaded(sample)

    delta = float((expected - actual).abs().max())
    return {
        "status": "pass" if delta == 0.0 else "fail",
        "artifact_contract": ["nano_gpt.py", "config.json", "final.pt"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "sample_shape": list(sample.shape),
        "logit_shape": list(actual.shape),
        "max_abs_logit_delta_after_reload": delta,
        "supports_auto_model": False,
        "supports_gguf": False,
    }


def verify_track_b(endpoint: str, model_name: str) -> dict:
    inspected = subprocess.run(
        ["docker", "model", "inspect", model_name],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(inspected.stdout)
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Answer with the result only."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        "temperature": 0,
        "max_tokens": 12,
    }
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        completion = json.load(response)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    answer = completion["choices"][0]["message"]["content"].strip()
    config = metadata["config"]
    passed = config["format"] == "gguf" and "Q4" in config["quantization"] and bool(answer)
    return {
        "status": "pass" if passed else "fail",
        "model": model_name,
        "runtime": "Docker Model Runner llama.cpp backend",
        "format": config["format"],
        "architecture": config["architecture"],
        "quantization": config["quantization"],
        "parameters": config["parameters"],
        "size": config["size"],
        "smoke_prompt": "What is 2 + 2?",
        "smoke_answer": answer,
        "latency_ms": latency_ms,
        "usage": completion.get("usage", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:12434/engines/v1/chat/completions")
    parser.add_argument("--model", default="ai/smollm2:360M-Q4_K_M")
    args = parser.parse_args()

    result = {
        "observed_at": "2026-07-19",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()},
        "track_a_custom_pytorch": verify_track_a(),
        "track_b_compatible_gguf": verify_track_b(args.endpoint, args.model),
    }
    output = ROOT / "verification-local.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
