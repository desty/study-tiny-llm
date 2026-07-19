# Deployment-boundary verification

This example turns the book's two deployment tracks into executable checks.

- Track A creates a small `GPTMini`, saves `nano_gpt.py + config.json + final.pt`, reloads it, and requires an exact logit match.
- Track B inspects a real Q4 GGUF artifact and sends a smoke prompt through Docker Model Runner's llama.cpp backend.

```bash
python -m venv .venv
.venv/bin/pip install -r examples/deployment-boundary/requirements.txt

docker desktop enable model-runner --tcp=12434
docker model pull ai/smollm2:360M-Q4_K_M
.venv/bin/python examples/deployment-boundary/verify_tracks.py
```

`verification-local.json` is the observed local run. Its latency is environment-specific and is not a benchmark. Track A intentionally reports `supports_auto_model=false` and `supports_gguf=false`; changing those values requires real Transformers and llama.cpp architecture support, not a renamed config file.
