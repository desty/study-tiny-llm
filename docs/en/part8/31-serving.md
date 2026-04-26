# Serving — llama.cpp server · vLLM · Latency Budget

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part8/ch31_serving.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - **3 serving stacks** — `llama.cpp server` / **vLLM** / HF TGI
    - **Latency budget** arithmetic — token count × TPS = latency
    - **Batching, concurrency, KV cache** — 5–10× throughput on the same GPU
    - **Health checks, graceful shutdown, adapter hot-swap**

!!! quote "Prerequisites"
    [Ch 19 Quantization](../part6/19-quantization.md), [Ch 20 GGUF](../part6/20-llamacpp-gguf.md). You're now taking responsibility for production inference.

---

![Serving stack comparison — llama.cpp · vLLM · TGI · Ollama](../assets/diagrams/serving-stacks.svg#only-light)
![Serving stack comparison — llama.cpp · vLLM · TGI · Ollama](../assets/diagrams/serving-stacks-dark.svg#only-dark)

## 1. Serving Stack Comparison

| Stack | Strengths | Weaknesses | Best for |
|---|---|---|---|
| **llama.cpp server** | CPU/Mac/Vulkan all work, lightweight | Struggles with large models and high concurrency | Internal network, laptops, small services |
| **vLLM** | PagedAttention, **best throughput** | GPU only, heavyweight | 1+ GPUs with many concurrent users |
| **HF TGI** | Auto-supports standard HF models | Operational learning curve | HF-native environments |
| **Ollama** | User-friendly | Not production-grade | Demos, development |

**Recommendation for this book**:
- Internal/small service (≤ 10 concurrent users) → **llama.cpp server**
- 1+ GPU with many users → **vLLM**

---

## 2. Latency Budget Arithmetic

```
latency = prefill_time + decode_time
prefill_time = (input token count) / prefill_TPS
decode_time  = (output token count) / decode_TPS
```

This book's model (Qwen 0.5B Q4, M2 Pro):

| Item | Value |
|---|---:|
| Prefill TPS | ~2000 tok/s |
| Decode TPS | ~150 tok/s |
| 200 input tokens | 0.1 s |
| 100 output tokens | 0.7 s |
| **Total** | **~0.8 s** |

If your p95 latency budget is 1.5 s, you're fine. 5 concurrent users: feasible. 50 concurrent users: you need vLLM or a GPU.

---

## 3. llama.cpp server — Internal Small Service

```bash title="serve.sh"
./llama.cpp/llama-server \
    -m dist/tiny-tale-q4km.gguf \
    --host 0.0.0.0 --port 8080 \
    --ctx-size 1024 \
    --threads 8 \
    --n-gpu-layers -1                 # Apple Silicon Metal
```

It auto-generates an OpenAI-compatible API:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="tiny-tale",
    messages=[{"role":"user","content":"Once upon a time"}],
    temperature=0.8, max_tokens=120,
)
print(resp.choices[0].message.content)
```

Benefit: the **standard OpenAI client** works unchanged. Zero migration friction.

---

## 4. vLLM — Maximum GPU Throughput

```bash title="vllm_serve.sh"
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```

vLLM's key innovation — **PagedAttention** (Kwon et al., 2023):
- Manages KV cache in page-sized blocks
- Shares memory across concurrent requests
- **5–10× throughput** vs. naive serving

On an A100 80GB:

| Model | Concurrent users | Throughput (tok/s, total) |
|---|---:|---:|
| Qwen 0.5B | 100 | 5,000+ |
| Qwen 7B | 30 | 2,500+ |

**When to use vLLM**: you have a GPU and ≥ 30 concurrent users.

---

## 5. Batching and Concurrency

Two serving patterns:
- **Static batching**: fill one batch, run it to completion. Works for low concurrency.
- **Continuous batching** (vLLM/TGI): merges requests from different arrival times at every decode step. Throughput scales up.

```python title="latency_test.py" linenums="1"
import asyncio, httpx, time

async def request(client, prompt):
    r = await client.post("http://localhost:8000/v1/chat/completions", json={
        "model":"qwen", "messages":[{"role":"user","content":prompt}],
        "max_tokens":50,
    })
    return r.json()

async def benchmark(n_concurrent):
    async with httpx.AsyncClient() as client:
        t0 = time.time()
        results = await asyncio.gather(*[request(client, "test") for _ in range(n_concurrent)])
        dt = time.time() - t0
        print(f"  {n_concurrent} concurrent: {dt:.2f}s, p95 ~{dt*0.95:.2f}s")

asyncio.run(benchmark(1))    # baseline
asyncio.run(benchmark(10))   # concurrency impact
asyncio.run(benchmark(50))   # throughput ceiling
```

---

## 6. Health Checks · Graceful Shutdown · Adapter Hot-Swap

### Health check

```python title="health.py" linenums="1"
@app.get("/health")
async def health():
    try:
        resp = await client.chat.completions.create(
            model="qwen", messages=[{"role":"user","content":"ping"}],
            max_tokens=5, timeout=2.0)
        return {"status":"ok"}
    except: return {"status":"unhealthy"}, 503
```

Wire this into your Kubernetes/Docker liveness probe.

### Graceful shutdown

Block new requests → wait for in-flight requests to finish → exit. Use the Kubernetes preStop hook + SIGTERM handling.

### Adapter hot-swap (LoRA)

vLLM supports dynamic LoRA adapter loading:

```python
# Start vLLM with --enable-lora --max-loras 4
# Specify the adapter at inference time
resp = client.chat.completions.create(
    model="qwen-with-adapter-v2",          # adapter name
    messages=[...]
)
```

→ **Rollback in under 30 seconds** — just swap the adapter, no base model reload needed. This connects to the production cycle in Ch 32.

---

## 7. Common Failure Modes

1. **Forgetting `--n-gpu-layers -1`** — falls back to CPU on Apple Silicon; runs ~30× slower.
2. **ctx-size set too large** — memory blows up. 1024 is enough for this book's models.
3. **Never measuring concurrent users** — fast at 1 user, broken at 20. Measure production load.
4. **vLLM OOM** — `gpu-memory-utilization 0.9` is the standard. Leaves room for the KV cache.
5. **No health check** — the model dies and you don't know. Handle this at the Kubernetes/Docker level.
6. **No graceful shutdown** — user requests get cut off during deployments.
7. **Merging the adapter before hot-swap** — keep LoRA separate if you want to swap it live.
8. **Not validating OpenAI API compatibility** — test `/v1/chat/completions` directly against the real OpenAI SDK.

---

## 8. Operational Checklist

Serving gates:

- [ ] Stack chosen (llama.cpp / vLLM / TGI)
- [ ] Latency budget arithmetic (p50/p95)
- [ ] Single-request latency measured
- [ ] Load test at 10/50/100 concurrent users
- [ ] OpenAI-compatible API validated
- [ ] Health check endpoint
- [ ] Graceful shutdown
- [ ] (LoRA) Adapter hot-swap working
- [ ] Monitoring set up (Ch 32)

---

## 9. Exercises

1. Serve this book's GGUF model with `llama-server` and call it via the OpenAI SDK.
2. Measure p50/p95 for single request, 10 concurrent, and 50 concurrent users.
3. Run the same load test on vLLM (if you have a GPU). Compare throughput.
4. Load two adapters into vLLM simultaneously and route individual requests to different adapters.
5. **(Think about it)** An AICC call needs a summary within 1 second of hanging up. Your p95 budget is 1 s. Which stack can you use?

---

## References

- Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* arXiv:2309.06180
- llama.cpp `examples/server/` README
- HuggingFace TGI docs
- "Designing Data-Intensive Applications" (Kleppmann) — serving patterns
