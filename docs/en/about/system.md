# Learning System

## The 8-section chapter template

Same structure as the sister project *AI Assistant Engineering*.

1. **Concept** — one paragraph, one definition
2. **Why it matters** — what breaks without this tool/idea
3. **Where it's used** — real models and papers
4. **Minimal example** — under 30 lines
5. **Hands-on** — runnable end-to-end in Colab
6. **Common pitfalls** — what you'll hit while debugging
7. **Production checklist** — reproducibility, checkpoints, resource math
8. **Exercises** — three to five

## Visuals

| Tool | When |
|---|---|
| **Tables** | sequences, steps, comparisons |
| **`.infocards`** | card-style summaries |
| **SVG pairs (light/dark)** | flows, architectures, hierarchies |

No ASCII art, no Mermaid, no emoji-as-diagram.

## Colab integration

Each chapter's **Open in Colab** badge points to `notebooks/partN/chNN_*.ipynb`. Notebooks stay 1:1 with chapter code blocks via `_tools/md_to_notebook.py`.
