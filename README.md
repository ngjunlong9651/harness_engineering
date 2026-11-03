# Harness Engineering: learn by building

**Agent = Model + Harness.** This course builds a small agent harness (loop, tools, policy, sandbox, context management, long-running handoff) and lets you read, run and break it.

## Contents
| File | What it is |
|---|---|
| `harness_engineering_learning.ipynb` | The lecture notebook (concepts, labs, checkpoints, exercises, glossary, reading list) |
| `harness.py` | The harness itself, about 200 lines, heavily commented |
| `requirements.txt` | numpy, pandas, matplotlib, ipykernel |

Everything runs **offline**: the "model" is scripted so results are deterministic. The notebook shows how to swap in a real LLM.

## Slides
A companion slide deck lives on the author's Claude account (private link, may not open for you): https://claude.ai/artifact/Me61gX9Et9msRCXVeN8iZR

## Setup
```bash
uv venv .venv
uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name harness-eng --display-name "Python (harness-eng)"
```
Open the notebook and select the **Python (harness-eng)** kernel.

## Topics
The agent loop, tool design, policy and sandboxing, verification as feedback, context engineering (truncation, masking, compaction), long-running agents with progress artifacts, debugging from traces.

## Series
Part of a small AI engineering learning series: Hybrid RAG, Harness Engineering, LLM Evals, Tool Use and MCP.
