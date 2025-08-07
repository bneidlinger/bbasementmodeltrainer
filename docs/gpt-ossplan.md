# BasementBrewAI — GPT‑OSS Integration & Fine‑Tuning Guide

> “Because if you’re going to unleash Skynet, you might as well document it.” — HK‑47 & Rick

## Prerequisites

* **Hardware:** NVIDIA RTX 3070 Ti (12 GB VRAM), 64 GB system RAM.
* **OS:** Windows 10/11 or Linux (WSL 2 on Windows works).
* **Python:** 3.10 +
* Git, CUDA 12.x, cuDNN configured with PyTorch.
* Latest clone of **BasementBrewAI** in `~/Projects/bbasementmodeltrainer`.

---

## Step‑By‑Step

### 1. Branch & Environment

```bash
git checkout -b feature/gpt-oss
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip wheel
```

### 2. Dependencies for LLM Work

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.43.* datasets accelerate bitsandbytes peft sentencepiece flash-attn --no-build-isolation
```

*Tip:* If `flash‑attn` fails on Windows, append `--force-cuda-arch 8.6`.

### 3. Project Structure Additions

```
trainer/
 └── llm/
     ├── __init__.py
     ├── registry.py          # `register_llm` decorator
     ├── gptoss_wrapper.py    # model + tokenizer loader
     ├── train_llm.py         # training entry point
     ├── data/
     │    └── text_dataset_plugins.py
     └── utils/
          └── tokenizer_tools.py
```

### 4. Modify `run.py`

Add LLM option:

```python
from trainer.llm.registry import get_llm
```

and extend CLI with `--task {cv,text}`.

### 5. Tokenizer & Dataset

1. Place raw text files under `datasets/raw/`.
2. Wrap HF datasets in `text_dataset_plugins.py`:

```python
from datasets import load_dataset

dataset = load_dataset("text", data_files={"train": "*.txt"}, split="train")
```

3. Prepare dataset:

```bash
python scripts/prepare_text_ds.py --source datasets/raw --max_seq_len 2048
```

### 6. Fine‑Tune with QLoRA

Create `configs/gptoss_20b_qlora.yml`:

```yaml
model_name: openai/gpt-oss-20b
load_in_4bit: true
lora_rank: 64
batch_size: 4
grad_accum: 8
epochs: 4
lr: 2e-4
warmup_steps: 50
use_flash_attn: true
gradient_checkpointing: true
eval_steps: 1000
```

Launch training:

```bash
accelerate launch --config_file configs/single_gpu.yml \
  trainer/llm/train_llm.py --config configs/gptoss_20b_qlora.yml
```

### 7. GUI Updates (Dear PyGui)

* Add dropdown **Task Type** → “Image 🖼️ / Text 📝”.
* For **Text**, show inputs: *Max Seq Len*, *LoRA Rank*, *Guard‑rails 🔒*.
* Spawn training in a background `multiprocessing.Process` to keep UI responsive.

### 8. Inference Micro‑Service (Optional)

```bash
uvicorn services.inference_server:app --port 8008 --reload
```

*Endpoints*

* `POST /generate` → `{"prompt":"..."}` returns `{"completion":"..."}`

### 9. Safety Lever

Inside `gptoss_wrapper.py`:

```python
if not args.enable_guardrails:
    model.disable_safety_layers()  # here be dragons 🐉
```

Remember: uncheck the box—you own the fallout.

### 10. Commit & Push

```bash
git add .
git commit -m "feat: initial GPT‑OSS integration with QLoRA support"
git push origin feature/gpt-oss
```

---

## Appendix

| Resource            | Link                                                                                   |
| ------------------- | -------------------------------------------------------------------------------------- |
| GPT‑OSS 20B Weights | [https://huggingface.co/openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| QLoRA Paper         | [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)                   |

### Troubleshooting

* **Out‑of‑memory:** Lower `batch_size` or sequence length.
* **CUDA kernel errors:** Update NVIDIA driver, reinstall matching PyTorch build.

> ***FIN*** — Iterate until the basement AI overlords are satisfied. ☠️
