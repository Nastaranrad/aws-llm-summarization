
# Dialogue Summarization with LLMs (FLAN-T5 + LoRA) — AWS Portfolio

This repository demonstrates **text summarization with Large Language Models (LLMs)** using the
**DialogSum** dataset and a **FLAN‑T5** model.

> Highlights
> - Zero‑shot baseline with `google/flan-t5-base`
> - Full fine‑tuning (Trainer) *and* PEFT/LoRA fine‑tuning path
> - ROUGE evaluation vs. ground truth
> - Ready‑to‑use CLI for inference
> - AWS‑friendly: works in a SageMaker Notebook / Studio or locally

---

## 🧱 Project Structure

```
aws-llm-summarization-portfolio/
├── README.md
├── requirements.txt
├── infer.py                  # Run summaries from a file or prompt
├── sample_dialog.txt         # Example input
└── notebooks/                # (Optional) put course lab notebook here
```

---

## 📦 Setup

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

> If using **SageMaker**, you can install requirements directly in the notebook kernel.

---

## 🚀 Quick Start (Inference)

Run a quick summary using the base FLAN‑T5 model:

```bash
python infer.py --text "Alice and Bob discussed travel plans. They agreed to meet Friday and book tickets."
```

Or summarize a file with multiple lines (one example per line):

```bash
python infer.py --file sample_dialog.txt
```

> To use a **fine‑tuned** checkpoint, pass `--checkpoint /path/to/checkpoint`.

---

## 📚 Dataset

- **DialogSum** (`knkarthick/dialogsum`) from Hugging Face Datasets.
- Each record contains a `dialogue` and a human‑written `summary`.

You can reproduce the dataset loading in your notebook like:

```python
from datasets import load_dataset
dataset = load_dataset("knkarthick/dialogsum")
```

---

## 🔧 Baseline & Training (What the lab does)

1. **Zero‑shot baseline** with `google/flan-t5-base`
   - Prepend an instruction like: `"Summarize the following dialogue:"`
   - Generate with beam search or sampling (temperature).

2. **Full Fine‑Tuning** (Hugging Face `Trainer`)
   - Tokenize inputs (`dialogue`) and labels (`summary`)
   - Train with a small number of epochs to keep costs low
   - Save checkpoint locally or to S3 (if in AWS)

3. **PEFT/LoRA Fine‑Tuning**
   - Wrap the base model with LoRA adapters
   - Train adapters only (much faster/cheaper)
   - Load adapters for inference

4. **Evaluation with ROUGE**
   - Compute ROUGE‑1/2/L on test samples
   - Compare *original model* vs *fine‑tuned* vs *LoRA*

> In the Coursera lab, you’ll see all of the above pieces as separate cells. This repo gives you a **clean way** to present the same skills publicly.

---

## 🧪 Reproduce (locally or SageMaker)

**Zero‑shot & evaluation** can be reproduced fully locally. For fine‑tuning, prefer a GPU (or SageMaker).

High‑level steps:
1. Load dataset and split
2. Tokenize with the FLAN‑T5 tokenizer (`t5` family uses `input_ids`/`labels`)
3. Train either:
   - Full fine‑tuning with `Trainer`
   - PEFT/LoRA with `peft.get_peft_model`
4. Save checkpoint to `./checkpoints/...` (or S3 in SageMaker)
5. Evaluate ROUGE with `evaluate`

---

## 📈 Portfolio Tips

- Commit **notebook + results** (`media/` screenshots, ROUGE table).
- In the README, include:
  - **Architecture/flow** diagram (dataset → tokenizer → model → evaluation)
  - A table with **ROUGE** before/after fine‑tuning
  - **Costs & runtime** notes (and how you optimized them)
- Add a short **demo video** (GIF/MP4) of `infer.py` running.

---

## 🧩 Example Prompts

```
Summarize the following dialogue in 2-3 sentences. Focus on key decisions and outcomes.

[dialogue text here]
```

```
Provide a concise meeting summary with action items and dates.

[dialogue text here]
```

---

## ✅ License

MIT
# aws-llm-summarization
# aws-llm-summarization
