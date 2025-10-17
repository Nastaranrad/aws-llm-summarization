
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_model(checkpoint: str):
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.float32)
    model.eval()
    return tok, model

def summarize(text: str, tok, model, max_new_tokens=120, temperature=0.7):
    prompt = f"Summarize the following dialogue:\n\n{text}\n\nSummary:"
    inputs = tok(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, help="Dialogue text to summarize")
    ap.add_argument("--file", type=str, help="Path to a text file with 1 dialogue per line")
    ap.add_argument("--checkpoint", type=str, default="google/flan-t5-base", help="HF model or local checkpoint path")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    tok, model = load_model(args.checkpoint)

    if args.text:
        print(summarize(args.text, tok, model, args.max_new_tokens, args.temperature))
    elif args.file:
        p = Path(args.file)
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line: continue
            print(f"--- Example {i} ---")
            print(summarize(line, tok, model, args.max_new_tokens, args.temperature))
            print()
    else:
        ap.error("Provide --text or --file")

if __name__ == "__main__":
    main()
