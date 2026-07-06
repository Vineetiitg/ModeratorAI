#!/usr/bin/env python3
"""
SafeChat — Generative Detoxification Fine-Tuning Pipeline for HingGPT

Fine-tunes `l3cube-pune/hing-gpt` (local causal LLM) on our curated code-mixed Hinglish
and Devanagari parallel dataset (`detox_train.jsonl`). Learns to rewrite toxic chats
politely. Tracks and saves all training metrics (Loss & Perplexity) to JSON.
"""

import os
import sys
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

BASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "hing-gpt-base")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "hing-gpt-detox-finetuned")
TRAIN_FILE = os.path.join(os.path.dirname(__file__), "training", "data", "optimal_detox_train.jsonl")
VAL_FILE = os.path.join(os.path.dirname(__file__), "training", "data", "detox_val.jsonl")
METRICS_FILE = os.path.join(OUTPUT_DIR, "detox_metrics.json")

class DetoxDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        self.examples = []
        if tokenizer.pad_token is None:
            tokenizer.pad_token = '[PAD]' if '[PAD]' in tokenizer.get_vocab() else tokenizer.eos_token
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                prompt = f"Rewrite this toxic text politely without any abuse:\nToxic: {data['toxic_text']}\nClean: {data['clean_text']}{tokenizer.eos_token if tokenizer.eos_token else ''}"
                enc = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                input_ids = enc["input_ids"].squeeze(0)
                attention_mask = enc["attention_mask"].squeeze(0)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                self.examples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            total_steps += 1
    avg_loss = total_loss / max(total_steps, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl

def main():
    print("="*85)
    print("🚀 SAFECHAT: HING-GPT DETOXIFICATION FINE-TUNING PIPELINE")
    print("="*85)

    if not os.path.exists(BASE_MODEL_DIR):
        print(f"❌ Error: Base model directory not found at {BASE_MODEL_DIR}")
        print("Please run `python download_hinggpt.py` first!")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Hardware Acceleration Device : {device}")

    print("\n[1/4] Loading Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '[PAD]' if '[PAD]' in tokenizer.get_vocab() else tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR).to(device)

    print("[2/4] Loading Curated Parallel Detoxification Datasets...")
    train_dataset = DetoxDataset(TRAIN_FILE, tokenizer)
    val_dataset = DetoxDataset(VAL_FILE, tokenizer)
    print(f" -> Training Examples   : {len(train_dataset)}")
    print(f" -> Validation Examples : {len(val_dataset)}")

    batch_size = 4
    epochs = 3
    lr = 5e-5

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(10, int(0.1*total_steps)), num_training_steps=total_steps)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"\n[3/4] Starting Fine-Tuning ({epochs} Epochs | FP16 AMP Enabled: {use_amp})...")
    print("-" * 85)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Train PPL':<12} | {'Val Loss':<12} | {'Val PPL':<12}")
    print("-" * 85)

    metrics_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            steps += 1

        avg_train_loss = total_train_loss / max(steps, 1)
        train_ppl = math.exp(avg_train_loss) if avg_train_loss < 20 else float("inf")

        val_loss, val_ppl = evaluate_model(model, val_loader, device)

        print(f"{epoch:<6} | {avg_train_loss:<12.4f} | {train_ppl:<12.2f} | {val_loss:<12.4f} | {val_ppl:<12.2f}")

        metrics_history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_perplexity": round(train_ppl, 2),
            "val_loss": round(val_loss, 4),
            "val_perplexity": round(val_ppl, 2)
        })

    print("-" * 85)
    print("[4/4] Saving Fine-Tuned Model Checkpoint and Training Metrics...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump({"hyperparameters": {"epochs": epochs, "batch_size": batch_size, "lr": lr}, "epochs": metrics_history}, f, indent=2)

    print(f"✅ Model saved to   : {OUTPUT_DIR}")
    print(f"📊 Metrics saved to : {METRICS_FILE}")
    print("="*85)

if __name__ == "__main__":
    main()
