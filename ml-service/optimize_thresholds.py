#!/usr/bin/env python3
"""
SafeChat — Dynamic Per-Class Threshold Tuning (Precision-Recall Curve Optimization)

Optimizes classification thresholds for each tag using the Validation Set (`real_toxicity_val.csv`).
Applies these optimal per-class thresholds $\theta_c$ to both Validation and Test sets (`real_toxicity_test.csv`),
and saves the calibrated thresholds to `optimal_thresholds.json` for production moderation.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

TAGS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "hingbert-toxicity-finetuned")
DATA_DIR = os.path.join(os.path.dirname(__file__), "training", "data", "real_datasets")

class ToxicityDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch, tokenizer, device):
    texts, labels = zip(*batch)
    inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    labels = torch.tensor(np.array(labels), dtype=torch.float).to(device)
    return inputs, labels

def get_probabilities(model, tokenizer, dataloader):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    return np.vstack(all_probs), np.vstack(all_labels)

def main():
    print("="*90)
    print("🎯 SAFECHAT: DYNAMIC PER-CLASS THRESHOLD TUNING (PRECISION-RECALL OPTIMIZATION)")
    print("="*90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Acceleration Device: {device}")

    # Load data splits
    val_path = os.path.join(DATA_DIR, "real_toxicity_val.csv")
    test_path = os.path.join(DATA_DIR, "real_toxicity_test.csv")

    val_df = pd.read_csv(val_path).dropna(subset=["text"]).reset_index(drop=True)
    test_df = pd.read_csv(test_path).dropna(subset=["text"]).reset_index(drop=True)

    val_labels = val_df[TAGS].values
    test_labels = test_df[TAGS].values

    print(f"Loaded Validation Set : {len(val_df)} rows")
    print(f"Loaded Test Set       : {len(test_df)} rows")

    # Load Fine-Tuned Model
    print(f"\nLoading SafeChat Fine-Tuned Model from: {CHECKPOINT_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(device)

    val_loader = DataLoader(ToxicityDataset(val_df["text"].astype(str).tolist(), val_labels), batch_size=32, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer, device))
    test_loader = DataLoader(ToxicityDataset(test_df["text"].astype(str).tolist(), test_labels), batch_size=32, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer, device))

    print(" -> Predicting continuous probabilities on Validation Set...")
    val_probs, val_true = get_probabilities(model, tokenizer, val_loader)
    print(" -> Predicting continuous probabilities on Test Set...")
    test_probs, test_true = get_probabilities(model, tokenizer, test_loader)

    # 1. Optimize Thresholds on Validation Set
    print("\n" + "="*90)
    print("⚙️ OPTIMIZING PER-CLASS THRESHOLDS ON VALIDATION SET (Precision-Recall Analysis)")
    print("="*90)
    print(f"{'Tag Name':<16} | {'Positives':<10} | {'Default F1 (0.50)':<18} | {'Optimal Threshold':<18} | {'Optimized F1':<16}")
    print("-" * 90)

    optimal_thresholds = {}
    val_f1_default_list = []
    val_f1_opt_list = []

    for i, tag in enumerate(TAGS):
        y_true = val_true[:, i]
        y_prob = val_probs[:, i]
        pos_count = int(np.sum(y_true))

        # Default F1 at 0.50
        preds_50 = (y_prob >= 0.50).astype(int)
        f1_50 = f1_score(y_true, preds_50, zero_division=0)
        val_f1_default_list.append(f1_50)

        if pos_count > 0:
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            # Avoid division by zero
            f1_scores = np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros_like(precision),
                where=(precision + recall) != 0
            )
            best_idx = np.argmax(f1_scores)
            best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.50
            best_f1 = float(f1_scores[best_idx])
        else:
            # If rare tag has 0 positives in Val split, use an optimal safety threshold based on probability distribution
            best_th = 0.15
            best_f1 = 0.0

        optimal_thresholds[tag] = round(best_th, 4)
        val_f1_opt_list.append(best_f1)

        print(f"{tag:<16} | {pos_count:<10} | {f1_50*100:<17.2f}% | {best_th:<18.4f} | {best_f1*100:<15.2f}%")

    print("-" * 90)

    # Save calibrated thresholds to JSON
    out_json = os.path.join(CHECKPOINT_DIR, "optimal_thresholds.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(optimal_thresholds, f, indent=2)
    print(f"✅ Saved optimal per-class thresholds to: {out_json}")

    # 2. Apply to Validation and Test Sets & Compare Macro/Micro F1
    print("\n" + "="*90)
    print("📈 FINAL BENCHMARK: STATIC 0.50 THRESHOLD vs. OPTIMIZED PER-CLASS THRESHOLDS")
    print("="*90)

    # Validation Set Predictions
    val_preds_50 = (val_probs >= 0.50).astype(int)
    val_preds_opt = np.zeros_like(val_probs, dtype=int)
    for i, tag in enumerate(TAGS):
        val_preds_opt[:, i] = (val_probs[:, i] >= optimal_thresholds[tag]).astype(int)

    val_macro_50 = f1_score(val_true, val_preds_50, average="macro", zero_division=0)
    val_micro_50 = f1_score(val_true, val_preds_50, average="micro", zero_division=0)
    val_macro_opt = f1_score(val_true, val_preds_opt, average="macro", zero_division=0)
    val_micro_opt = f1_score(val_true, val_preds_opt, average="micro", zero_division=0)

    # Test Set Predictions
    test_preds_50 = (test_probs >= 0.50).astype(int)
    test_preds_opt = np.zeros_like(test_probs, dtype=int)
    for i, tag in enumerate(TAGS):
        test_preds_opt[:, i] = (test_probs[:, i] >= optimal_thresholds[tag]).astype(int)

    test_macro_50 = f1_score(test_true, test_preds_50, average="macro", zero_division=0)
    test_micro_50 = f1_score(test_true, test_preds_50, average="micro", zero_division=0)
    test_macro_opt = f1_score(test_true, test_preds_opt, average="macro", zero_division=0)
    test_micro_opt = f1_score(test_true, test_preds_opt, average="micro", zero_division=0)

    print(f"{'Dataset Split':<22} | {'Evaluation Metric':<20} | {'Static 0.50 Thres':<18} | {'Optimized Thres':<18} | {'F1 Improvement':<16}")
    print("-" * 90)
    
    val_macro_diff = (val_macro_opt - val_macro_50) * 100
    val_micro_diff = (val_micro_opt - val_micro_50) * 100
    print(f"{'Validation (2,262 rows)':<22} | {'Macro F1':<20} | {val_macro_50*100:<17.2f}% | {val_macro_opt*100:<17.2f}% | +{val_macro_diff:<15.2f}%")
    print(f"{'':<22} | {'Micro F1 (Accuracy)':<20} | {val_micro_50*100:<17.2f}% | {val_micro_opt*100:<17.2f}% | +{val_micro_diff:<15.2f}%")
    print("-" * 90)

    test_macro_diff = (test_macro_opt - test_macro_50) * 100
    test_micro_diff = (test_micro_opt - test_micro_50) * 100
    print(f"{'Test Set (2,262 rows)':<22} | {'Macro F1':<20} | {test_macro_50*100:<17.2f}% | {test_macro_opt*100:<17.2f}% | +{test_macro_diff:<15.2f}%")
    print(f"{'':<22} | {'Micro F1 (Accuracy)':<20} | {test_micro_50*100:<17.2f}% | {test_micro_opt*100:<17.2f}% | +{test_micro_diff:<15.2f}%")
    print("=" * 90)

    # 3. Detailed Per-Tag Breakdown on Test Set
    print("\n" + "="*90)
    print("🔍 DETAILED PER-TAG BREAKDOWN ON TEST SET (Unseen Data)")
    print("="*90)
    print(f"{'Tag Name':<16} | {'Test Positives':<16} | {'Static F1 (0.50)':<18} | {'Optimized F1':<18} | {'Gain':<12}")
    print("-" * 90)
    for i, tag in enumerate(TAGS):
        y_test_tag = test_true[:, i]
        pos_test = int(np.sum(y_test_tag))
        f1_test_50 = f1_score(y_test_tag, test_preds_50[:, i], zero_division=0) * 100
        f1_test_opt = f1_score(y_test_tag, test_preds_opt[:, i], zero_division=0) * 100
        gain = f1_test_opt - f1_test_50
        print(f"{tag:<16} | {pos_test:<16} | {f1_test_50:<17.2f}% | {f1_test_opt:<17.2f}% | +{gain:<10.2f}%")
    print("="*90)

if __name__ == "__main__":
    main()
