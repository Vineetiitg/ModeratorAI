#!/usr/bin/env python3
"""
SafeChat — Download and Save Base HingGPT Model Locally

Downloads `l3cube-pune/hing-gpt` (Hinglish GPT-2 architecture) from Hugging Face
and stores it in `checkpoints/hing-gpt-base` for local generative detoxification.
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

MODEL_NAME = "l3cube-pune/hing-gpt"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "hing-gpt-base")

def main():
    print("="*80)
    print("📦 SAFECHAT: DOWNLOADING BASE HING-GPT MODEL LOCALLY")
    print("="*80)
    print(f"Target Model       : {MODEL_NAME}")
    print(f"Destination Folder : {OUTPUT_DIR}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/2] Downloading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT-2 does not have a pad token by default; assign eos_token as pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Tokenizer saved successfully.")

    print("\n[2/2] Downloading Causal LM Model Weights...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.save_pretrained(OUTPUT_DIR)
    print("✅ Model weights saved successfully.")

    print("\n" + "="*80)
    print(f"🎉 HingGPT model downloaded and saved locally at:\n    {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
