#!/usr/bin/env python3
"""
SafeChat — Live Interactive Toxicity Inference CLI Demo

Loads the fine-tuned `hingbert-toxicity-finetuned` multi-label model and calibrated
optimal thresholds. Allows users/reviewers to type custom chats in English, Devanagari Hindi,
or Romanized Hinglish to see real-time content moderation predictions!
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure UTF-8 console output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

TAGS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "hingbert-toxicity-finetuned")
THRESHOLDS_PATH = os.path.join(CHECKPOINT_DIR, "optimal_thresholds.json")

SAMPLE_PROMPTS = [
    "Bhai aaj ka live stream bahut mast tha! Great job team!",
    "नमस्ते सर, क्या आप मुझे कल के मीटिंग का समय भेज सकते हैं?",
    "tu ek number ka bewakoof aur gadhha hai, nikal yahan se",
    "shut up you absolute motherfucking piece of sh1t, go to hell",
    "bhenchod bakwas mat kar warna accha nahi hoga harami saale chutiya",
    "तुझे घर से निकाल कर गोली मार दूंगा, आज रात तेरा आखिरी दिन है"
]

def render_bar(prob, width=25):
    filled = int(prob * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return f"[{bar}]"

def main():
    print("=" * 80)
    print("🛡️  SAFECHAT: LIVE INTERACTIVE CONTENT MODERATION CLI DEMO")
    print("    Fine-Tuned Hing-RoBERTa Multi-Label Classification Engine")
    print("=" * 80)

    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Error: Model checkpoint not found at: {CHECKPOINT_DIR}")
        print("Please run `python train_hingbert_toxicity.py` first!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Hardware Acceleration Device: {device}")
    print(f"📦 Loading model weights from: {CHECKPOINT_DIR}...")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(device)
    model.eval()

    thresholds = {}
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
        print(f"🎯 Calibrated Optimal Thresholds Loaded: {thresholds}")
    else:
        thresholds = {tag: 0.50 for tag in TAGS}
        print("⚠️  Warning: optimal_thresholds.json not found. Using default 0.50 thresholds.")

    print("\n" + "=" * 80)
    print("💡 INSTRUCTIONS: Type any text in English, Hindi, or Hinglish and press Enter.")
    print("   Type 'samples' to run quick built-in test messages.")
    print("   Type 'exit' or 'quit' to close the demo.")
    print("=" * 80)

    while True:
        try:
            print("\n" + "-" * 80)
            user_input = input("💬 Enter message to moderate > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting SafeChat Demo. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting SafeChat Demo. Goodbye!")
            break

        messages_to_test = []
        if user_input.lower() == "samples":
            messages_to_test = SAMPLE_PROMPTS
            print("\n🚀 Running 6 built-in benchmark samples...")
        else:
            messages_to_test = [user_input]

        for idx, text in enumerate(messages_to_test, 1):
            if len(messages_to_test) > 1:
                print(f"\n--- Sample #{idx} ---")
                print(f"💬 Message: \"{text}\"")

            inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            triggered_tags = []
            for tag, p in zip(TAGS, probs):
                th = thresholds.get(tag, 0.50)
                if p >= th:
                    triggered_tags.append(f"{tag.upper()} ({p*100:.1f}%)")

            if not triggered_tags:
                badge = "✅ SAFE / CLEAN"
                color_code = "\033[92m" # Green
            else:
                badge = f"🚨 TOXIC VIOLATION DETECTED -> {', '.join(triggered_tags)}"
                color_code = "\033[91m" # Red
            reset_code = "\033[0m"

            print(f"\n🛡️  MODERATION RESULT: {badge}")
            print("📊 Probability Distribution across 6 Tags:")
            for tag, p in zip(TAGS, probs):
                th = thresholds.get(tag, 0.50)
                status_symbol = "⚠️ " if p >= th else "  "
                bar_str = render_bar(p)
                print(f"   {status_symbol}{tag:<14} : {bar_str} {p*100:5.1f}% (Threshold: {th*100:4.0f}%)")

if __name__ == "__main__":
    main()
