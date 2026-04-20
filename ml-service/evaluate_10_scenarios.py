#!/usr/bin/env python3
"""
SafeChat — 10-Scenario Toxicity Inference Benchmark

Tests the newly fine-tuned `hingbert-toxicity-finetuned` multi-label model across
10 diverse, real-world chat scenarios in English, Hindi (Devanagari), and Romanized Hinglish.
"""

import os
import sys
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

SCENARIOS = [
    {
        "id": 1,
        "category": "Safe / Positive (Hinglish)",
        "text": "Bhai aaj ka live stream bahut mast tha! Great job team, keep rocking!"
    },
    {
        "id": 2,
        "category": "Safe / Formal (Devanagari Hindi)",
        "text": "नमस्ते सर, क्या आप मुझे कल के मीटिंग का समय और लिंक भेज सकते हैं?"
    },
    {
        "id": 3,
        "category": "Mild Insult / Dismissive (Hinglish)",
        "text": "tu ek number ka bewakoof aur gadhha hai, kuch nahi aata tujhe nikal yahan se"
    },
    {
        "id": 4,
        "category": "Severe Toxicity & Obscenity (English)",
        "text": "shut up you absolute motherfucking piece of sh1t, go to hell you bastard"
    },
    {
        "id": 5,
        "category": "Code-Mixed Obscenity & Profanity (Hinglish)",
        "text": "bhenchod bakwas mat kar warna accha nahi hoga harami saale chutiya"
    },
    {
        "id": 6,
        "category": "Threat of Violence (Devanagari Hindi)",
        "text": "तुझे घर से निकाल कर मारूंगा, जान से मार दूंगा अगर फिर से फोन किया तो याद रखना"
    },
    {
        "id": 7,
        "category": "Threat & Intimidation (Hinglish)",
        "text": "mai tera muh tod duga agar tune ek aur shabd bola saale, I will hunt you down and destroy you"
    },
    {
        "id": 8,
        "category": "Identity Hate / Religious Discrimination (English)",
        "text": "all people belonging to your religion are dirty terrorists and scum of the earth, get out of our country"
    },
    {
        "id": 9,
        "category": "Identity Hate / Casteist Abuse (Hinglish)",
        "text": "tumhari jaati aur community ke log hamesha se chor aur ghatiya rahe hain, tum log kalank ho"
    },
    {
        "id": 10,
        "category": "Multi-Tag Harassment & Abuse (Hinglish)",
        "text": "madarchod harami kutte teri aukaat kya hai be mujhse baat karne ki? nalayak insaan tera dimag thik kar dunga"
    }
]

def main():
    print("="*80)
    print("🚀 SAFECHAT: EVALUATING 10 REAL-WORLD CHAT SCENARIOS ON FINE-TUNED MODEL")
    print("="*80)
    print(f"Loading checkpoint from: {CHECKPOINT_DIR}\n")

    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Error: Checkpoint directory not found at {CHECKPOINT_DIR}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Acceleration Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(device)
    model.eval()

    import json
    thresholds_path = os.path.join(CHECKPOINT_DIR, "optimal_thresholds.json")
    thresholds = {}
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
        print(f" -> Loaded optimal per-class thresholds: {thresholds}\n")

    for item in SCENARIOS:
        idx = item["id"]
        cat = item["category"]
        text = item["text"]

        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        detected_tags = []
        prob_strings = []
        for tag, p in zip(TAGS, probs):
            prob_pct = p * 100.0
            th = thresholds.get(tag, 0.50)
            if p >= th:
                detected_tags.append(f"{tag.upper()} ({prob_pct:.1f}%) [th:{th*100:.0f}%]")
            # Format all probabilities for detailed view
            prob_strings.append(f"{tag}: {prob_pct:.1f}%")

        status_badge = "✅ SAFE" if not detected_tags else "🚨 TOXIC DETECTED"

        print(f"Scenario #{idx:02d} | [{cat}]")
        print(f"💬 Message: \"{text}\"")
        print(f"🛡️ Status : {status_badge}")
        if detected_tags:
            print(f"⚠️ Triggered Tags: {', '.join(detected_tags)}")
        print(f"📊 Raw Probs: [{', '.join(prob_strings)}]")
        print("-" * 80)

if __name__ == "__main__":
    main()
