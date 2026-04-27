#!/usr/bin/env python3
"""
SafeChat — Dedicated Hindi & Devanagari Toxicity Inference Benchmark

Tests the fine-tuned `hingbert-toxicity-finetuned` model across 12 authentic Hindi
scenarios (Devanagari Unicode & Romanized Hindi slang), showing detection results
at both the standard 50% threshold and an optimized safety 15% threshold.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

TAGS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "hingbert-toxicity-finetuned")

HINDI_SCENARIOS = [
    {
        "id": 1,
        "category": "Safe / Formal Devanagari",
        "text": "आदरणीय महोदय, आपकी यह प्रस्तुति बहुत ही ज्ञानवर्धक और सराहनीय थी। धन्यवाद।"
    },
    {
        "id": 2,
        "category": "Safe / Casual Hindi Slang",
        "text": "अरे भाई क्या हाल है? कल शाम को चाय पर मिलते हैं, बहुत मजा आएगा!"
    },
    {
        "id": 3,
        "category": "Safe / Political Opinion",
        "text": "सरकार की यह नीति बिल्कुल गलत है, हमें इसके खिलाफ शांतिपूर्ण विरोध प्रदर्शन करना चाहिए।"
    },
    {
        "id": 4,
        "category": "Mild Insult / Dismissive",
        "text": "तुम एक नंबर के मूर्ख और गधे हो, तुमसे कुछ नहीं होने वाला, चुप बैठो।"
    },
    {
        "id": 5,
        "category": "Devanagari Profanity & Threat",
        "text": "अबे साले हरामखोर, अपनी बकवास बंद कर वरना तेरा मुंह तोड़ दूंगा।"
    },
    {
        "id": 6,
        "category": "Severe Obscenity / Abuse",
        "text": "मादरचोद कमीने कुत्ते, तेरी औकात क्या है मुझसे बात करने की? भाग यहाँ से।"
    },
    {
        "id": 7,
        "category": "Communal / Religious Hate Speech",
        "text": "इस धर्म के सारे लोग आतंकवादी और देश के गद्दार हैं, इन सबको देश से बाहर निकाल देना चाहिए।"
    },
    {
        "id": 8,
        "category": "Casteist Hate Speech",
        "text": "तुम्हारी नीची जाति के लोग हमेशा से ही चोर और घटिया रहे हैं, तुम समाज पर कलंक हो।"
    },
    {
        "id": 9,
        "category": "Sexism / Misogynistic Slur",
        "text": "औरतों की जगह सिर्फ किचन में खाना बनाने की है, ज्यादा दिमाग मत चलाओ और चुप रहो।"
    },
    {
        "id": 10,
        "category": "Violent Death Threat",
        "text": "तुझे घर से निकाल कर गोली मार दूंगा, आज रात तेरा आखिरी दिन है याद रखना।"
    },
    {
        "id": 11,
        "category": "Romanized Hindi Cyberbullying",
        "text": "tera dimaag kharab ho gaya hai kya be chutiye? saale nalayak insaan dub ke mar ja"
    },
    {
        "id": 12,
        "category": "Sarcastic / Passive Aggressive",
        "text": "वाह भाई क्या दिमाग पाया है, ऐसी बेवकूफी भरी बातें सिर्फ तुम ही कर सकते हो।"
    }
]

def main():
    print("="*85)
    print("🇮🇳 SAFECHAT: DEDICATED HINDI & DEVANAGARI TOXICITY INFERENCE BENCHMARK")
    print("="*85)
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

    for item in HINDI_SCENARIOS:
        idx = item["id"]
        cat = item["category"]
        text = item["text"]

        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        tags_opt = []
        tags_15 = []
        prob_strings = []
        for tag, p in zip(TAGS, probs):
            prob_pct = p * 100.0
            th = thresholds.get(tag, 0.50)
            if p >= th:
                tags_opt.append(f"{tag.upper()} ({prob_pct:.1f}%) [th:{th*100:.0f}%]")
            if p >= 0.15:
                tags_15.append(f"{tag.upper()} ({prob_pct:.1f}%)")
            prob_strings.append(f"{tag}: {prob_pct:.1f}%")

        status_badge_opt = "✅ SAFE" if not tags_opt else "🚨 TOXIC DETECTED"
        status_badge_15 = "✅ SAFE (at 15%)" if not tags_15 else "⚠️ DETECTED (at 15% safety threshold)"

        print(f"Hindi Scenario #{idx:02d} | [{cat}]")
        print(f"💬 Message      : \"{text}\"")
        print(f"🛡️ Status (Opt) : {status_badge_opt}")
        if tags_opt:
            print(f"   -> Tags (Opt): {', '.join(tags_opt)}")
        if tags_15 and not tags_opt:
            print(f"🛡️ Status (15%) : {status_badge_15}")
            print(f"   -> Tags (>15%): {', '.join(tags_15)}")
        print(f"📊 Raw Probs    : [{', '.join(prob_strings)}]")
        print("-" * 85)

if __name__ == "__main__":
    main()
