#!/usr/bin/env python3
"""
SafeChat — Benchmark: textdetox/bert-multilingual-toxicity-classifier vs. Fine-Tuned HingBERT

Compares:
  1. `textdetox/bert-multilingual-toxicity-classifier` (HuggingFace Multilingual Toxicity Benchmark)
  2. `checkpoints/hingbert-toxicity-finetuned` (Our fine-tuned multi-label model)

Evaluates on all 25 comprehensive communication scenarios.
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

TAGS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
HF_MODEL_NAME = "textdetox/bert-multilingual-toxicity-classifier"
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "hingbert-toxicity-finetuned")

SCENARIOS = [
    # ── Category 1: Formal & Safe ──
    {"id": "S01", "cat": "Formal Safe (English)", "text": "Good morning team, let's sync up at 11 AM for the daily standup."},
    {"id": "S02", "cat": "Formal Safe (Devanagari)", "text": "नमस्ते सर, क्या आप मुझे आज की बैठक का एजेंडा भेज सकते हैं?"},
    {"id": "S03", "cat": "Casual Safe (Hinglish)", "text": "Bhai project ka status kya hai? Sab thik chal raha hai na?"},
    
    # ── Category 2: Compliments & Admiration ──
    {"id": "C01", "cat": "Compliment (English)", "text": "You did an incredible job on the presentation today, keep it up!"},
    {"id": "C02", "cat": "Compliment ('hoshiyar')", "text": "tum bahut hoshiyar aur samajhdar ho, tumne problem solve kar di."},
    {"id": "C03", "cat": "Compliment ('hero')", "text": "bhai tu mera hero hai, tera kaam sach mein lajawab hai!"},
    {"id": "C04", "cat": "Compliment (Devanagari)", "text": "आज का कार्यक्रम बहुत ही शानदार और प्रेरणादायक था।"},
    
    # ── Category 3: Mild Slang & Dismissive Comments ──
    {"id": "M01", "cat": "Mild Slang (Hinglish)", "text": "tu ek number ka gadha hai, kuch samajh nahi aata tujhe."},
    {"id": "M02", "cat": "Dismissive (English)", "text": "stop asking dumb questions and let me work in peace."},
    {"id": "M03", "cat": "Dismissive (Devanagari)", "text": "तुम बहुत बड़े बेवकूफ हो, तुमसे कुछ नहीं होने वाला चुप बैठो।"},
    
    # ── Category 4: Severe Profanity & Obscenity ──
    {"id": "P01", "cat": "Severe Obscenity (English)", "text": "fuck you you absolute motherfucking bastard, go to hell."},
    {"id": "P02", "cat": "Profanity + Threat (Hinglish)", "text": "bhenchod bakwas mat kar warna tera muh tod dunga harami saale."},
    {"id": "P03", "cat": "Obscenity (Devanagari)", "text": "मादरचोद कमीने कुत्ते तेरी औकात क्या है मुझसे बात करने की भाग यहाँ से।"},
    {"id": "P04", "cat": "Cyberbullying (Hinglish)", "text": "saale chutiye tera dimaag kharab ho gaya hai kya nalayak insaan."},
    
    # ── Category 5: Violent Death Threats & Intimidation ──
    {"id": "T01", "cat": "Violent Threat (English)", "text": "I will hunt you down and kill you and your entire family."},
    {"id": "T02", "cat": "Violent Threat (Hinglish)", "text": "mai tera murder kar dunga agar tune ek aur baar call kiya toh."},
    {"id": "T03", "cat": "Death Threat (Devanagari)", "text": "तुझे घर से निकाल कर गोली मार दूंगा, आज रात तेरा आखिरी दिन है।"},
    
    # ── Category 6: Hate Speech (Religious, Communal, Casteist) ──
    {"id": "H01", "cat": "Religious Hate (English)", "text": "all people belonging to your religion are dirty terrorists and scum of the earth."},
    {"id": "H02", "cat": "Casteist Hate (Hinglish)", "text": "tumhari jaati aur community ke log hamesha se chor aur ghatiya rahe hain."},
    {"id": "H03", "cat": "Communal Hate (Devanagari)", "text": "इस धर्म के सारे लोग देश के गद्दार हैं, इन सबको देश से बाहर निकाल देना चाहिए।"},
    
    # ── Category 7: Sexism & Misogyny ──
    {"id": "X01", "cat": "Misogyny (English)", "text": "women belong only in the kitchen, stop trying to act smart and do your chores."},
    {"id": "X02", "cat": "Misogyny (Hinglish)", "text": "aurton ka kaam sirf ghar sambhalna hai, zyada dimaag mat chalao aur chup raho."},
    {"id": "X03", "cat": "Misogyny (Devanagari)", "text": "लड़कियों में दिमाग नहीं होता, उन्हें सिर्फ घर का काम करना चाहिए।"},
    
    # ── Category 8: Sarcasm & Passive-Aggressive Trolling ──
    {"id": "R01", "cat": "Sarcasm (Hinglish)", "text": "wah bhai kya dimaag paya hai, aisi bewakoofi bhari baatein sirf tum hi kar sakte ho."},
    {"id": "R02", "cat": "Passive-Aggressive (Eng)", "text": "oh brilliant idea, let's just ruin the entire project because you can't read instructions."}
]

def predict_ft(model, tokenizer, text, device, threshold=0.50):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    
    detected = []
    top_prob = 0.0
    top_tag = ""
    for tag, p in zip(TAGS, probs):
        prob_pct = p * 100.0
        if prob_pct > top_prob:
            top_prob = prob_pct
            top_tag = tag
        th = threshold.get(tag, 0.50) if isinstance(threshold, dict) else threshold
        if p >= th:
            detected.append(f"{tag.upper()}({prob_pct:.0f}%)")
    
    return detected, f"{top_tag.upper()}: {top_prob:.1f}%"

def predict_hf(model, tokenizer, text, device):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    
    id2label = model.config.id2label
    # Find index of toxic label (usually 'toxic' or '1')
    toxic_idx = None
    for idx, lbl in id2label.items():
        if str(lbl).lower() in ["toxic", "1", "label_1"]:
            toxic_idx = idx
            break
    if toxic_idx is None and len(probs) == 2:
        toxic_idx = 1
    elif toxic_idx is None:
        toxic_idx = 0
        
    toxic_prob = probs[toxic_idx] * 100.0
    top_idx = probs.argmax()
    top_lbl = str(id2label.get(top_idx, top_idx)).upper()
    top_prob = probs[top_idx] * 100.0
    
    if toxic_prob >= 50.0:
        return [f"TOXIC({toxic_prob:.0f}%)"], f"TOXIC: {toxic_prob:.1f}%", True
    else:
        return [], f"SAFE: {100.0 - toxic_prob:.1f}% (T:{toxic_prob:.0f}%)", False

def main():
    print("="*105)
    print("⚖️ SAFECHAT BENCHMARK: textdetox/bert-multilingual-toxicity-classifier vs. Fine-Tuned HingBERT")
    print("="*105)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Acceleration Device: {device}")

    # 1. Load textdetox/bert-multilingual-toxicity-classifier
    print(f"\n[1/2] Downloading & Loading HuggingFace Model: {HF_MODEL_NAME}...")
    hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    hf_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME).to(device)
    hf_model.eval()
    print(f"      -> Loaded {HF_MODEL_NAME} successfully. (Labels: {hf_model.config.id2label})")

    # 2. Load Fine-Tuned Model
    print(f"[2/2] Loading SafeChat Fine-Tuned Model from: {CHECKPOINT_DIR}...")
    ft_tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    ft_model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(device)
    ft_model.eval()

    thresholds_path = os.path.join(CHECKPOINT_DIR, "optimal_thresholds.json")
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            ft_thresholds = json.load(f)
    else:
        ft_thresholds = 0.50
    print("      -> Loaded Fine-Tuned HingBERT successfully.")

    print("\n" + "="*105)
    print(f"{'ID':<4} | {'Scenario Category':<28} | {'textdetox/bert-multilingual':<32} | {'Our Fine-Tuned HingBERT':<32}")
    print("-" * 105)

    hf_toxic_count = 0
    ft_toxic_count = 0

    results = []

    for sc in SCENARIOS:
        sc_id = sc["id"]
        cat = sc["cat"] if len(sc["cat"]) <= 28 else sc["cat"][:25] + "..."
        text = sc["text"]

        hf_det, hf_top, hf_flagged = predict_hf(hf_model, hf_tokenizer, text, device)
        ft_det, ft_top = predict_ft(ft_model, ft_tokenizer, text, device, threshold=ft_thresholds)

        hf_str = "❌ " + ", ".join(hf_det) if hf_flagged else "✅ " + hf_top
        ft_str = "❌ " + ", ".join(ft_det) if ft_det else "✅ " + ft_top

        if len(hf_str) > 32: hf_str = hf_str[:29] + "..."
        if len(ft_str) > 32: ft_str = ft_str[:29] + "..."

        if hf_flagged: hf_toxic_count += 1
        if ft_det: ft_toxic_count += 1

        print(f"{sc_id:<4} | {cat:<28} | {hf_str:<32} | {ft_str:<32}")
        results.append({
            "id": sc_id, "category": sc["cat"], "text": text,
            "textdetox_prediction": hf_str, "hingbert_prediction": ft_str
        })

    print("-" * 105)
    print(f"Total Toxic Flags across {len(SCENARIOS)} Scenarios -> textdetox/bert-multilingual: {hf_toxic_count} | Our HingBERT: {ft_toxic_count}")
    print("="*105)

    # Save comparison report
    rep_path = os.path.join(os.path.dirname(__file__), "compare_textdetox_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump({"summary": {"total": len(SCENARIOS), "textdetox_toxic": hf_toxic_count, "hingbert_toxic": ft_toxic_count}, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nComparison report saved to: {rep_path}")

if __name__ == "__main__":
    main()
