#!/usr/bin/env python3
"""
SafeChat — Comprehensive End-to-End Test Suite for Fine-Tuned HingBERT

Tests our fine-tuned model (`checkpoints/hingbert-toxicity-finetuned`) across 25 diverse
real-world communication scenarios:
  1. Formal & Safe (English, Hindi Devanagari, Hinglish)
  2. Compliments & Admiration (Testing words like 'hero', 'hoshiyar', 'mast')
  3. Mild Slang & Dismissive Comments
  4. Severe Profanity & Obscenity
  5. Violent Death Threats & Intimidation
  6. Hate Speech (Religious, Communal, Casteist)
  7. Sexism & Misogyny
  8. Sarcasm & Passive-Aggressive Trolling
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.models.toxicity_classifier import ToxicityClassifier

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

TAGS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "hingbert-toxicity-finetuned")

SCENARIOS = [
    # ── Category 1: Formal & Safe ──
    {"id": "S01", "cat": "Formal Safe (English)", "text": "Good morning team, let's sync up at 11 AM for the daily standup."},
    {"id": "S02", "cat": "Formal Safe (Hindi Devanagari)", "text": "नमस्ते सर, क्या आप मुझे आज की बैठक का एजेंडा भेज सकते हैं?"},
    {"id": "S03", "cat": "Casual Safe (Hinglish)", "text": "Bhai project ka status kya hai? Sab thik chal raha hai na?"},
    
    # ── Category 2: Compliments & Admiration (Testing false-positive triggers) ──
    {"id": "C01", "cat": "Compliment (English)", "text": "You did an incredible job on the presentation today, keep it up!"},
    {"id": "C02", "cat": "Compliment (Hinglish 'hoshiyar')", "text": "tum bahut hoshiyar aur samajhdar ho, tumne problem solve kar di."},
    {"id": "C03", "cat": "Compliment (Hinglish 'hero')", "text": "bhai tu mera hero hai, tera kaam sach mein lajawab hai!"},
    {"id": "C04", "cat": "Compliment (Hindi Devanagari)", "text": "आज का कार्यक्रम बहुत ही शानदार और प्रेरणादायक था।"},
    
    # ── Category 3: Mild Slang & Dismissive Comments ──
    {"id": "M01", "cat": "Mild Slang (Hinglish)", "text": "tu ek number ka gadha hai, kuch samajh nahi aata tujhe."},
    {"id": "M02", "cat": "Dismissive (English)", "text": "stop asking dumb questions and let me work in peace."},
    {"id": "M03", "cat": "Dismissive (Hindi Devanagari)", "text": "तुम बहुत बड़े बेवकूफ हो, तुमसे कुछ नहीं होने वाला चुप बैठो।"},
    
    # ── Category 4: Severe Profanity & Obscenity ──
    {"id": "P01", "cat": "Severe Obscenity (English)", "text": "fuck you you absolute motherfucking bastard, go to hell."},
    {"id": "P02", "cat": "Profanity + Threat (Hinglish)", "text": "bhenchod bakwas mat kar warna tera muh tod dunga harami saale."},
    {"id": "P03", "cat": "Severe Obscenity (Hindi Devanagari)", "text": "मादरचोद कमीने कुत्ते तेरी औकात क्या है मुझसे बात करने की भाग यहाँ से।"},
    {"id": "P04", "cat": "Cyberbullying (Hinglish)", "text": "saale chutiye tera dimaag kharab ho gaya hai kya nalayak insaan."},
    
    # ── Category 5: Violent Death Threats & Intimidation ──
    {"id": "T01", "cat": "Violent Threat (English)", "text": "I will hunt you down and kill you and your entire family."},
    {"id": "T02", "cat": "Violent Threat (Hinglish)", "text": "mai tera murder kar dunga agar tune ek aur baar call kiya toh."},
    {"id": "T03", "cat": "Death Threat (Hindi Devanagari)", "text": "तुझे घर से निकाल कर गोली मार दूंगा, आज रात तेरा आखिरी दिन है।"},
    
    # ── Category 6: Hate Speech (Religious, Communal, Casteist) ──
    {"id": "H01", "cat": "Religious Hate (English)", "text": "all people belonging to your religion are dirty terrorists and scum of the earth."},
    {"id": "H02", "cat": "Casteist Hate (Hinglish)", "text": "tumhari jaati aur community ke log hamesha se chor aur ghatiya rahe hain."},
    {"id": "H03", "cat": "Communal Hate (Hindi Devanagari)", "text": "इस धर्म के सारे लोग देश के गद्दार हैं, इन सबको देश से बाहर निकाल देना चाहिए।"},
    
    # ── Category 7: Sexism & Misogyny ──
    {"id": "X01", "cat": "Misogyny (English)", "text": "women belong only in the kitchen, stop trying to act smart and do your chores."},
    {"id": "X02", "cat": "Misogyny (Hinglish)", "text": "aurton ka kaam sirf ghar sambhalna hai, zyada dimaag mat chalao aur chup raho."},
    {"id": "X03", "cat": "Misogyny (Hindi Devanagari)", "text": "लड़कियों में दिमाग नहीं होता, उन्हें सिर्फ घर का काम करना चाहिए।"},
    
    # ── Category 8: Sarcasm & Passive-Aggressive Trolling ──
    {"id": "R01", "cat": "Sarcasm (Hinglish)", "text": "wah bhai kya dimaag paya hai, aisi bewakoofi bhari baatein sirf tum hi kar sakte ho."},
    {"id": "R02", "cat": "Passive-Aggressive (English)", "text": "oh brilliant idea, let's just ruin the entire project because you can't read instructions."}
]

def predict(model, tokenizer, text, device, threshold=0.50):
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
    
    return detected, f"{top_tag.upper()}: {top_prob:.1f}%", probs

def main():
    print("="*105)
    print("🛡️ SAFECHAT COMPREHENSIVE BENCHMARK: FINE-TUNED HINGBERT ACROSS 25 SCENARIOS")
    print("="*105)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Acceleration Device: {device}")

    print(f"\nLoading SafeChat Fine-Tuned Model (with Integrated Multilingual Gatekeeper)...")
    classifier = ToxicityClassifier()
    classifier.load()
    print("-> Model and Gatekeeper loaded successfully!")

    print("\n" + "="*105)
    print(f"{'ID':<4} | {'Scenario Category':<32} | {'Status & Detected Tags':<32} | {'Top Tag Score':<16} | {'Message Snippet'}")
    print("-" * 105)

    toxic_count = 0
    safe_count = 0

    results_for_report = []

    for sc in SCENARIOS:
        sc_id = sc["id"]
        cat = sc["cat"] if len(sc["cat"]) <= 32 else sc["cat"][:29] + "..."
        text = sc["text"]
        snippet = text if len(text) <= 30 else text[:27] + "..."

        res = classifier._predict_sync(text)
        is_flagged = res["is_toxic"]
        cats = res["categories"]
        top_tag = max(cats, key=cats.get)
        top_prob = cats[top_tag] * 100.0
        top_str = f"{top_tag.upper()}: {top_prob:.1f}%"

        det = [f"{k.upper()}({v*100:.0f}%)" for k, v in cats.items() if v >= 0.50]

        if is_flagged:
            status_str = "❌ TOXIC: " + ", ".join([d.split('(')[0] for d in det]) if det else f"❌ TOXIC({top_prob:.0f}%)"
            toxic_count += 1
        else:
            status_str = "✅ SAFE"
            safe_count += 1

        if len(status_str) > 32: status_str = status_str[:29] + "..."

        print(f"{sc_id:<4} | {cat:<32} | {status_str:<32} | {top_str:<16} | {snippet}")
        results_for_report.append({
            "id": sc_id, "category": sc["cat"], "text": text, "flagged": is_flagged, "detected_tags": det, "top_score": top_str
        })

    print("-" * 105)
    print(f"Total Scenarios Tested: {len(SCENARIOS)} | Flagged as Toxic: {toxic_count} | Passed as Safe: {safe_count}")
    print("="*105)

    # Save to JSON report
    report_path = os.path.join(os.path.dirname(__file__), "comprehensive_test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": {"total": len(SCENARIOS), "toxic": toxic_count, "safe": safe_count}, "results": results_for_report}, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    main()
