"""
SafeChat — Toxicity Classifier (Fine-tuned HingBERT)

Architecture:
  ┌─────────┐
  │  Input  │ (Hindi, English, Hinglish, code-mixed)
  └────┬────┘
       │
       ▼
  ┌───────────────────────────┐
  │ Preprocessing             │
  │ • Script-aware lang detect│
  │ • Adversarial normalization│
  └───────┬───────────────────┘
          │
          ▼
  ┌───────────────────────────┐
  │ Fine-tuned HingBERT       │
  │ (SequenceClassification)  │
  │ num_labels=6, multi-label │
  └───────┬───────────────────┘
          │ Sigmoid
          ▼
  ┌───────────────────────────┐
  │ Labels & Severity         │
  └───────────────────────────┘

Context-Aware Mode:
  When conversation history is provided, it is passed as
  the `text` argument to the tokenizer and the current
  message as `text_pair`, producing correct BERT segment
  IDs (token_type_ids) for cross-attention between context
  and current message.
"""

import time
from typing import Dict, List, Optional

import anyio
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from loguru import logger

from app.config import settings
from app.utils.preprocessing import clean_text, detect_language, normalize_for_toxicity

# Labels as defined in our training data
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class ToxicityClassifier:
    """
    Toxicity classifier powered by fine-tuned HingBERT base.

    Features:
      - Multi-label classification (6 toxicity categories)
      - Context-aware via proper BERT segment tokenization
      - Real GPU-batched inference for the batch endpoint
      - Async-safe: heavy inference runs in a thread pool
    """

    def __init__(self, model_name: str = settings.CLASSIFIER_MODEL, device: str = settings.DEVICE):
        self.device = device
        self._loaded = False
        self._model_name = model_name

        self.tokenizer = None
        self.model = None
        self.model_version = settings.APP_VERSION

    def load(self) -> None:
        """Load fine-tuned HingBERT model into memory."""
        logger.info(f"Loading tokenizer and model ({self._model_name}) on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name,
                num_labels=len(LABELS),
                problem_type="multi_label_classification",
            )
            self.model.to(self.device)
            self.model.eval()

            # Load secondary multilingual gatekeeper architecture to eliminate Indic false positives
            try:
                gate_name = "textdetox/bert-multilingual-toxicity-classifier"
                self.gate_tokenizer = AutoTokenizer.from_pretrained(gate_name)
                self.gate_model = AutoModelForSequenceClassification.from_pretrained(gate_name).to(self.device)
                self.gate_model.eval()
                logger.info("Multilingual gatekeeper integrated into Hing-RoBERTa pipeline.")
            except Exception as ge:
                logger.warning(f"Could not load secondary gatekeeper: {ge}")
                self.gate_model = None

            self._loaded = True
            logger.success(f"HingBERT toxicity model loaded from {self._model_name}")
        except Exception as e:
            logger.error(f"Failed to load HingBERT model: {e}")
            raise e

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _predict_sync(self, text: str, context: Optional[List[str]] = None) -> Dict:
        """
        Synchronous single-sample inference (runs on calling thread).

        Context-aware tokenization:
          If context is provided, it is joined and passed as the first
          segment (`text` argument), while the current message becomes
          the second segment (`text_pair`). This produces correct
          token_type_ids so HingBERT can attend across context and message.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call classifier.load() first.")

        start_time = time.perf_counter()

        # Step 1: Preprocess
        normalized = normalize_for_toxicity(text)
        lang = detect_language(text)

        # Step 2: Tokenize with proper segment handling
        context_str = None
        if context and len(context) > 0:
            # Use last 4 turns as context segment
            context_str = " ".join(context[-4:])

        inputs = self.tokenizer(
            context_str if context_str else normalized,
            normalized if context_str else None,
            return_tensors="pt",
            max_length=settings.MAX_SEQ_LENGTH,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Step 3: Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)[0].cpu().numpy().tolist()

        # Step 3b: Gatekeeping check to prevent false positives on Devanagari/Indic compliments
        if hasattr(self, "gate_model") and self.gate_model is not None:
            try:
                gate_inputs = self.gate_tokenizer(
                    normalized,
                    return_tensors="pt",
                    max_length=settings.MAX_SEQ_LENGTH,
                    truncation=True,
                    padding=True,
                ).to(self.device)
                with torch.no_grad():
                    gate_logits = self.gate_model(**gate_inputs).logits
                    gate_probs = torch.softmax(gate_logits, dim=-1)[0].cpu().numpy()
                
                gate_toxic_prob = float(gate_probs[1]) if len(gate_probs) == 2 else 0.5
                if gate_toxic_prob < 0.40 and max(probs) >= settings.THRESHOLD_SAFE:
                    probs = [min(p, gate_toxic_prob * 0.8) for p in probs]
                elif gate_toxic_prob >= 0.60 and max(probs) < settings.THRESHOLD_SAFE:
                    probs[0] = max(probs[0], gate_toxic_prob)
            except Exception as ge:
                logger.debug(f"Gatekeeping skipped: {ge}")

        # Step 4: Map to categories
        categories = {LABELS[i]: round(probs[i], 4) for i in range(len(LABELS))}

        # Step 5: Overall score + severity
        overall_score = round(max(categories.values()), 4)
        severity = self._score_to_severity(overall_score)

        inference_time_ms = int((time.perf_counter() - start_time) * 1000)

        return {
            "is_toxic": overall_score >= settings.THRESHOLD_SAFE,
            "overall_score": overall_score,
            "severity": severity,
            "categories": categories,
            "detected_language": lang,
            "model_version": self.model_version,
            "inference_time_ms": inference_time_ms,
        }

    async def predict(self, text: str, context: Optional[List[str]] = None) -> Dict:
        """
        Async-safe single-sample prediction.

        Runs the synchronous PyTorch inference in a thread pool
        so it does not block the FastAPI asyncio event loop.
        """
        return await anyio.to_thread.run_sync(self._predict_sync, text, context)

    def _predict_batch_sync(self, texts: List[str]) -> List[Dict]:
        """
        Real GPU-batched inference for multiple texts.

        Tokenizes all texts together with padding, feeds a single
        batched tensor to the model, and splits results.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call classifier.load() first.")

        start_time = time.perf_counter()

        # Step 1: Preprocess all texts
        normalized_texts = [normalize_for_toxicity(t) for t in texts]
        languages = [detect_language(t) for t in texts]

        # Step 2: Batch tokenize
        inputs = self.tokenizer(
            normalized_texts,
            return_tensors="pt",
            max_length=settings.MAX_SEQ_LENGTH,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Step 3: Batched inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            all_probs = torch.sigmoid(logits).cpu().numpy().tolist()

        # Step 3b: Batch Gatekeeping check
        if hasattr(self, "gate_model") and self.gate_model is not None:
            try:
                gate_inputs = self.gate_tokenizer(
                    normalized_texts,
                    return_tensors="pt",
                    max_length=settings.MAX_SEQ_LENGTH,
                    truncation=True,
                    padding=True,
                ).to(self.device)
                with torch.no_grad():
                    gate_logits = self.gate_model(**gate_inputs).logits
                    gate_probs = torch.softmax(gate_logits, dim=-1).cpu().numpy()
                
                new_all_probs = []
                for i, probs_list in enumerate(all_probs):
                    probs_list = list(probs_list)
                    gate_toxic_prob = float(gate_probs[i][1]) if len(gate_probs[i]) == 2 else 0.5
                    if gate_toxic_prob < 0.40 and max(probs_list) >= settings.THRESHOLD_SAFE:
                        probs_list = [min(p, gate_toxic_prob * 0.8) for p in probs_list]
                    elif gate_toxic_prob >= 0.60 and max(probs_list) < settings.THRESHOLD_SAFE:
                        probs_list[0] = max(probs_list[0], gate_toxic_prob)
                    new_all_probs.append(probs_list)
                all_probs = new_all_probs
            except Exception as ge:
                logger.debug(f"Batch gatekeeping skipped: {ge}")

        # Step 4: Build results for each sample
        total_time_ms = int((time.perf_counter() - start_time) * 1000)
        per_sample_ms = total_time_ms // max(len(texts), 1)

        results = []
        for i, probs in enumerate(all_probs):
            categories = {LABELS[j]: round(float(probs[j]), 4) for j in range(len(LABELS))}
            overall_score = round(max(categories.values()), 4)

            results.append({
                "is_toxic": overall_score >= settings.THRESHOLD_SAFE,
                "overall_score": overall_score,
                "severity": self._score_to_severity(overall_score),
                "categories": categories,
                "detected_language": languages[i],
                "model_version": self.model_version,
                "inference_time_ms": per_sample_ms,
            })

        return results

    async def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Async-safe batched prediction."""
        return await anyio.to_thread.run_sync(self._predict_batch_sync, texts)

    @staticmethod
    def _score_to_severity(score: float) -> str:
        """Map a toxicity score to a severity level."""
        if score < settings.THRESHOLD_SAFE:
            return "SAFE"
        elif score < settings.THRESHOLD_LOW:
            return "LOW"
        elif score < settings.THRESHOLD_MEDIUM:
            return "MEDIUM"
        else:
            return "HIGH"

    def get_info(self) -> Dict:
        """Return model metadata for health checks."""
        return {
            "model": {
                "name": self._model_name,
                "loaded": self._loaded,
                "labels": LABELS,
            },
            "gatekeeper": {
                "model": "textdetox/bert-multilingual-toxicity-classifier",
                "loaded": hasattr(self, "gate_model") and self.gate_model is not None,
            },
            "device": self.device,
            "version": self.model_version,
        }

