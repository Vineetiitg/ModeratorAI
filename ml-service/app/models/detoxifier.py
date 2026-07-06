"""
SafeChat — Text Detoxifier (IndicBART Generation)

Converts toxic Hindi/Hinglish/English sentences to polite versions.
Uses ai4bharat/IndicBART architecture.
"""

from typing import Dict, Optional
from loguru import logger
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import settings
from app.utils.preprocessing import detect_language


class Detoxifier:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        logger.info("Detoxifier initialized (IndicBART mode).")

    def load_model(self) -> None:
        if not settings.USE_MODEL_DETOX:
            return

        try:
            logger.info(f"Loading IndicBART detox model: {settings.DETOX_MODEL}...")
            # For IndicBART, we usually use its AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(settings.DETOX_MODEL)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(settings.DETOX_MODEL)
            self._model.to(settings.DEVICE)
            self._model.eval()
            self._model_loaded = True
            logger.success(f"IndicBART Detox model loaded successfully on {settings.DEVICE}.")
        except Exception as e:
            logger.warning(f"Failed to load IndicBART: {e}")
            self._model_loaded = False

    def detoxify(
        self,
        text: str,
        toxicity_categories: Optional[Dict[str, float]] = None,
        target_language: Optional[str] = None,
    ) -> Dict:
        lang = target_language or detect_language(text)

        # Try IndicBART Generation
        if self._model_loaded and settings.USE_MODEL_DETOX:
            result = self._model_detoxify(text, lang)
            if result:
                return {
                    "original": text,
                    "detoxified": result,
                    "method": "indic_bart",
                    "language": lang,
                    "confidence": 0.85,
                }

        # Absolute Fallback if Model goes OOM or crashes
        return {
            "original": text,
            "detoxified": "Let's keep the conversation respectful and polite.",
            "method": "fallback",
            "language": lang,
            "confidence": 0.50,
        }

    def _model_detoxify(self, text: str, lang: str) -> Optional[str]:
        if not self._model or not self._tokenizer:
            return None

        try:
            # Prepare prompt
            prompt = f"Make this sentence polite: {text}"

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                max_length=settings.MAX_SEQ_LENGTH,
                truncation=True,
            )
            inputs = {k: v.to(settings.DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=settings.DETOX_MAX_LENGTH,
                    num_beams=settings.DETOX_NUM_BEAMS,
                    early_stopping=True,
                )

            result = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.strip() if result.strip() else None

        except Exception as e:
            logger.error(f"IndicBART detoxification failed: {e}")
            return None

    def get_info(self) -> Dict:
        return {
            "mode": "indic_bart" if self._model_loaded else "fallback",
            "model_name": settings.DETOX_MODEL if self._model_loaded else None,
        }
