"""
SafeChat — Moderation Service

Orchestrates the full moderation pipeline:
  1. Classify text for toxicity (fine-tuned MuRIL)
  2. Generate polite alternative via LLM if toxic
  3. Return combined result

This is the main entry point called by the API routes.
"""

import time
from typing import Dict, List, Optional

from loguru import logger

from app.models.model_manager import model_manager
from app.schemas.moderation import ModerationResponse


class ModerationService:
    """
    Orchestrates toxicity classification + LLM detoxification.

    Stateless service — all state lives in ModelManager.
    """

    @staticmethod
    async def moderate(
        text: str,
        context: Optional[List[str]] = None,
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ModerationResponse:
        """
        Full moderation pipeline for a single message.

        Args:
            text: Raw message text
            context: Previous conversation messages for context-aware classification
            channel_id: Channel for policy lookup (future use)
            user_id: Sender ID for tracking (future use)

        Returns:
            ModerationResponse with toxicity scores and suggestion
        """
        start_time = time.perf_counter()

        # Step 1: Classify toxicity (async — runs in thread pool)
        classifier = model_manager.classifier
        if not classifier or not classifier.is_loaded:
            raise RuntimeError("Toxicity classifier not available")

        classification = await classifier.predict(text, context=context)

        # Step 2: Generate suggestion if toxic (MEDIUM or HIGH severity)
        suggestion = None
        if classification["is_toxic"] and classification["severity"] in ("MEDIUM", "HIGH"):
            detoxifier = model_manager.detoxifier
            if detoxifier:
                detox_result = await detoxifier.detoxify(
                    text=text,
                    toxicity_categories=classification["categories"],
                    target_language=classification["detected_language"],
                    context=context,
                )
                suggestion = detox_result["detoxified"]

        total_time_ms = int((time.perf_counter() - start_time) * 1000)

        return ModerationResponse(
            is_toxic=classification["is_toxic"],
            overall_score=classification["overall_score"],
            severity=classification["severity"],
            categories=classification["categories"],
            detected_language=classification["detected_language"],
            suggestion=suggestion,
            model_version=classification["model_version"],
            inference_time_ms=total_time_ms,
        )

    @staticmethod
    async def moderate_batch(
        texts: List[str],
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[ModerationResponse]:
        """Moderate multiple messages using real batched inference."""
        start_time = time.perf_counter()

        # Step 1: Batch classify (real GPU batching)
        classifier = model_manager.classifier
        if not classifier or not classifier.is_loaded:
            raise RuntimeError("Toxicity classifier not available")

        classifications = await classifier.predict_batch(texts)

        # Step 2: Generate suggestions for toxic messages
        results = []
        detoxifier = model_manager.detoxifier

        for i, classification in enumerate(classifications):
            suggestion = None
            if classification["is_toxic"] and classification["severity"] in ("MEDIUM", "HIGH"):
                if detoxifier:
                    detox_result = await detoxifier.detoxify(
                        text=texts[i],
                        toxicity_categories=classification["categories"],
                        target_language=classification["detected_language"],
                    )
                    suggestion = detox_result["detoxified"]

            total_time_ms = int((time.perf_counter() - start_time) * 1000)

            results.append(ModerationResponse(
                is_toxic=classification["is_toxic"],
                overall_score=classification["overall_score"],
                severity=classification["severity"],
                categories=classification["categories"],
                detected_language=classification["detected_language"],
                suggestion=suggestion,
                model_version=classification["model_version"],
                inference_time_ms=total_time_ms,
            ))

        return results


# Singleton service instance
moderation_service = ModerationService()

