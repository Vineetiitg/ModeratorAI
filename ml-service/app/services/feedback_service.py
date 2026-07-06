"""
SafeChat — Feedback Service

Handles moderator feedback storage and continuous learning triggers.
Feedback is stored in MongoDB (via async motor driver) and used to
retrain models when the threshold is reached.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timezone

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings


class FeedbackService:
    """
    Manages the moderator feedback loop with MongoDB persistence.

    Flow:
      1. Moderator reviews a flagged message
      2. Submits correction via POST /api/v1/feedback
      3. Feedback stored in MongoDB (persistent)
      4. When feedback count reaches threshold → trigger retraining
    """

    def __init__(self):
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
        self._last_retrain_at: Optional[datetime] = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to MongoDB. Called during FastAPI startup."""
        try:
            self._client = AsyncIOMotorClient(settings.MONGODB_URL)
            self._db = self._client[settings.MONGODB_DB]

            # Verify connection
            await self._client.admin.command("ping")
            self._connected = True
            logger.success(f"Connected to MongoDB: {settings.MONGODB_URL}/{settings.MONGODB_DB}")
        except Exception as e:
            logger.warning(
                f"MongoDB connection failed: {e}. "
                f"Feedback will use in-memory fallback (data lost on restart)."
            )
            self._connected = False

    async def disconnect(self) -> None:
        """Disconnect from MongoDB. Called during FastAPI shutdown."""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed.")

    @property
    def collection(self):
        """Get the feedback collection."""
        if self._db is not None:
            return self._db[settings.FEEDBACK_COLLECTION]
        return None

    async def submit_feedback(self, feedback: Dict) -> Dict:
        """
        Store moderator feedback and check if retraining should be triggered.
        """
        feedback_entry = {
            **feedback,
            "submitted_at": datetime.now(timezone.utc),
        }

        feedback_id = None

        if self.collection is not None:
            # MongoDB mode — persistent storage
            result = await self.collection.insert_one(feedback_entry)
            feedback_id = str(result.inserted_id)
        else:
            # Fallback: just log it (no persistence)
            feedback_id = f"fb-fallback-{datetime.now(timezone.utc).timestamp()}"
            logger.warning(f"Feedback {feedback_id} stored in memory only (no MongoDB)")

        # Get current counts
        stats = await self.get_stats()

        # Check if retraining should be triggered
        retrain_triggered = False
        feedback_since = stats["feedback_since_last_retrain"]
        if feedback_since >= settings.FEEDBACK_THRESHOLD_FOR_RETRAIN:
            retrain_triggered = await self._trigger_retraining()

        logger.info(
            f"Feedback {feedback_id} received. "
            f"Total: {stats['total_feedback']}, "
            f"Accuracy: {stats['accuracy']:.2%}. "
            f"Retrain triggered: {retrain_triggered}"
        )

        return {
            "feedback_id": feedback_id,
            "message": "Feedback recorded successfully",
            "total_feedback_count": stats["total_feedback"],
            "retrain_threshold": settings.FEEDBACK_THRESHOLD_FOR_RETRAIN,
            "retrain_triggered": retrain_triggered,
        }

    async def get_stats(self) -> Dict:
        """Return feedback statistics from MongoDB."""
        total = 0
        correct = 0
        incorrect = 0

        if self.collection is not None:
            total = await self.collection.count_documents({})
            correct = await self.collection.count_documents({"model_prediction_was_correct": True})
            incorrect = await self.collection.count_documents({"model_prediction_was_correct": False})

        accuracy = correct / total if total > 0 else 0.0

        # Count feedback since last retrain
        feedback_since = total  # Default: all feedback counts
        if self._last_retrain_at and self.collection is not None:
            feedback_since = await self.collection.count_documents(
                {"submitted_at": {"$gt": self._last_retrain_at}}
            )

        return {
            "total_feedback": total,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "accuracy": round(accuracy, 4),
            "feedback_since_last_retrain": feedback_since,
            "retrain_threshold": settings.FEEDBACK_THRESHOLD_FOR_RETRAIN,
            "next_retrain_at": max(0, settings.FEEDBACK_THRESHOLD_FOR_RETRAIN - feedback_since),
            "last_retrain_at": self._last_retrain_at,
        }

    async def _trigger_retraining(self) -> bool:
        """
        Trigger model retraining.

        Exports incorrect feedback data to JSONL for the training pipeline,
        then resets the retrain counter.
        """
        logger.warning(
            f"Retraining threshold reached. "
            f"Exporting feedback data for retraining pipeline..."
        )

        try:
            # Export incorrect predictions as training data
            training_data = await self._export_training_data()

            if training_data:
                # Save to JSONL file for the training pipeline
                export_path = Path(settings.MODEL_CHECKPOINT_DIR) / "feedback_training_data.jsonl"
                export_path.parent.mkdir(parents=True, exist_ok=True)

                with open(export_path, "a", encoding="utf-8") as f:
                    for entry in training_data:
                        f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")

                logger.info(f"Exported {len(training_data)} feedback samples to {export_path}")

            self._last_retrain_at = datetime.now(timezone.utc)
            logger.info("Retraining counter reset.")
            return True

        except Exception as e:
            logger.error(f"Retraining trigger failed: {e}")
            return False

    async def _export_training_data(self) -> list:
        """Export incorrect feedback data for retraining."""
        if self.collection is None:
            return []

        cursor = self.collection.find(
            {"model_prediction_was_correct": False},
            {"_id": 0},  # Exclude MongoDB _id
        )

        return await cursor.to_list(length=None)


# Singleton instance
feedback_service = FeedbackService()

