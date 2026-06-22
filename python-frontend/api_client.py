"""
SafeChat API Client — Talks to the ML Service.
"""
import requests
import os

API_URL = os.getenv("SAFECHAT_API_URL", "http://localhost:8001")


def check_health() -> dict | None:
    """Check if ML Service is reachable. Returns health payload or None."""
    try:
        r = requests.get(f"{API_URL}/api/v1/health", timeout=3)
        if r.status_code == 200:
            return r.json()
        return None
    except requests.exceptions.RequestException:
        return None


def moderate(text: str, context: list[str] | None = None) -> dict | None:
    """POST /api/v1/moderate — returns full moderation result."""
    payload = {"text": text}
    if context:
        payload["context"] = context
    try:
        r = requests.post(f"{API_URL}/api/v1/moderate", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def detoxify(text: str, context: list[str] | None = None) -> dict | None:
    """POST /api/v1/detoxify — returns detoxified text."""
    payload = {"text": text}
    if context:
        payload["context"] = context
    try:
        r = requests.post(f"{API_URL}/api/v1/detoxify", json=payload, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def get_feedback_stats() -> dict | None:
    """GET /api/v1/feedback/stats — continuous learning metrics."""
    try:
        r = requests.get(f"{API_URL}/api/v1/feedback/stats", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def submit_feedback(
    message_id: str,
    moderator_id: str,
    was_correct: bool,
    correct_label: str | None = None,
    notes: str | None = None,
) -> dict | None:
    """POST /api/v1/feedback — submit moderator correction."""
    payload = {
        "message_id": message_id,
        "moderator_id": moderator_id,
        "model_prediction_was_correct": was_correct,
    }
    if correct_label:
        payload["correct_label"] = correct_label
    if notes:
        payload["notes"] = notes
    try:
        r = requests.post(f"{API_URL}/api/v1/feedback", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None
