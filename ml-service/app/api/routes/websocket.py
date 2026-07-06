"""
SafeChat — WebSocket Route for Real-Time Chat Moderation

WS /ws/chat — Real-time toxicity classification with streaming LLM detoxification

Protocol:
  Client sends JSON:
    {"type": "message", "text": "...", "user_id": "optional"}

  Server responds with JSON events:
    {"type": "classification", "data": {...}}          — Immediate toxicity result
    {"type": "detox_start", "data": {}}                — Detoxification started
    {"type": "detox_chunk", "data": {"chunk": "..."}}  — Streamed token
    {"type": "detox_end", "data": {"full_text": "..."}} — Detoxification complete
    {"type": "error", "data": {"detail": "..."}}       — Error
"""

import json
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.models.model_manager import model_manager

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections and per-connection conversation history."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.conversation_history: Dict[str, List[str]] = {}

    async def connect(self, websocket: WebSocket, connection_id: str) -> None:
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.conversation_history[connection_id] = []
        logger.info(f"WebSocket connected: {connection_id}")

    def disconnect(self, connection_id: str) -> None:
        self.active_connections.pop(connection_id, None)
        self.conversation_history.pop(connection_id, None)
        logger.info(f"WebSocket disconnected: {connection_id}")

    def add_to_history(self, connection_id: str, message: str) -> None:
        """Add a message to the conversation history (keep last 5)."""
        if connection_id in self.conversation_history:
            self.conversation_history[connection_id].append(message)
            # Keep only last 5 messages for context
            if len(self.conversation_history[connection_id]) > 5:
                self.conversation_history[connection_id] = self.conversation_history[connection_id][-5:]

    def get_history(self, connection_id: str) -> List[str]:
        return self.conversation_history.get(connection_id, [])


manager = ConnectionManager()


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    Real-time chat moderation over WebSocket.

    Flow per message:
      1. Client sends message → server immediately classifies with MuRIL
      2. Classification result sent back (~50-200ms)
      3. If toxic (MEDIUM/HIGH): LLM detoxification starts
      4. Detoxified text streams back token-by-token
      5. Message added to conversation history for context-aware next prediction
    """
    # Generate a unique connection ID
    connection_id = f"ws-{id(websocket)}"

    await manager.connect(websocket, connection_id)

    try:
        while True:
            # Receive message from client
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "data": {"detail": "Invalid JSON. Send: {\"type\": \"message\", \"text\": \"...\"}"}
                })
                continue

            msg_type = data.get("type", "message")
            text = data.get("text", "").strip()

            if msg_type != "message" or not text:
                await websocket.send_json({
                    "type": "error",
                    "data": {"detail": "Missing 'text' field or invalid type."}
                })
                continue

            # Check if models are ready
            if not model_manager.is_ready:
                await websocket.send_json({
                    "type": "error",
                    "data": {"detail": "Models are still loading. Please wait."}
                })
                continue

            # Get conversation context for this connection
            context = manager.get_history(connection_id)

            # ── Step 1: Classify toxicity (async, thread-safe) ─────
            try:
                classification = await model_manager.classifier.predict(text, context=context)
            except Exception as e:
                logger.error(f"Classification failed for WS {connection_id}: {e}")
                await websocket.send_json({
                    "type": "error",
                    "data": {"detail": f"Classification failed: {str(e)}"}
                })
                continue

            # Send classification result immediately
            await websocket.send_json({
                "type": "classification",
                "data": classification,
            })

            # Add message to conversation history
            manager.add_to_history(connection_id, text)

            # ── Step 2: Stream detoxification if needed ────────────
            if classification["is_toxic"] and classification["severity"] in ("MEDIUM", "HIGH"):
                detoxifier = model_manager.detoxifier

                if detoxifier:
                    # Signal that detoxification is starting
                    await websocket.send_json({
                        "type": "detox_start",
                        "data": {"language": classification["detected_language"]},
                    })

                    # Stream tokens
                    full_text = ""
                    try:
                        async for chunk in detoxifier.detoxify_stream(
                            text=text,
                            toxicity_categories=classification["categories"],
                            target_language=classification["detected_language"],
                            context=context,
                        ):
                            full_text += chunk
                            await websocket.send_json({
                                "type": "detox_chunk",
                                "data": {"chunk": chunk},
                            })
                    except Exception as e:
                        logger.error(f"Detox streaming failed: {e}")
                        full_text = "Could not generate suggestion."

                    # Signal completion
                    await websocket.send_json({
                        "type": "detox_end",
                        "data": {"full_text": full_text.strip()},
                    })

    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
        manager.disconnect(connection_id)
