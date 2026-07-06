"""
SafeChat API Gateway — Real-Time WebSocket Router

WS /ws/chat — The real-time messaging and moderation pipeline:
  1. Rate limiting via Redis sliding window
  2. Context retrieval from MongoDB
  3. MuRIL toxicity classification via ML Service
  4. Token-by-token Gemini AI detox streaming
  5. Persistent document logging in MongoDB
  6. Channel broadcasting to connected clients
"""

import json
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from loguru import logger

from safety_platform.services.chat_service import ws_manager, ChatService
from safety_platform.services.ml_client import ml_client
from safety_platform.core.security import authenticate_websocket

router = APIRouter()


@router.websocket("/ws/chat")
async def websocket_chat_gateway(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="JWT Bearer Token"),
    channel: str = Query("general", description="Target discussion channel"),
    user_id: Optional[str] = Query(None, description="Fallback demo user ID"),
    user_name: Optional[str] = Query(None, description="Fallback demo display name"),
):
    """
    Real-Time WebSocket Gateway for Chat & Content Safety.
    Works with authenticated JWT users or demo session fallbacks from React UI.
    """
    # Attempt JWT authentication
    user = await authenticate_websocket(websocket, token)
    if user:
        client_id = user.id
        client_name = user.display_name
    else:
        # Fallback to demo identity from frontend parameters or random ID
        client_id = user_id or f"demo-{id(websocket)}"
        client_name = user_name or "Vineet (Demo)"

    channel_clean = channel.strip().lower()
    if channel_clean.startswith("#"):
        channel_clean = channel_clean[1:]

    await ws_manager.connect(websocket, channel_id=channel_clean, user_id=client_id, user_name=client_name)

    try:
        while True:
            raw_msg = await websocket.receive_text()

            try:
                payload = json.loads(raw_msg)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "data": {"detail": "Invalid JSON format."}
                })
                continue

            msg_type = payload.get("type", "message")
            text = payload.get("text", "").strip()
            skip_moderation = payload.get("skip_moderation", False)

            if msg_type not in ("message", "safe_message") or not text:
                continue

            # ── Step 1: Redis Sliding-Window Rate Limit Check ────────
            allowed = await ChatService.check_rate_limit(client_id)
            if not allowed:
                await websocket.send_json({
                    "type": "error",
                    "data": {"detail": "Rate limit exceeded (max 20 msgs/min). Please slow down."}
                })
                continue

            # Per user request: only classify current message without previous 4 context messages
            context_snapshot = []

            # ── Step 3: Call ML Service (MuRIL Toxicity Classifier) ──
            if msg_type == "safe_message" or skip_moderation:
                classification = {"is_toxic": False, "severity": "SAFE", "categories": {}}
            else:
                try:
                    classification = await ml_client.moderate_text(text=text, context=context_snapshot)
                except Exception as e:
                    logger.error(f"Classification failed: {e}")
                    classification = {"is_toxic": False, "severity": "SAFE", "categories": {}}

            # Immediately send classification results to client
            await websocket.send_json({
                "type": "classification",
                "data": classification,
            })

            # Determine moderation status
            is_toxic = classification.get("is_toxic", False)
            severity = classification.get("severity", "SAFE")
            suggestion = classification.get("suggestion")
            detected_lang = classification.get("detected_language", "en")

            msg_status = "DELIVERED"
            if severity == "HIGH":
                msg_status = "BLOCKED"
            elif severity == "MEDIUM":
                msg_status = "FLAGGED"

            # ── Step 4: Stream Gemini AI Detoxification if Toxic ─────
            if is_toxic and severity in ("MEDIUM", "HIGH"):
                await websocket.send_json({
                    "type": "detox_start",
                    "data": {"language": detected_lang},
                })

                if not suggestion:
                    detox_res = await ml_client.detoxify_text(
                        text=text,
                        target_language=detected_lang,
                        context=context_snapshot
                    )
                    suggestion = detox_res.get("detoxified_text") or detox_res.get("suggestion")

                suggestion_text = suggestion or text
                classification["suggestion"] = suggestion_text

                # Simulate token-by-token streaming over WebSocket for visual WOW factor
                words = suggestion_text.split(" ")
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    await websocket.send_json({
                        "type": "detox_chunk",
                        "data": {"chunk": chunk},
                    })

                await websocket.send_json({
                    "type": "detox_end",
                    "data": {"full_text": suggestion_text},
                })

            # ── Step 5: Save Full Record into MongoDB ────────────────
            saved_doc = await ChatService.save_message_to_mongo(
                channel_id=channel_clean,
                sender_id=client_id,
                sender_name=client_name,
                content=text,
                status=msg_status,
                moderation_data=classification,
                context_snapshot=context_snapshot,
            )

            # ── Step 6: Broadcast Delivered Message to Channel ───────
            if msg_status == "DELIVERED":
                await ws_manager.broadcast_to_channel(
                    channel_id=channel_clean,
                    message_data={
                        "type": "new_message",
                        "data": saved_doc.model_dump(),
                    }
                )

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error for {client_name}: {e}")
        ws_manager.disconnect(websocket)
