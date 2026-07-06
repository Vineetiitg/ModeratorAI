package com.safechat.service;

import com.safechat.dto.ChatMessageRequest;
import com.safechat.dto.ModerationResponse;
import com.safechat.model.ChatMessage;
import com.safechat.repository.ChatMessageRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class ChatService {

    private final ChatMessageRepository chatMessageRepository;
    private final MlClientService mlClientService;
    private final SimpMessagingTemplate messagingTemplate;

    /**
     * Main pipeline:
     * 1. Fetch the last 4 messages from this channel as context
     * 2. Save the new message initially as PENDING
     * 3. Call ML service with text + context
     * 4. Based on severity: DELIVER, FLAG, or BLOCK
     */
    public void processMessage(ChatMessageRequest request) {
        log.info("Processing message for channel: {} by user: {}", request.getChannelId(), request.getSenderId());

        // Step 1: Fetch last 4 messages from this channel for context
        List<ChatMessage> recentMessages = chatMessageRepository
                .findByChannelIdOrderByCreatedAtAsc(request.getChannelId());
        // Take last 4 delivered or flagged messages for context
        List<String> context = recentMessages.stream()
                .filter(m -> "DELIVERED".equals(m.getStatus()) || "FLAGGED".equals(m.getStatus()))
                .map(ChatMessage::getContent)
                .collect(Collectors.toList());
        int fromIndex = Math.max(0, context.size() - 4);
        context = context.subList(fromIndex, context.size());

        // Step 2: Save as PENDING for audit trail
        ChatMessage message = new ChatMessage();
        message.setChannelId(request.getChannelId());
        message.setSenderId(request.getSenderId());
        message.setSenderName(request.getSenderName());
        message.setContent(request.getContent());
        message.setStatus("PENDING");
        message.setContextSnapshot(context);
        message = chatMessageRepository.save(message);

        // Step 3: Call ML service asynchronously
        final String messageId = message.getId();
        final List<String> finalContext = context;

        mlClientService
                .moderateText(request.getContent(), request.getChannelId(), request.getSenderId(), finalContext)
                .subscribe(
                        resp -> handleModerationResult(messageId, request, resp),
                        err -> handleModerationError(messageId, request, err)
                );
    }

    private void handleModerationResult(String messageId, ChatMessageRequest request, ModerationResponse resp) {
        chatMessageRepository.findById(messageId).ifPresent(message -> {

            // Build moderation result
            ChatMessage.ModerationResult mod = new ChatMessage.ModerationResult();
            mod.setIsToxic(resp.getIsToxic());
            mod.setSeverity(resp.getSeverity());
            mod.setOverallScore(resp.getOverallScore());
            mod.setCategories(resp.getCategories());
            mod.setSuggestion(resp.getSuggestion());
            mod.setDetectedLanguage(resp.getDetectedLanguage());
            mod.setInferenceTimeMs(resp.getInferenceTimeMs());
            mod.setModelVersion(resp.getModelVersion());

            message.setModeration(mod);
            message.setModeratedAt(LocalDateTime.now());

            String severity = resp.getSeverity() != null ? resp.getSeverity() : "SAFE";

            if ("HIGH".equalsIgnoreCase(severity)) {
                // High toxicity: BLOCK — never broadcast to channel
                message.setStatus("BLOCKED");
                chatMessageRepository.save(message);
                // Notify only the sender
                messagingTemplate.convertAndSend(
                        "/topic/user/" + request.getSenderId() + "/alerts",
                        buildAlert("BLOCKED", message)
                );
                log.warn("Message BLOCKED [HIGH toxicity] from user: {}", request.getSenderId());

            } else if ("MEDIUM".equalsIgnoreCase(severity)) {
                // Medium toxicity: flag but still broadcast with warning
                message.setStatus("FLAGGED");
                chatMessageRepository.save(message);
                messagingTemplate.convertAndSend("/topic/channel/" + request.getChannelId(), message);
                // Also notify sender with suggestion
                messagingTemplate.convertAndSend(
                        "/topic/user/" + request.getSenderId() + "/alerts",
                        buildAlert("FLAGGED", message)
                );
                log.info("Message FLAGGED [MEDIUM toxicity] from user: {}", request.getSenderId());

            } else if ("LOW".equalsIgnoreCase(severity)) {
                // Low toxicity: deliver but annotate
                message.setStatus("DELIVERED");
                chatMessageRepository.save(message);
                messagingTemplate.convertAndSend("/topic/channel/" + request.getChannelId(), message);
                log.info("Message DELIVERED [LOW toxicity, annotated]");

            } else {
                // SAFE: deliver normally
                message.setStatus("DELIVERED");
                chatMessageRepository.save(message);
                messagingTemplate.convertAndSend("/topic/channel/" + request.getChannelId(), message);
            }
        });
    }

    private void handleModerationError(String messageId, ChatMessageRequest request, Throwable err) {
        log.error("ML service error for message {}: {}", messageId, err.getMessage());
        // Fallback: deliver the message and mark as DELIVERED with error note
        chatMessageRepository.findById(messageId).ifPresent(message -> {
            message.setStatus("DELIVERED"); // fail-open: don't block on ML outage
            chatMessageRepository.save(message);
            messagingTemplate.convertAndSend("/topic/channel/" + request.getChannelId(), message);
        });
    }

    private java.util.Map<String, Object> buildAlert(String type, ChatMessage message) {
        return java.util.Map.of(
                "type", type,
                "messageId", message.getId(),
                "severity", message.getModeration() != null ? message.getModeration().getSeverity() : "UNKNOWN",
                "suggestion", message.getModeration() != null && message.getModeration().getSuggestion() != null
                        ? message.getModeration().getSuggestion() : "",
                "overallScore", message.getModeration() != null ? message.getModeration().getOverallScore() : 0.0,
                "categories", message.getModeration() != null ? message.getModeration().getCategories() : java.util.Map.of()
        );
    }
}
