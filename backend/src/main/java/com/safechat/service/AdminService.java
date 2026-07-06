package com.safechat.service;

import com.safechat.model.ChatMessage;
import com.safechat.repository.ChatMessageRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * AdminService — Powers the Admin Panel.
 *
 * Provides analytics, moderation stats, and initiates model retraining
 * by triggering the ML service via an HTTP call.
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class AdminService {

    private final ChatMessageRepository chatMessageRepository;
    private final MlClientService mlClientService;

    /**
     * Returns moderation analytics for the admin dashboard.
     */
    public Map<String, Object> getModerationStats() {
        long total = chatMessageRepository.count();
        List<ChatMessage> blocked = chatMessageRepository.findByStatusIn(List.of("BLOCKED"));
        List<ChatMessage> flagged = chatMessageRepository.findByStatusIn(List.of("FLAGGED"));
        List<ChatMessage> delivered = chatMessageRepository.findByStatusIn(List.of("DELIVERED"));

        // Severity breakdown
        Map<String, Long> bySeverity = new HashMap<>();
        bySeverity.put("HIGH", (long) blocked.size());
        bySeverity.put("MEDIUM", (long) flagged.size());
        bySeverity.put("LOW_OR_SAFE", (long) delivered.size());

        return Map.of(
                "totalMessages", total,
                "bySeverity", bySeverity,
                "toxicRate", total > 0 ? (double)(blocked.size() + flagged.size()) / total : 0.0
        );
    }

    /**
     * Export flagged and blocked messages as CSV-friendly DTO list
     * for continuous learning retraining pipeline.
     */
    public List<Map<String, Object>> exportForRetraining() {
        List<ChatMessage> toxicMessages = chatMessageRepository.findByStatusIn(List.of("BLOCKED", "FLAGGED"));
        return toxicMessages.stream().map(m -> {
            Map<String, Object> entry = new HashMap<>();
            entry.put("messageId", m.getId());
            entry.put("content", m.getContent());
            entry.put("severity", m.getModeration() != null ? m.getModeration().getSeverity() : "UNKNOWN");
            entry.put("detectedLanguage", m.getModeration() != null ? m.getModeration().getDetectedLanguage() : "");
            entry.put("suggestion", m.getModeration() != null ? m.getModeration().getSuggestion() : "");
            entry.put("context", m.getContextSnapshot());
            entry.put("timestamp", m.getCreatedAt());
            return entry;
        }).toList();
    }
}
