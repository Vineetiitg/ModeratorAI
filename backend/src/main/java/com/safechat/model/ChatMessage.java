package com.safechat.model;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Document(collection = "chat_messages")
@Data
@NoArgsConstructor
public class ChatMessage {

    @Id
    private String id;

    @Indexed
    private String channelId;

    private String senderId;
    private String senderName;
    private String content;

    /**
     * PENDING | DELIVERED | FLAGGED | BLOCKED
     */
    private String status;

    private ModerationResult moderation;

    /**
     * Saved here for continuous learning pipeline —
     * these are the 4 preceding messages passed to the ML service.
     */
    private List<String> contextSnapshot = new ArrayList<>();

    /** Human feedback: true = correctly identified, false = wrong */
    private Boolean feedbackCorrect;
    private LocalDateTime feedbackAt;

    private LocalDateTime createdAt = LocalDateTime.now();
    private LocalDateTime moderatedAt;

    @Data
    @NoArgsConstructor
    public static class ModerationResult {
        private Boolean isToxic;
        private Double overallScore;
        /** SAFE | LOW | MEDIUM | HIGH */
        private String severity;
        private Map<String, Double> categories;
        private String suggestion;
        private String detectedLanguage;
        private Integer inferenceTimeMs;
        private String modelVersion;
    }
}
