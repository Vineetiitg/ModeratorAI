package com.safechat.dto;

import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class ChatMessageRequest {
    private String channelId;
    private String content;
    private String senderId;
    private String senderName;
    /** Client-provided token to resolve user from JWT */
    private String token;
}
