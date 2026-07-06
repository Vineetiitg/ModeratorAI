package com.safechat.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ModerationRequest {
    private String text;
    private String channelId;
    private String userId;
    /** Up to the 4 most recent messages for context-aware moderation */
    private List<String> context;
}
