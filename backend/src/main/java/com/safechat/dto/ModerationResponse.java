package com.safechat.dto;

import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.Map;

@Data
@NoArgsConstructor
public class ModerationResponse {
    private Boolean isToxic;
    private Double overallScore;
    /** SAFE | LOW | MEDIUM | HIGH */
    private String severity;
    private Map<String, Double> categories;
    private String detectedLanguage;
    private String suggestion;
    private String modelVersion;
    private Integer inferenceTimeMs;
}
