package com.safechat.service;

import com.safechat.dto.ModerationRequest;
import com.safechat.dto.ModerationResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.List;

@Service
@Slf4j
public class MlClientService {

    private final WebClient webClient;

    public MlClientService(
            WebClient.Builder webClientBuilder,
            @Value("${safechat.ml-service.url}") String mlServiceUrl
    ) {
        this.webClient = webClientBuilder
                .baseUrl(mlServiceUrl)
                .build();
    }

    /**
     * Send text + last-4-messages context to the FastAPI ML service.
     * Returns a reactive Mono<ModerationResponse>.
     */
    public Mono<ModerationResponse> moderateText(
            String text,
            String channelId,
            String userId,
            List<String> context
    ) {
        ModerationRequest request = new ModerationRequest(text, channelId, userId, context);

        log.debug("Calling ML service with context length: {}", context != null ? context.size() : 0);

        return this.webClient.post()
                .uri("/api/v1/moderate")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(ModerationResponse.class)
                .timeout(Duration.ofSeconds(30))
                .onErrorResume(e -> {
                    log.error("ML service unreachable: {}", e.getMessage());
                    // Fail-safe: return SAFE so users are not blocked when ML is down
                    ModerationResponse fallback = new ModerationResponse();
                    fallback.setIsToxic(false);
                    fallback.setSeverity("SAFE");
                    fallback.setOverallScore(0.0);
                    fallback.setModelVersion("fallback");
                    return Mono.just(fallback);
                });
    }
}
