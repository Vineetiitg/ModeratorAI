package com.safechat.controller;

import com.safechat.dto.ChatMessageRequest;
import com.safechat.model.ChatMessage;
import com.safechat.repository.ChatMessageRepository;
import com.safechat.service.ChatService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/chat")
public class ChatController {

    private final ChatService chatService;
    private final ChatMessageRepository chatMessageRepository;

    /** WebSocket entry point — /app/chat.send */
    @MessageMapping("/chat.send")
    public void sendMessage(@Payload ChatMessageRequest request) {
        chatService.processMessage(request);
    }

    /** REST: get message history for a channel */
    @GetMapping("/channel/{channelId}")
    public ResponseEntity<List<ChatMessage>> getChannelMessages(@PathVariable String channelId) {
        return ResponseEntity.ok(chatMessageRepository.findByChannelIdOrderByCreatedAtAsc(channelId));
    }

    /** REST: get a single message by ID */
    @GetMapping("/message/{id}")
    public ResponseEntity<ChatMessage> getMessage(@PathVariable String id) {
        return chatMessageRepository.findById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    /** REST: mark human feedback on a message (for continuous learning) */
    @PostMapping("/message/{id}/feedback")
    public ResponseEntity<ChatMessage> feedback(
            @PathVariable String id,
            @RequestParam boolean correct
    ) {
        return chatMessageRepository.findById(id).map(msg -> {
            msg.setFeedbackCorrect(correct);
            msg.setFeedbackAt(java.time.LocalDateTime.now());
            chatMessageRepository.save(msg);
            return ResponseEntity.ok(msg);
        }).orElse(ResponseEntity.notFound().build());
    }
}
