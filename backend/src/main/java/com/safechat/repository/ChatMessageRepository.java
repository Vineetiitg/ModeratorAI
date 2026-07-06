package com.safechat.repository;

import com.safechat.model.ChatMessage;
import org.springframework.data.mongodb.repository.MongoRepository;

import java.util.List;
import java.util.stream.Stream;

public interface ChatMessageRepository extends MongoRepository<ChatMessage, String> {

    List<ChatMessage> findByChannelIdOrderByCreatedAtAsc(String channelId);

    /** For admin analytics */
    List<ChatMessage> findBySenderIdAndStatus(String senderId, String status);

    /** For the continuous-learning export: collect flagged/blocked messages */
    List<ChatMessage> findByStatusIn(List<String> statuses);

    /** Count toxic messages per channel for dashboard */
    long countByChannelIdAndModerationIsToxicTrue(String channelId);
}
