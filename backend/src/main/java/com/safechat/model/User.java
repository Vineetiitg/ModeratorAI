package com.safechat.model;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;

@Document(collection = "users")
@Data
@NoArgsConstructor
public class User {

    @Id
    private String id;

    @Indexed(unique = true)
    private String email;

    private String passwordHash;

    private String displayName;

    /** USER | ADMIN */
    private String role = "USER";

    private Boolean active = true;
    private Boolean isMuted = false;
    private LocalDateTime mutedUntil;
    private Integer violationCount = 0;

    /** OTP fields for email-based auth */
    private String otpCode;
    private LocalDateTime otpExpiry;

    private LocalDateTime createdAt = LocalDateTime.now();
    private LocalDateTime updatedAt = LocalDateTime.now();
}
