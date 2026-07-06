package com.safechat.dto;

import lombok.Data;
import lombok.NoArgsConstructor;

public class AuthDtos {

    @Data
    @NoArgsConstructor
    public static class RegisterRequest {
        private String email;
        private String password;
        private String displayName;
    }

    @Data
    @NoArgsConstructor
    public static class LoginRequest {
        private String email;
        private String password;
    }

    @Data
    @NoArgsConstructor
    public static class AuthResponse {
        private String token;
        private String userId;
        private String displayName;
        private String email;
        private String role;

        public AuthResponse(String token, String userId, String displayName, String email, String role) {
            this.token = token;
            this.userId = userId;
            this.displayName = displayName;
            this.email = email;
            this.role = role;
        }
    }
}
