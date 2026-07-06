package com.safechat.controller;

import com.safechat.dto.AuthDtos;
import com.safechat.service.AuthService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;

    @PostMapping("/register")
    public ResponseEntity<AuthDtos.AuthResponse> register(@RequestBody AuthDtos.RegisterRequest req) {
        return ResponseEntity.ok(authService.register(req));
    }

    @PostMapping("/login")
    public ResponseEntity<AuthDtos.AuthResponse> login(@RequestBody AuthDtos.LoginRequest req) {
        return ResponseEntity.ok(authService.login(req));
    }

    /** Quick token validation for frontend */
    @GetMapping("/me")
    public ResponseEntity<?> me(@RequestHeader("Authorization") String authHeader) {
        String token = authHeader.replace("Bearer ", "").trim();
        return authService.getUserFromToken(token)
                .map(user -> ResponseEntity.ok(new AuthDtos.AuthResponse(
                        token, user.getId(), user.getDisplayName(), user.getEmail(), user.getRole())))
                .orElse(ResponseEntity.status(401).build());
    }
}
