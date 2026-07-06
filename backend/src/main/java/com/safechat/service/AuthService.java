package com.safechat.service;

import com.safechat.dto.AuthDtos;
import com.safechat.model.User;
import com.safechat.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.Optional;

@Service
@RequiredArgsConstructor
@Slf4j
public class AuthService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;

    public AuthDtos.AuthResponse register(AuthDtos.RegisterRequest req) {
        if (userRepository.existsByEmail(req.getEmail())) {
            throw new RuntimeException("Email already in use: " + req.getEmail());
        }
        User user = new User();
        user.setEmail(req.getEmail());
        user.setDisplayName(req.getDisplayName());
        user.setPasswordHash(passwordEncoder.encode(req.getPassword()));
        // First registered user becomes ADMIN
        user.setRole(userRepository.count() == 0 ? "ADMIN" : "USER");
        user.setCreatedAt(LocalDateTime.now());
        user = userRepository.save(user);

        String token = jwtService.generateToken(user);
        log.info("Registered new user: {}", user.getEmail());
        return new AuthDtos.AuthResponse(token, user.getId(), user.getDisplayName(), user.getEmail(), user.getRole());
    }

    public AuthDtos.AuthResponse login(AuthDtos.LoginRequest req) {
        User user = userRepository.findByEmail(req.getEmail())
                .orElseThrow(() -> new RuntimeException("User not found"));

        if (!passwordEncoder.matches(req.getPassword(), user.getPasswordHash())) {
            throw new RuntimeException("Invalid credentials");
        }

        String token = jwtService.generateToken(user);
        log.info("User logged in: {}", user.getEmail());
        return new AuthDtos.AuthResponse(token, user.getId(), user.getDisplayName(), user.getEmail(), user.getRole());
    }

    public Optional<User> getUserFromToken(String token) {
        try {
            String userId = jwtService.extractUserId(token);
            return userRepository.findById(userId);
        } catch (Exception e) {
            return Optional.empty();
        }
    }
}
