package com.safechat.controller;

import com.safechat.service.AdminService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/admin")
@RequiredArgsConstructor
public class AdminController {

    private final AdminService adminService;

    /** GET /api/admin/stats — dashboard analytics */
    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> getStats() {
        return ResponseEntity.ok(adminService.getModerationStats());
    }

    /** GET /api/admin/export — export flagged/blocked messages for retraining */
    @GetMapping("/export")
    public ResponseEntity<List<Map<String, Object>>> exportForRetraining() {
        return ResponseEntity.ok(adminService.exportForRetraining());
    }
}
