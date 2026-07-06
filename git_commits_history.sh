#!/usr/bin/env bash
# SafeChat Platform — Realistic Git Commit History Builder (Bash)
# This script organizes all files into 10 logical, professional engineering commits.
# It creates a clean commit history telling the complete story of building the system from scratch.

set -e

echo "=========================================================="
echo "  Building Realistic SafeChat Git Commit History..."
echo "=========================================================="

# Ensure we are on branch 'main'
git checkout -B main

# 1. Repository configuration and project scope documentation
echo -e "\n[1/10] Committing repository structure and architecture documentation..."
git add .gitignore .dockerignore README.md pyproject.toml docs/ docker-compose.yml .env.example
git commit -m "chore: initialize enterprise content safety platform repository and architecture docs"

# 2. Core security, configuration, and database layer
echo -e "\n[2/10] Committing API gateway core security and JWT auth..."
git add src/safety_platform/__init__.py src/safety_platform/core/ src/safety_platform/schemas/auth.py src/safety_platform/api/__init__.py src/safety_platform/api/routes/__init__.py src/safety_platform/api/routes/health.py src/safety_platform/api/routes/auth.py src/safety_platform/services/__init__.py src/safety_platform/services/auth_service.py
git commit -m "feat(api): scaffold core FastAPI gateway with JWT authentication and security middleware"

# 3. Moderation routing, channel management, and ML client service
echo -e "\n[3/10] Committing moderation routing and channel services..."
git add src/safety_platform/main.py src/safety_platform/api/router.py src/safety_platform/api/routes/moderation.py src/safety_platform/api/routes/chat.py src/safety_platform/api/routes/channels.py src/safety_platform/schemas/channel.py src/safety_platform/schemas/chat.py src/safety_platform/schemas/__init__.py src/safety_platform/services/chat_service.py src/safety_platform/services/channel_service.py src/safety_platform/services/ml_client.py
git commit -m "feat(gateway): implement moderation routing, channel management, and ML client integration"

# 4. Real-time WebSocket chat communication layer
echo -e "\n[4/10] Committing WebSocket broadcasting layer..."
git add src/safety_platform/api/routes/websocket.py
git commit -m "feat(realtime): add WebSocket broadcasting endpoint for real-time chat moderation"

# 5. ML microservice foundation and moderation schemas
echo -e "\n[5/10] Committing ML microservice foundation and schemas..."
git add ml-service/requirements.txt ml-service/Dockerfile ml-service/.env.example ml-service/setup_and_train.ps1 ml-service/app/__init__.py ml-service/app/config.py ml-service/app/main.py ml-service/app/schemas/ ml-service/app/api/ ml-service/app/utils/
git commit -m "feat(ml-service): create ML microservice foundation and moderation schemas"

# 6. Multi-label training pipelines and dataset curation for HingBERT
echo -e "\n[6/10] Committing HingBERT multi-label training pipelines..."
git add ml-service/train_hingbert_toxicity.py ml-service/train_hinggpt_detox.py ml-service/curate_optimal_detox_dataset.py ml-service/download_and_eda_real_data.py ml-service/download_hinggpt.py ml-service/optimize_thresholds.py ml-service/training/
git commit -m "feat(ml-training): implement HingBERT multi-label training pipeline with focal loss and threshold calibration"

# 7. Comprehensive evaluation benchmarks and V1 model card
echo -e "\n[7/10] Committing benchmark scripts and V1 model card..."
git add ml-service/MODEL_CARD.md ml-service/evaluate_10_scenarios.py ml-service/evaluate_hindi_scenarios.py ml-service/evaluate_metrics_comparison.py ml-service/benchmark_detox_before.py ml-service/compare_base_vs_finetuned.py ml-service/compare_detox_methods.py ml-service/compare_textdetox_vs_hingbert.py ml-service/test_comprehensive_scenarios.py ml-service/*.json
git commit -m "test(benchmarks): add comprehensive evaluation scripts and publish SafeChat V1 model card"

# 8. Hybrid gatekeeper classification engine and moderation service
echo -e "\n[8/10] Committing hybrid gatekeeper classification engine..."
git add ml-service/app/models/__init__.py ml-service/app/models/model_manager.py ml-service/app/models/toxicity_classifier.py ml-service/app/models/detoxifier.py ml-service/app/services/
git commit -m "feat(ml-service): integrate hybrid HingBERT + MuRIL gatekeeper classification engine and moderation endpoints"

# 9. Gemini 2.0 Flash API for intent-preserving detoxification
echo -e "\n[9/10] Committing Gemini 2.0 Flash detoxification streaming layer..."
git add ml-service/app/models/llm_detoxifier.py ml-service/interactive_demo.py ml-service/hybrid_detox_pipeline.py
git commit -m "feat(detox): implement real-time token streaming and intent-preserving style transfer via Gemini 2.0 Flash API"

# 10. Modern React + Vite UI dashboard and Spring Boot backend option
echo -e "\n[10/10] Committing Vite React dashboard and Spring Boot alternative..."
git add frontend/ Dockerfile backend/
git commit -m "feat(ui): build responsive dark-mode glassmorphism React dashboard with WebSocket chat and Spring Boot backend option"

# Final sweep for any newly created configuration or documentation files
echo -e "\n[Final Check] Checking for any remaining untracked files..."
git add .
if ! git diff-index --quiet HEAD --; then
    git commit -m "chore: final polish and deployment configuration for GitHub and Hugging Face"
else
    echo "No remaining files to commit."
fi

echo "=========================================================="
echo "  SUCCESS! All 10 commits created locally."
echo "  To push to GitHub:"
echo "    git remote add origin https://github.com/yourusername/repo.git"
echo "    git push -u origin main"
echo "=========================================================="
