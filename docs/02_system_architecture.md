# System Architecture

## Architecture principle

Start with a modular monolith for speed and clarity, then split services only where there is a real operational reason.

## Recommended service boundaries

### `moderation-api`

- handles synchronous moderation requests
- performs normalization, language detection, rule checks, model inference, and policy decision
- returns low-latency action and explanation metadata

### `review-worker`

- stores events
- prepares reviewer queues
- triggers rewrite generation if it is not in the synchronous path
- exports hard cases for labeling

### `trainer`

- runs offline training jobs
- evaluates candidate models
- registers artifacts in MLflow
- supports champion versus challenger comparisons

## Request flow

```text
client message
  -> normalization
  -> script and language analysis
  -> rules and lexical checks
  -> classifier inference
  -> policy engine
  -> allow / warn / block / review
  -> event logging
```

## Context strategy

The model input should eventually include:

- current message
- previous 3 to 5 turns
- speaker role markers
- community metadata
- optional conversation-level features such as repeated abuse patterns

Example input packing:

```text
[COMMUNITY=gaming]
[USER=user_17]
[TURN-3] assistant: Please calm down.
[TURN-2] user_22: Why are you spamming?
[TURN-1] user_17: Stop talking.
[CURRENT] user_17: ...
```

## Data plane versus control plane

### Data plane

- real-time moderation decisions
- low latency
- deterministic policy handling

### Control plane

- threshold updates
- policy versioning
- model approvals
- rollback
- reviewer overrides

## Storage design

### PostgreSQL

- moderation events
- reviewer feedback
- tenant policy configurations
- model and policy version references

### Object storage

- training snapshots
- exported evaluation reports
- annotation batches
- model artifacts

### Redis

- low-latency caching
- rate limiting
- asynchronous task queue or stream support

## Observability

Capture at minimum:

- request count
- p50 and p95 latency
- per-action counts
- per-language counts
- model version usage
- reviewer override rate
- class-specific false positive analysis

## Security and governance

- hash or tokenize user identifiers in logs
- avoid storing raw production text longer than necessary
- keep data lineage for every model
- never auto-promote retrained models without evaluation gates

## Deployment shape

### Local development

- FastAPI app
- PostgreSQL
- Redis
- MLflow
- Label Studio

### Resume-grade production demo

- API on Cloud Run or ECS Fargate
- Postgres managed database
- Redis managed cache
- model artifact store
- Prometheus and Grafana for metrics

## Growth path to multimodal

The moderation contract should remain modality-agnostic even if `v1` is text-first.

Future extensions:

- image upload plus OCR
- screenshot or meme text moderation
- voice chat moderation via ASR
- shared policy engine across modalities

