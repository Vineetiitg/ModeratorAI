# Product Scope

## Project statement

Build a real-time moderation platform that ingests chat messages, detects unsafe or toxic content, applies configurable policy decisions, and suggests safer rewrites when needed.

## First principles

- The project should look like a deployable product, not just a model demo.
- The core challenge is not only toxicity detection. It is policy, context, multilingual variability, and operational reliability.
- English-only moderation is not enough for Indian usage patterns.
- Hindi and Hinglish are mandatory first-class citizens, not afterthoughts.

## Supported languages by phase

### Phase 1

- English
- Hindi in Devanagari
- Hinglish and Roman Hindi mixed with English

### Phase 2

- Marathi
- Bengali
- Tamil
- Telugu
- Kannada
- Malayalam

### Phase 3

- wider Indic coverage through benchmarked expansion and fallback review flows

## Product goals

- detect insult, threat, hate, self-harm, profanity, sexual content, and spam
- process messages in real time with low latency
- support configurable policy actions per community or tenant
- preserve auditability for every moderation decision
- create a defensible retraining workflow from moderator feedback

## Non-goals for v1

- full multimodal moderation
- online self-updating training in production
- equal accuracy claims across all Indian languages
- perfect rewrite generation

## Realistic service-level targets

- `p50` moderation latency under `80 ms`
- `p95` moderation latency under `150 ms`
- synchronous path only depends on local inference and policy evaluation
- rewrite suggestions can be slower and may run out-of-band if needed

## Core user stories

- As a chat platform, I want to block severe threats before they appear.
- As a community admin, I want tenant-specific thresholds and policies.
- As a moderator, I want to inspect why a message was flagged.
- As an ML engineer, I want uncertain or disagreed examples routed to annotation.

## Moderation taxonomy

Use multi-label classification rather than a single toxicity score.

- `insult`
- `threat`
- `hate`
- `self_harm`
- `sexual_explicit`
- `profanity`
- `spam`

## Policy actions

- `allow`
- `warn`
- `block`
- `review`

`rewrite` is not a policy action by itself. It is a companion capability triggered when the final action is `warn`, `block`, or `review`.

## Why multilingual moderation is harder than it looks

- Indian chat is often code-mixed within the same sentence.
- Toxicity appears in native script and Roman transliteration.
- Simple translation-first pipelines lose nuance and often miss slurs, sarcasm, and intent.
- Public toxicity datasets over-index on English and social media, not chat.

## What success looks like

The final project should be able to demonstrate:

- a live moderation API
- context-aware classification
- language and script-aware preprocessing
- benchmark tables across English, Hindi, and Hinglish
- reviewer feedback collection
- a controlled retraining pipeline design

