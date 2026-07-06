# Model Benchmarking Plan

## Goal

Find a production-friendly moderation stack for English, Hindi, and Hinglish that balances:

- classification quality
- latency
- memory footprint
- robustness on code-mixed and transliterated inputs

## Candidate model families

### Baseline encoders

- `XLM-R-base`
- `MuRIL`
- `IndicBERT-v3-270M`

These are the right first candidates because they are realistic to fine-tune and serve. Do not begin with a large generative model.

## Benchmark tracks

### Track 1: single-message classification

Purpose:

- establish the non-contextual baseline

### Track 2: context-aware classification

Purpose:

- measure uplift from previous turns and speaker markers

### Track 3: hybrid cascade

Purpose:

- use rules or a lightweight model first
- route uncertain cases to a stronger model

This is often the best production design.

## Evaluation metrics

For each label:

- precision
- recall
- F1
- PR-AUC

Operational metrics:

- false positive rate on safe profanity and banter
- severe-class recall at fixed precision
- p50 and p95 latency
- throughput under concurrent load

## Required evaluation slices

- English only
- Hindi only
- Hinglish or Roman Hindi mixed with English
- Devanagari plus Latin mixed script
- low-context easy cases
- context-required hard cases
- obfuscated abusive text

## Benchmark table template

Track these columns:

- model name
- context mode
- macro F1
- severe-class recall
- Hinglish F1
- p95 latency
- memory footprint
- ONNX export success

## Promotion rules

A candidate should not become the default model unless it:

- beats the current model on Hinglish and Hindi slices
- preserves or improves severe-class recall
- meets latency target after export or quantization
- has reviewed failure analysis

## Error analysis checklist

Inspect failures across:

- profanity but not harassment
- friendly banter
- transliterated abuse
- sarcastic threats
- quoted hate content
- self-harm support versus self-harm encouragement

## Recommended experimentation order

1. rules-only baseline
2. single-message encoder baseline
3. context-packed encoder
4. class-weighting or focal-loss variants
5. threshold calibration
6. ONNX export and latency optimization
7. hybrid cascade

## Model serving plan

### Development

- direct PyTorch inference is acceptable

### Production demo

- export champion encoder to ONNX
- benchmark with ONNX Runtime
- consider Triton only if GPU batching becomes necessary

## Rewrite system benchmark

Keep rewrite evaluation separate from moderation classification.

Measure:

- toxicity reduction
- meaning preservation
- politeness
- rewrite acceptance rate in demo usage

