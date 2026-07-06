# Execution Roadmap

## Week 1: Freeze the spec

Deliverables:

- problem statement rewritten as a product specification
- supported language list
- moderation taxonomy
- latency target
- first API contract

Exit criteria:

- no more ambiguity about what `v1` includes and excludes

## Week 2: Annotation design

Deliverables:

- labeling schema
- annotation guidelines
- examples of edge cases
- reviewer instructions for Hindi, Hinglish, and contextual cases

Exit criteria:

- two annotators can label the same sample with acceptable consistency

## Week 3: Data ingestion

Deliverables:

- raw dataset catalog
- license checklist
- ingestion scripts
- cleaned and deduplicated splits
- source metadata tracked per sample

Exit criteria:

- reproducible train, validation, and test partitions

## Week 4: Multilingual normalization

Deliverables:

- Unicode normalization
- repeated-character normalization
- URL, mention, and number masking
- script detection
- code-mix heuristics
- Roman Hindi normalization design

Exit criteria:

- preprocessing handles noisy chat inputs without breaking Indic text

## Week 5: Baseline moderation system

Deliverables:

- rules and lexical baseline
- baseline English-Hindi-Hinglish classifier
- evaluation script
- first metrics report

Exit criteria:

- a baseline exists for comparison and is served behind the API

## Week 6: Context-aware model

Deliverables:

- packed context input format
- multi-turn training data builder
- context-aware benchmark
- ablation against single-message input

Exit criteria:

- measurable improvement on contextual abuse cases

## Week 7: Policy engine and tenant configs

Deliverables:

- threshold maps
- severity-aware actions
- tenant or community policy config schema
- moderation decision audit trail

Exit criteria:

- same model output can drive different community policies

## Week 8: Rewrite suggestions

Deliverables:

- rewrite trigger rules
- rewrite evaluation rubric
- fallback template-based rewrites
- optional LLM-based rewrite path

Exit criteria:

- flagged messages get safer alternatives without blocking moderation availability

## Week 9: Reviewer workflow

Deliverables:

- feedback API
- review queue design
- moderator override flow
- disagreement tracking

Exit criteria:

- reviewer signals can be stored and exported back to training

## Week 10: MLOps layer

Deliverables:

- experiment tracking
- model registry
- champion versus challenger evaluation
- approval checklist for promotion

Exit criteria:

- every promoted model is reproducible and versioned

## Week 11: Inference optimization

Deliverables:

- ONNX export
- quantization
- latency benchmark
- throughput benchmark

Exit criteria:

- chosen model meets real-time constraints on affordable hardware

## Week 12: Final demo and polish

Deliverables:

- clean README
- architecture diagram
- benchmark table
- failure analysis report
- deployment instructions
- recorded demo or live deployment

Exit criteria:

- you can send the repo to a recruiter and it reads like a serious engineering project

## Resume checkpoints

### Strong halfway point after Week 6

- multilingual baseline done
- context-aware model done
- working API demo

### Strong final point after Week 12

- deployable service
- metrics and monitoring
- reviewer feedback loop
- documented benchmark results

## Common failure modes

- trying to add image or voice before text is strong
- training on whatever data is easy to download without license checks
- reporting only overall accuracy instead of per-class and per-language metrics
- claiming continuous learning without evaluation gates
- spending too much time on infrastructure before the model and data pipeline are real

