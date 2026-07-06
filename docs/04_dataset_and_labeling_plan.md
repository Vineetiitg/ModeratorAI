# Dataset and Labeling Plan

## Data strategy

Use public datasets for bootstrapping and your own curated annotation set for credibility. Public toxicity datasets are useful, but they are not enough for Indian multilingual chat moderation.

## Dataset buckets

### Bucket A: general toxicity baselines

- Jigsaw Toxic Comment Classification Challenge
- Jigsaw Unintended Bias in Toxicity Classification
- Jigsaw Multilingual Toxic Comment Classification
- OLID and OffensEval style datasets

Use these for:

- English baseline learning
- general toxicity transfer
- calibration experiments

## Bucket B: chat and conversational toxicity

- ToxicChat
- other conversation-oriented moderation datasets where license allows

Use these for:

- chat-specific evaluation
- contextual false-positive analysis
- rewrite trigger design

## Bucket C: Indian and code-mixed abuse data

- HASOC shared task datasets
- GLUECoS code-switched resources
- DravidianLangTech offensive-language datasets
- Hindi-English code-mixed abuse datasets from shared tasks or papers

Use these for:

- Hindi and Hinglish benchmarks
- code-mix robustness
- script-aware evaluation

## Bucket D: your own gold set

This is mandatory if you want the project to stand out.

Target:

- `1,500` to `3,000` manually reviewed chat examples
- balanced across English, Hindi, Hinglish, and selected additional Indian languages
- include both toxic and non-toxic difficult negatives

Hard negatives to include:

- banter between friends
- quoted abuse
- counterspeech
- profanity without harassment
- sarcasm
- veiled threats
- Romanized abuse with spelling variation

## License discipline

For every dataset, track:

- source
- license
- domain
- language coverage
- label schema
- intended use restrictions

Do not mix datasets blindly and then claim production readiness.

## Label schema

Each sample should include:

- `text`
- `context`
- `language_primary`
- `language_secondary`
- `script`
- `code_mixed`
- `labels`
- `severity`
- `target_type`
- `action_recommendation`
- `annotator_confidence`
- `notes`

## Severity scale

- `0`: safe
- `1`: mild disrespect or profanity
- `2`: clear harassment or abuse
- `3`: severe threat, hate, sexual exploitation, or self-harm encouragement

## Target type

- individual
- protected group
- self
- community
- unknown

## Annotation rules you should enforce

- label the intent, not just the presence of profanity
- do not mark quoted abusive text as direct abuse unless the speaker is endorsing it
- treat self-harm references separately from general toxicity
- if context changes meaning, mark `context_required=true`
- if language is mixed, label every relevant language present

## Split design

Keep at least three evaluation slices:

- in-domain validation
- out-of-domain test
- code-mixed hard test set

Also maintain challenge sets for:

- obfuscation
- transliteration variance
- sarcasm
- false-positive prone profanity

## Data balance guidance

Avoid this common mistake:

- training on 90 percent English and then announcing multilingual support

Aim for a deliberate training mix, for example:

- `35%` English
- `30%` Hindi
- `25%` Hinglish and Roman Hindi
- `10%` other Indian languages for robustness experiments

These numbers can shift based on available data, but the mix should be intentional.

## Annotation tooling

Recommended:

- Label Studio for human review
- CSV or JSONL export with stable sample IDs
- reviewer disagreement reports

## Data versioning

Every training run should reference:

- data snapshot ID
- label guideline version
- preprocessing version
- model config

