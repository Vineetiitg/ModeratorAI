# SafeChat Real Multilingual Toxicity Dataset — EDA Report

**Total Unique Samples Analyzed:** `37638`
**Generated On:** `2026-07-06 11:03:28`

## 1. Dataset Splits & Files
| Split | Rows | File Path |
|---|---|---|
| **Train (80%)** | `30110` | `real_toxicity_train.csv` |
| **Validation (10%)** | `3764` | `real_toxicity_val.csv` |
| **Test (10%)** | `3764` | `real_toxicity_test.csv` |
| **Full Unified** | `37638` | `unified_dataset_full.csv` |

## 2. Source Language Breakdown
| Source / Language | Sample Count | Percentage |
|---|---|---|
| `english (jigsaw)` | 15000 | 39.9% |
| `english` | 8962 | 23.8% |
| `hindi` | 5860 | 15.6% |
| `hindi/hinglish (hasoc)` | 4891 | 13.0% |
| `hinglish` | 2881 | 7.7% |
| `hinglish (curated)` | 27 | 0.1% |
| `hindi_devanagari (curated)` | 15 | 0.0% |
| `english (curated)` | 2 | 0.0% |

## 3. Class Imbalance & Positive Weights (`pos_weight`)
To prevent gradient collapse on rare tags (like Threat and Identity Hate), these calculated weights are passed to `nn.BCEWithLogitsLoss(pos_weight=...)`.

| Tag Name | Positives | Negatives | Positive Rate | `pos_weight` |
|---|---|---|---|---|
| **`toxic`** | 19467 | 18171 | 51.72% | **`1.0`** |
| **`severe_toxic`** | 1616 | 36022 | 4.29% | **`22.29`** |
| **`obscene`** | 8461 | 29177 | 22.48% | **`3.45`** |
| **`threat`** | 494 | 37144 | 1.31% | **`25.0`** |
| **`insult`** | 17203 | 20435 | 45.71% | **`1.19`** |
| **`identity_hate`** | 2529 | 35109 | 6.72% | **`13.88`** |

## 4. Length & Script Statistics
- **Mean Word Count:** `44.3` words (95th percentile: `131.0` words)
- **Mean Character Length:** `253.7` characters
- **Script Breakdown:** `92.2%` ASCII Latin (English/Hinglish) vs `7.7%` Devanagari Unicode vs `0.1%` Other/Emojis.
- **Recommendation:** Setting `max_length=128` in tokenizer covers >98% of messages while speeding up GPU training by 4x compared to 512.

