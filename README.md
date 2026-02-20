# Fold Catastrophe Geometry in GPT-2's Residual Stream

When GPT-2 encounters an ambiguous token — a period that could be a decimal or a sentence boundary — it resolves the ambiguity by crossing a *fold*: a low-dimensional decision boundary with the geometric properties predicted by catastrophe theory.

**Paper:** paper.md
**Author:** Karli Joy (karlijoyj@gmail.com)

## Quick Start

```
pip install torch transformers numpy
python reproduce_all.py
```

36 automated checks. 29 seconds on CPU. No GPU needed.

## What's Here

| File | Description |
|------|-------------|
| `reproduce_all.py` | Full reproduction script — all 18 findings, 3 architectures, 36 checks |
| `verified_results.md` | Verified results document with every number from the paper |
| `dominance_matrix.py` | Standalone: 12×12 layer-pair authority contests |
| `confabulation_stats.py` | Standalone: split-brain confabulation across 20 prompts |
| `inception_dose_response.py` | Standalone: sigmoid dose-response + multi-layer fold erasure |
| `logit_linearity_check.py` | Standalone: verifies the sharp DM transition isn't a softmax artifact |

## Key Findings

- **Sharp fold transition** at layers 2–3 with a critical band showing 70× logit acceleration
- **Directional specificity**: 0/100 random directions trigger the flip
- **4.1:1 basin asymmetry** between structural and numerical interpretations
- **Cusp geometry**: context-dependent thresholds varying 2.3× across prompts
- **Cross-architecture replication** on Pythia-160M and GPT-2 Medium
- **Power-law exponent** δ≈0.45–0.57, bracketing the Landau mean-field prediction of 0.5

## License

MIT
