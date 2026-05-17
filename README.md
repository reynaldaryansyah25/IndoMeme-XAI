# IndoMeme-XAI

**Multimodal Hate Speech Detection in Indonesian Memes with Explainable AI Pipeline**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-INDOMEME-orange.svg)](https://huggingface.co/datasets/aiatums/INDOMEME)

## Overview

IndoMeme-XAI is a multimodal framework that integrates cross-modal attention fusion with an Explainable AI (XAI) pipeline for detecting hate speech in Indonesian memes. The project systematically compares four fusion architectures and provides comprehensive interpretability analysis through four attribution methods.

### Key Results

| Variant | Macro-F1 | AUC |
|---------|----------|-----|
| A — Late Fusion (baseline) | 0.7912 | 0.8827 |
| B — Single Cross-Attention | 0.7814 | 0.8768 |
| **C — ICAM Three-Stream** | **0.7946** | **0.8841** |
| D — No Topic Embedding | 0.7456 | 0.8330 |

Best ensemble (5 seeds): **Macro-F1 = 0.8076**, ROC-AUC = 0.8875

### Key Findings

1. **ICAM achieves best performance** — the three-stream fusion (bidirectional cross-attention + incongruity stream) marginally outperforms late fusion on INDOMEME (5,023 samples)
2. **Topic information is critical** — removing topic embedding drops Macro-F1 by 4.9 percentage points, confirming strong topic-label correlation
3. **Text dominance** — ROAR evaluation proves the model relies predominantly on textual features; Integrated Gradients is the only faithful attribution method, while all three visual methods operate at noise floor

## Architecture

```
Input Meme
    ├── Text:  [topic] caption [SEP] ocr  →  IndoBERTweet (124M params)
    └── Image: 224×224 resize              →  ViT-Base (86M params)
                                                    │
                              ┌─────────────────────┤
                              │   Fusion Module      │
                              │   (4 variants)       │
                              └─────────────────────┤
                                                    │
                              Topic Embedding (32-dim) ──┐
                                                         │
                              Classification Head ───────┘
                              (800→256→2, GELU, dropout)
                                        │
                                  hate / not-hate
```

## Dataset

This project uses [INDOMEME](https://huggingface.co/datasets/aiatums/INDOMEME) — the first benchmark dataset for hate speech detection in Indonesian memes.

| Split | Hate | Not Hate | Total |
|-------|------|----------|-------|
| Train | 1,178 | 2,237 | 3,415 |
| Validation | 208 | 395 | 603 |
| Test | 347 | 658 | 1,005 |

## Training Configuration

- **Encoders**: IndoBERTweet + ViT-Base (2 layers frozen each)
- **Loss**: Focal Loss (γ=2.0) + Supervised Contrastive Loss (λ=0.3) with curriculum
- **Optimizer**: AdamW with discriminative LR (encoder: 1e-5, fusion: 3e-5, head: 5e-5)
- **Scheduler**: Cosine annealing with 10% warmup
- **Batch size**: 64 effective (32 × 2 gradient accumulation)
- **Hardware**: Kaggle T4×2, mixed precision (fp16)

## XAI Pipeline

Four attribution methods evaluated with ROAR faithfulness:

| Method | Domain | Faithful? |
|--------|--------|-----------|
| Integrated Gradients | Text | Yes |
| Cross-Attention Map | Image | No |
| Attention Rollout | Image | No |
| Incongruity Heatmap (novel) | Image | No (but class-discriminative) |

## Reproducibility

All experiments use fixed seeds (42, 123, 2024) for the ablation study and (42, 123, 2024, 7, 999) for ensemble. Notebooks are designed to run on Kaggle with T4 GPU.

```bash
# Clone and install dependencies
git clone https://github.com/YOUR_USERNAME/IndoMeme-XAI.git
cd IndoMeme-XAI
pip install -r requirements.txt
```
## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [INDOMEME Dataset](https://huggingface.co/datasets/aiatums/INDOMEME) by Pamungkas et al. (2024)
- [IndoBERTweet](https://huggingface.co/indolem/indobertweet-base-uncased) by Koto et al. (2021)
- [Captum](https://captum.ai/) for Integrated Gradients implementation
- Kaggle for GPU compute resources
