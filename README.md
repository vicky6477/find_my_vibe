# 🧠 Find-My-Vibe

*A practical demo that fine-tunes and extends FashionCLIP — a CLIP-based model for the fashion domain — to enable attribute prediction and multi-style item retrieval.*

---

### Contents

1. [Quick-start](#quick-start)
2. [Overview](#overview)
3. [Installation](#installation)
4. [Build the Faiss index](#build-the-faiss-index)
5. [CLI demo](#cli-demo)
6. [REST API + UI](#rest-api--ui)
7. [Using the embeddings directly](#using-the-embeddings-directly)
8. [What We Changed and Why](#what-we-changed-and-why)
9. [Project layout](#project-layout)
10. [Training & Embedding Details](#training--embedding-details)
11. [Industrial Impact](#industrial-impact)
12. [Troubleshooting](#troubleshooting)
13. [Citation](#citation)

---

## Overview

### 🔍 About FashionCLIP

FashionCLIP is a fine-tuned version of OpenAI’s CLIP model, adapted for the fashion domain. Built using over 700K \<image, text> pairs from the Farfetch dataset, it enhances CLIP’s zero-shot capabilities for fashion-specific tasks like multi-modal retrieval and classification. The model architecture remains CLIP ViT-B/32, but its weights are refined to capture fine-grained fashion concepts.

For more, see [patrickjohncyh/fashion-clip on Hugging Face](https://huggingface.co/patrickjohncyh/fashion-clip) or [the original paper](https://www.nature.com/articles/s41598-022-23052-9).

### 💡 What *Find-My-Vibe* Builds on Top

This project starts with FashionCLIP and extends it by:

- Adding five classification heads for attribute prediction (item\_type, gender, color, season, style)
- Training a new triplet projection head using curated anchor-positive-negative samples
- Enabling hybrid retrieval: strict same-item match + mix-and-match by style
- Providing a real-time API and user interface with retrieval mode selection

---

## Quick-start

```bash
git clone https://github.com/<you>/find_my_vibe.git
cd find_my_vibe

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build indices (choose based on retrieval mode)
python -m backend.build_index_5way      # classification-based index
python -m backend.build_index_proj      # projection head index

python main.py     # API and UI at http://127.0.0.1:8000
```

## Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Build the Faiss index

```bash
python -m backend.build_index_5way     # creates data/fclip_cosine.index
python -m backend.build_index_proj     # creates data/fclip_triplet.index
```

## CLI demo

```bash
python demo_find.py uploads/example.jpg -k 3
```

## REST API + UI

```bash
python main.py
```

Then visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)

The frontend lets users choose:

- **Same-item (strict)**: filters by item\_type, color, etc.
- **Style-combo (mix & match)**: relaxed mode using hybrid embedding similarity

## Using the embeddings directly

```python
from fashion_clip.fashion_clip import FashionCLIP
fclip = FashionCLIP('fashion-clip')
image_embeddings = fclip.encode_images(images, batch_size=32)
```

---

## What We Changed and Why

| Stage                    | What we added / changed                                                                     | Why it matters                                                | Key file(s)                 |
| ------------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------- |
| 1. Fine-tune FashionCLIP | Added `train_fclip_multitask.py` with 5 heads (style, item\_type, gender, colour, season)   | Enables prediction of item\_type, color, etc.                 | `train_fclip_multitask.py`  |
| 2. Train Projection Head | New 256-D projection head trained with triplet loss on `our_dataset/`                       | Supports fine-grained visual similarity in custom style space | `train_proj_head.py`        |
| 3. Embedding + Indexing  | `build_index_5way.py` + `build_index_proj.py` embed and index both head outputs             | Makes both retrieval paths available for querying             | `build_index_*.py`, `data/` |
| 4. Hybrid Retrieval      | `recommend_combo.py` uses 5-way prediction for filtering, then ranks with triplet embedding | Combines structure-aware filtering and visual similarity      | `recommend_combo.py`        |
| 5. CLI Demo              | `demo_find.py` prints predictions and visual grid                                           | Easy local testing                                            | `demo_find.py`              |
| 6. FastAPI + UI          | REST API + HTML frontend with retrieval selector                                            | Makes system user-facing                                      | `api.py`, `index.html`      |
| 7. Cross-platform Fixes  | POSIX paths, numpy pin, libomp fix, env guards                                              | Ensures reproducibility                                       | `requirements.txt`, etc.    |

---

## Project layout

```
backend/
├─ train_fclip_multitask.py    # train 5-way attribute heads
├─ train_proj_head.py          # train custom projection head
├─ build_index_5way.py         # classic index build
├─ build_index_proj.py         # projection head index
├─ recommend.py                # 5-way filter matcher
├─ recommend_combo.py          # hybrid matcher
├─ api.py                      # FastAPI logic
frontend/index.html            # UI with retrieval selector
checkpoints/                   # saved models
our_dataset/                   # triplet samples + images
uploads/                       # test images
```

---

## 🧠 Training & Embedding Details

### 1. Fashion Attribute Classifier (5-way heads)

- Model: ViT-B/32 backbone + 5 linear heads (style, item\_type, gender, colour, season)
- Loss: CrossEntropyLoss for each head
- Training data: 31K labeled samples from `styles.csv`
- Script: [`train_fclip_multitask.py`](backend/train_fclip_multitask.py)

### 2. Triplet Projection Head

- Model: ViT-B/32 backbone + 256-D projection head
- Loss: TripletMarginLoss (anchor, positive, negative from `train_triplets.csv`)
- Dataset: `our_dataset/fashion_images/`
- Script: [`train_proj_head.py`](backend/train_proj_head.py)

### 3. Embedding & Indexing

- Embedding: image → 512-D (CLIP), or 256-D (projection head), normalized with L2 norm
- Index: FAISS `IndexFlatIP` used for nearest-neighbor search
- Build scripts:
  - [`build_index_5way.py`](backend/build_index_5way.py): uses 5-way head
  - [`build_index_proj.py`](backend/build_index_proj.py): uses projection head

---

## 💼 Industrial Impact

This system bridges the gap between foundational vision-language models (CLIP) and domain-specific recommendation tasks in fashion e-commerce. It:

- Supports attribute-aware recommendations for structured search (e.g. "white women's summer dress")
- Enables flexible outfit discovery through style-combo matching (e.g. "show me items that go well with this top")
- Offers fast, on-device inference and scalable API integration

This approach demonstrates how pretrained multi-modal models can be **adapted and extended for commercial use** by combining zero-shot learning, lightweight task-specific heads, and fast retrieval backends like Faiss.

---

## Troubleshooting

| Symptom                       | Fix                                                                |
| ----------------------------- | ------------------------------------------------------------------ |
| Seg-fault during build\_index | Use wheels in requirements.txt; delete old data/\*.index; rebuild. |
| Thumbnails 404                | Ensure `/fashion-dataset/{basename}` URLs are accessible.          |
| NumPy 2 ImportError           | `pip install "numpy<2"` – needed for faiss compatibility.          |

---

## Citation

```bibtex
@Article{Chia2022,
    title="Contrastive language and vision learning of general fashion concepts",
    author="Chia, Patrick John et al.",
    journal="Scientific Reports",
    year="2022",
    volume="12",
    number="1",
    pages="18958",
    doi="10.1038/s41598-022-23052-9"
}
```

