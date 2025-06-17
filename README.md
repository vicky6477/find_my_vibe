# 🧠 Find-My-Vibe

*A practical demo that fine-tunes and extends FashionCLIP — a CLIP-based model for the fashion domain — to enable attribute prediction and multi-style item retrieval.*

---

### Contents

- [🧠 Find-My-Vibe](#-find-my-vibe)
    - [Contents](#contents)
  - [Overview](#overview)
    - [🔍 About FashionCLIP](#-about-fashionclip)
    - [💡 What *Find-My-Vibe* Builds on Top](#-what-find-my-vibe-builds-on-top)
  - [Quick-start](#quick-start)
  - [Installation](#installation)
  - [Build the Faiss index](#build-the-faiss-index)
  - [CLI demo](#cli-demo)
  - [REST API + UI](#rest-api--ui)
  - [Using the embeddings directly](#using-the-embeddings-directly)
  - [What We Changed and Why](#what-we-changed-and-why)
  - [Project layout](#project-layout)
  - [🧠 Training \& Embedding Details](#-training--embedding-details)
    - [1. Fashion Attribute Classifier (5-way heads)](#1-fashion-attribute-classifier-5-way-heads)
    - [2. Triplet Projection Head](#2-triplet-projection-head)
    - [3. Embedding \& Indexing](#3-embedding--indexing)
  - [💼 Industrial Impact](#-industrial-impact)
  - [Considering to have:](#considering-to-have)
  - [Troubleshooting](#troubleshooting)
  - [Citation](#citation)

---

## Overview

### 🔍 About FashionCLIP

FashionCLIP is a fine-tuned version of OpenAI’s CLIP model, adapted for the fashion domain. Built using over 700K \<image, text> pairs from the Farfetch dataset, it enhances CLIP’s zero-shot capabilities for fashion-specific tasks like multi-modal retrieval and classification. The model architecture remains CLIP ViT-B/32, but its weights are refined to capture fine-grained fashion concepts.

For more, see [patrickjohncyh/fashion-clip on Hugging Face](https://huggingface.co/patrickjohncyh/fashion-clip) or [the original paper](https://www.nature.com/articles/s41598-022-23052-9).

### 💡 What *Find-My-Vibe* Builds on Top

This project starts with FashionCLIP and extends it by:

* Adding five classification heads for attribute prediction (item\_type, gender, color, season, style)
* Training a new triplet projection head using curated anchor-positive-negative samples
* Enabling dual retrieval modes:

  * **Strict mode** (based on attribute filtering)
  * **Combo mode** (hybrid logic using triplet-style similarity)
* Providing a real-time API and user interface with retrieval mode selection

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

The system supports **two types of input** and **two retrieval modes**:

* 📷 **Image input**: Upload a fashion item image

  * **Strict mode** → filters candidates by predicted attributes (via `recommend.py`)
  * **Style-combo mode** → hybrid scoring using style embedding + filtering (`recommend_combo.py`)

---

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
| 4. Strict Retrieval      | `recommend.py` filters candidates based on predicted attributes (hard rules)                | Provides fast and interpretable recommendations               | `recommend.py`              |
| 5. Hybrid Retrieval      | `recommend_combo.py` uses 5-way prediction for filtering, then ranks with triplet embedding | Combines structure-aware filtering and visual similarity      | `recommend_combo.py`        |
| 6. CLI Demo              | `demo_find.py` prints predictions and visual grid                                           | Easy local testing                                            | `demo_find.py`              |
| 7. FastAPI + UI          | REST API + HTML frontend with retrieval selector                                            | Makes system user-facing                                      | `api.py`, `index.html`      |
| 8. Cross-platform Fixes  | POSIX paths, numpy pin, libomp fix, env guards                                              | Ensures reproducibility                                       | `requirements.txt`, etc.    |

---

## Project layout

```
backend/
├─ train_fclip_multitask.py    # train 5-way attribute heads
├─ train_proj_head.py          # train custom projection head
├─ build_index_5way.py         # classic index build
├─ build_index_proj.py         # projection head index
├─ recommend.py                # 5-way filter-based strict matcher
├─ recommend_combo.py          # hybrid matcher with triplet logic
├─ api.py                      # FastAPI logic
frontend/index.html            # UI with retrieval selector
checkpoints/                   # saved models
our_dataset/                   # triplet samples + images
uploads/                       # test images
data/                          # generated FAISS indices and metadata
```

---

## 🧠 Training & Embedding Details

### 1. Fashion Attribute Classifier (5-way heads)

* Model: ViT-B/32 backbone + 5 linear heads (style, item\_type, gender, colour, season)
* Loss: CrossEntropyLoss for each head
* Training data: 31K labeled samples from `styles.csv`
* Script: [`train_fclip_multitask.py`](backend/train_fclip_multitask.py)

### 2. Triplet Projection Head

* Model: ViT-B/32 backbone + 256-D projection head
* Loss: TripletMarginLoss (anchor, positive, negative from `train_triplets.csv`)
* Dataset: `our_dataset/fashion_images/`
* Script: [`train_proj_head.py`](backend/train_proj_head.py)

### 3. Embedding & Indexing

* Embedding: image → 512-D (CLIP), or 256-D (projection head), normalized with L2 norm
* Index: FAISS `IndexFlatIP` used for nearest-neighbor search
* Build scripts:

  * [`build_index_5way.py`](backend/build_index_5way.py): uses 5-way head
  * [`build_index_proj.py`](backend/build_index_proj.py): uses projection head

---

## 💼 Industrial Impact

This system demonstrates a scalable **image-based** and **text-based** fashion recommendation pipeline, designed for practical use in fashion e-commerce and personalized retail experiences.

* 📷 **Image-to-Image Retrieval**:

  * Users upload an image of a fashion item they like.
  * The system predicts attributes and recommends visually or structurally similar items.
  * Two retrieval modes:

    * **Strict match**: Uses attribute classification and hard filtering (`recommend.py`)
    * **Style-combo**: Uses hybrid scoring with triplet-based embedding similarity (`recommend_combo.py`)
    * 
## Considering to have:
* 📝 **Text-to-Image Retrieval**:

  * Users enter a text description such as "white summer dress".
  * The system encodes the text using FashionCLIP’s text encoder and retrieves the most visually matching catalog images (`recommend_from_text`).

* ⚡ Built with FAISS for efficient large-scale retrieval.

* 🛍 Supports applications like visual product search, personal styling, and smart outfit suggestions.

This project extends the original FashionCLIP by adding multi-head classification, triplet-based training, and supporting both image and text input modes—providing a complete and extensible recommendation system.

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
