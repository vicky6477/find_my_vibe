# 🧠 Find-My-Vibe

*A practical demo that fine-tunes FashionCLIP, predicts five key fashion
attributes, and returns the three catalogue items that best match a query
image.*

## Dataset

This project uses the [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset from Kaggle.
---

### Contents

- [🧠 Find-My-Vibe](#-find-my-vibe)
  - [Dataset](#dataset)
  - [This project uses the Fashion Product Images (Small) dataset from Kaggle.](#this-project-uses-the-fashion-product-images-small-dataset-from-kaggle)
    - [Contents](#contents)
  - [Quick-start](#quick-start)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Build the Faiss index](#build-the-faiss-index)
  - [CLI demo](#cli-demo)
    - [Predicted attributes](#predicted-attributes)
  - [REST API + UI](#rest-api--ui)
    - [Endpoint](#endpoint)
      - [Example Response `200 OK`](#example-response-200-ok)
  - [Using the embeddings directly](#using-the-embeddings-directly)
  - [What We Changed and Why](#what-we-changed-and-why)
    - [✨ End-to-End Flow](#-end-to-end-flow)
  - [Project layout](#project-layout)
  - [Troubleshooting](#troubleshooting)
  - [Citation](#citation)

---

## Quick-start

```bash
git clone https://github.com/<you>/find_my_vibe.git
cd find_my_vibe

# 1  create env + install wheels
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy<2 • torch 2.2.2 • faiss-cpu 1.7.4 …

# 2  one-time: build Faiss index (~8 min CPU, <3 min Apple-Silicon MPS)
python -m backend.build_index

# 3  run CLI grid
python backend/demo_find.py uploads/example.jpg -k 3

# 4  run API + front-end
python main.py           # http://127.0.0.1:8000  /docs for Swagger
```

## Overview

| Block        | Details                                                                                  |
| ------------ | ---------------------------------------------------------------------------------------- |
| Backbone     | patrickjohncyh/fashion-clip (ViT-B/32)                                                   |
| Heads        | 5 linear layers trained on 31k catalogue imgs → item-type, gender, colour, season, style |
| Hybrid score | 0.7 · CLIP cosine + 0.3 · style-propensity dot                                           |
| Retrieval    | Faiss IndexFlatIP on L2-normed 512-D vectors                                             |
| UI           | FastAPI backend + plain-HTML front-end                                                   |

## Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> requirements.txt pins: numpy < 2 · torch 2.2.2 (cpu) · faiss-cpu 1.7.4 · transformers ≥ 4.52 + FastAPI runtime deps.

## Build the Faiss index

```bash
python -m backend.build_index     # creates data/fclip_cosine.index (~80 MB)
```

Re-run only when you upgrade the Faiss wheel or add catalogue images.

## CLI demo

```bash
python backend/demo_find.py uploads/dress.jpg -k 3
```

### Predicted attributes

```
 item_type : dress
 gender    : women
 colour    : white
 season    : summer
 style top-3: ethnic (14.3 %), casual (13.2 %), formal (11.0 %)
```

\[matplotlib grid with query + 3 matches]

## REST API + UI

Start server:

```bash
python main.py
```

Then visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)

### Endpoint

`POST /predict/`

* Content-Type: `multipart/form-data (file=jpg/png)`

#### Example Response `200 OK`

```json
{
  "prediction": {
     "item_type": "tshirts",
     "gender": "men",
     "colour": "navy blue",
     "season": "fall",
     "style_top3": { "casual":0.88, "streetwear":0.08, "sports":0.03 }
  },
  "recommendations": [
     "/fashion-dataset/50556.jpg",
     "/fashion-dataset/50193.jpg",
     "/fashion-dataset/17875.jpg"
  ]
}
```

The front-end demo (`frontend/index.html`) uploads an image and renders the
attributes plus the three matches returned.

## Using the embeddings directly

```python
import numpy as np
from fashion_clip.fashion_clip import FashionCLIP

fclip = FashionCLIP('fashion-clip')

image_embeddings = fclip.encode_images(images, batch_size=32)
text_embeddings  = fclip.encode_text(texts,  batch_size=32)

# L2-norm so dot == cosine
image_embeddings /= np.linalg.norm(image_embeddings, axis=-1, keepdims=True)
text_embeddings  /= np.linalg.norm(text_embeddings,  axis=-1, keepdims=True)

similarity = image_embeddings @ text_embeddings.T   # (n_img × n_txt)
```

This is exactly how `backend/build_index.py` prepares vectors for Faiss.

## What We Changed and Why

| Stage                    | What we added / changed                                                                                                                                    | Why it matters                                                   | Key file(s)                                       |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| 1. Fine-tune FashionCLIP | Added `train_fclip_multitask.py` to load HuggingFace ViT-B/32, add 5 heads (style, item\_type, gender, colour, season), train only heads (freeze backbone) | Allows single forward pass to predict 5 fashion attributes       | backend/train\_fclip\_multitask.py                |
| 2. Embedding + indexing  | `build_index.py` embeds \~40k images, predicts soft style, stores 512-D vector and meta in Faiss + pickle                                                  | Efficient searchable index with full attribute metadata          | backend/build\_index.py, data/*.pkl, data/*.index |
| 3. Hybrid retrieval      | `recommend.py` combines CLIP vector similarity + soft style score, filters mismatched types                                                             | Improves accuracy by ensuring style and semantics both match     | backend/recommend\.py                          |
| 4. CLI visualisation     | `demo_find.py` shows predicted attributes + matplotlib grid or fallback to terminal paths                                                                  | Quick way to sanity-check output visually                        | backend/demo\_find.py                             |
| 5. FastAPI service       | `api.py`, `main.py`, and `index.html` power RESTful interface and upload UI                                                                                | Microservice + interactive preview without external dependencies | backend/api.py, frontend/index.html               |
| 6. Cross-platform fixes  | POSIX paths, pinned versions, libomp segfault fix, KMP warning fix                                                                                         | Ensures reproducibility across macOS/Linux/WSL                   | requirements.txt, recommend\.py                |

### ✨ End-to-End Flow

```
upload JPG  ──▶ FastAPI ──▶ recommend.py
                                │
             (backbone+heads)   │ (Faiss IP search)
             CLIP 512-D vector  │
             5-head probs       ▼
               + labels     top-3 catalogue images
                 ▲    ▲            ▲
                 └─────┘────────┘
           JSON  {attributes + /fashion-dataset/img.jpg}
```

## Project layout

```
backend/
├─ train_fclip_multitask.py    # fine-tune 5 heads
├─ build_index.py              # embed + index catalogue
├─ recommend.py             # hybrid matcher
├─ api.py                      # FastAPI app
demo_find.py                # CLI visualizer
main.py                        # server entry
frontend/index.html            # upload UI
data/                          # index & metadata live here
uploads/                       # user uploads land here
```

## Troubleshooting

| Symptom                       | Fix                                                                           |
| ----------------------------- | ----------------------------------------------------------------------------- |
| Seg-fault during build\_index | Use wheels in requirements.txt; delete old data/fclip\_cosine.index; rebuild. |
| Thumbnails 404                | Make sure api.py returns /fashion-dataset/{basename} URLs.                    |
| NumPy 2 ImportError           | `pip install "numpy<2"` – Faiss wheels target NumPy-1 ABI.                    |

## Citation

FashionCLIP: https://github.com/patrickjohncyh/fashion-clip

```bibtex
@Article{Chia2022,
  title   = "Contrastive language and vision learning of general fashion concepts",
  journal = "Scientific Reports",
  year    = "2022",
  author  = "Patrick John Chia et al.",
  doi     = "10.1038/s41598-022-23052-9"
}
```

MIT License © Huijing Yi · Jingyi Chen · Chenxu Lan · Wenyue Zhu
