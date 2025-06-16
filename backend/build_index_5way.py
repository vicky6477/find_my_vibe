#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Embed all images (recursively) with FashionCLIP + 5-way heads,
   then build cosine FAISS index and metadata pickle."""

import pathlib, pickle, time, numpy as np, faiss, torch
from tqdm import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from backend.train_fclip_multitask import FiveHead

# ▸ ▸ ▸ EDIT THESE PATHS IF YOUR TREE DIFFERS ◂ ◂ ◂
CKPT_PATH = pathlib.Path("checkpoints/fclip_heads_5way_sub.pth")
IMG_ROOT  = pathlib.Path("fashion_dataset/images")
INDEX_OUT = pathlib.Path("data/fclip_cosine.index")
META_OUT  = pathlib.Path("data/fclip_meta.pkl")

# ---------- 1.  Load checkpoint, backbone, processor, heads ----------
print("• Loading checkpoint …")
ckpt      = torch.load(CKPT_PATH, map_location="cpu")
backbone  = CLIPModel.from_pretrained(ckpt["fclip_id"])
processor = CLIPProcessor.from_pretrained(ckpt["fclip_id"])
heads = FiveHead(backbone.config.projection_dim,
                 len(ckpt["styles"]), len(ckpt["items"]),
                 len(ckpt["genders"]), len(ckpt["colours"]),
                 len(ckpt["seasons"]))
heads.load_state_dict(ckpt["heads"])
backbone.eval(); heads.eval()

DEVICE = ("mps"  if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available()        else
          "cpu")
backbone.to(DEVICE); heads.to(DEVICE)
print(f"• Device: {DEVICE}, proj_dim={backbone.config.projection_dim}")

# ----------- 2.  Collect all image paths -----------------------------
paths = sorted(IMG_ROOT.rglob("*.jpg")) + sorted(IMG_ROOT.rglob("*.png"))
assert paths, f"No images found under {IMG_ROOT}"
print(f"• Images found: {len(paths):,}")

vecs = np.empty((len(paths), backbone.config.projection_dim), dtype="float32")
meta = []

# ----------- 3.  Embed images in batches -----------------------------
BATCH = 64 if DEVICE != "cpu" else 8                
t0 = time.time()

for s in tqdm(range(0, len(paths), BATCH), ncols=80, desc="Embedding"):
    batch_paths = paths[s:s+BATCH]
    imgs = [Image.open(p).convert("RGB") for p in batch_paths]
    pix  = processor(images=imgs, return_tensors="pt", padding=True
                     )["pixel_values"].to(DEVICE)

    with torch.inference_mode():
        feat = backbone.get_image_features(pix)               # (B,D)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        vecs[s:s+len(imgs)] = feat.cpu().numpy()

        style_probs = torch.softmax(heads.h_style (feat), dim=-1).cpu().numpy()
        item_idx    = torch.softmax(heads.h_item  (feat), dim=-1).argmax(-1).cpu()
        gender_idx  = torch.softmax(heads.h_gender(feat), dim=-1).argmax(-1).cpu()
        colour_idx  = torch.softmax(heads.h_colour(feat), dim=-1).argmax(-1).cpu()
        season_idx  = torch.softmax(heads.h_season(feat), dim=-1).argmax(-1).cpu()

    for j, p in enumerate(batch_paths):
        meta.append((
            str(p),
            ckpt["genders"][gender_idx[j]],
            ckpt["items"  ][item_idx[j]],
            ckpt["colours"][colour_idx[j]],
            ckpt["seasons"][season_idx[j]],
            style_probs[j],
        ))

print(f"• Embedding finished in {time.time()-t0:.1f}s")


# ---------- 4.  Build FAISS index (inner-product = cosine) -----------
print("• Building FAISS index …")
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)

INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(INDEX_OUT))

META_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(META_OUT, "wb") as f:
    pickle.dump(meta, f)

print(f" Saved {index.ntotal:,} vectors to {INDEX_OUT}")
print(f" Metadata → {META_OUT}")
