"""
recommend.py – Retrieve visually similar outfits with attribute filtering
=========================================================================

Public API
----------
    paths, attrs = recommend_combo("path/to/query.jpg", k=3, n=1000)
"""

import numpy as np
import pickle, faiss, torch
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from backend.train_proj_head       import ProjHead
from backend.train_fclip_multitask import FiveHead

# ------------------------------------------------------------------ #
# 0.  One-time loads
# ------------------------------------------------------------------ #

CLIP_ID = "patrickjohncyh/fashion-clip"
DEVICE  = "cpu"                                 # set to "cuda"/"mps" if preferred

clip = CLIPModel.from_pretrained(CLIP_ID).eval().to(DEVICE)
clip.requires_grad_(False)
proc = CLIPProcessor.from_pretrained(CLIP_ID)

proj_head = ProjHead().to(DEVICE).eval()
proj_head.load_state_dict(
    torch.load("checkpoints/proj_head_256.pth", map_location=DEVICE))

ckpt = torch.load("checkpoints/fclip_heads_5way.pth", map_location=DEVICE)
five_head = FiveHead(
    clip.config.projection_dim,
    len(ckpt["styles"]), len(ckpt["items"]),
    len(ckpt["genders"]), len(ckpt["colours"]),
    len(ckpt["seasons"])
).to(DEVICE).eval()
five_head.load_state_dict(ckpt["heads"])

idx_trip  = faiss.read_index("data/fclip_triplet.index")
meta_trip = pickle.load(open("data/fclip_triplet_meta.pkl", "rb"))

idx_five  = faiss.read_index("data/fclip_cosine.index")
meta_five = pickle.load(open("data/fclip_meta.pkl", "rb"))

assert idx_trip.ntotal == len(meta_trip)
assert idx_five.ntotal == len(meta_five)

# ------------------------------------------------------------------ #
# helper functions
# ------------------------------------------------------------------ #

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Plain cosine similarity (assumes unit vectors)."""
    return float(np.dot(a, b))


def _hybrid(q_vec, q_style, c_vec, c_style, alpha: float = 0.3) -> float:
    """
    Blend visual and style similarities.
    alpha=0.3  → balanced 30 % visual / 70 % style.
    """
    return alpha * _cos(q_vec, c_vec) + (1.0 - alpha) * float(np.dot(q_style, c_style))

# ------------------------------------------------------------------ #
# public API
# ------------------------------------------------------------------ #

def recommend_combo(img_path: str | Path, k: int = 3, n: int = 1000):
    """Return *k* recommended outfit image paths + detected attributes."""
    # 1. encode query to 256-D
    pix = proc(images=Image.open(img_path).convert("RGB"),
               return_tensors="pt")["pixel_values"].to(DEVICE)

    with torch.no_grad():
        vec256 = proj_head(clip.get_image_features(pixel_values=pix)).cpu().numpy()
    vec256 /= np.linalg.norm(vec256)

    # 2. triplet-space neighbours (ordered unique)
    _, id_arr = idx_trip.search(vec256.astype("float32"), n)
    cand_ids  = list(dict.fromkeys(id_arr[0]))

    # 3. query attributes from 512-D features
    with torch.no_grad():
        feat512 = clip.get_image_features(pixel_values=pix)
        feat512 = feat512 / feat512.norm(dim=-1, keepdim=True)

        style_q = torch.softmax(five_head.h_style(feat512), dim=-1).cpu().squeeze().numpy()
        g_q = ckpt["genders"][torch.softmax(five_head.h_gender(feat512), dim=-1).argmax()]
        i_q = ckpt["items"  ][torch.softmax(five_head.h_item  (feat512), dim=-1).argmax()]
        c_q = ckpt["colours"][torch.softmax(five_head.h_colour(feat512), dim=-1).argmax()]
        s_q = ckpt["seasons"][torch.softmax(five_head.h_season(feat512), dim=-1).argmax()]

    # 4. layered filtering – colour kept for first two levels
    rules = [
        lambda m: m[1] == g_q and m[2] == i_q and m[3] == c_q and m[4] == s_q,
        lambda m: m[1] == g_q and m[2] == i_q and m[3] == c_q,
        lambda m: m[1] == g_q and m[2] == i_q,
        lambda m: True
    ]
    for rule in rules:
        filt_ids = [i for i in cand_ids if rule(meta_trip[i])]
        filt_ids = list(dict.fromkeys(filt_ids))
        if len(filt_ids) >= k:
            break
    if not filt_ids:
        return [], {}

    # 5. re-rank with hybrid score
    cand_vec512 = np.stack([idx_five.reconstruct(int(i)) for i in filt_ids])
    cand_style  = [meta_five[int(i)][-1] for i in filt_ids]

    scores = [_hybrid(feat512.cpu().numpy()[0], style_q, v, s)
              for v, s in zip(cand_vec512, cand_style)]
    topk  = np.argsort(scores)[::-1][:k]

    paths = [meta_five[int(filt_ids[i])][0].replace("\\", "/") for i in topk]

    # 6. attribute summary
    top3_idx = style_q.argsort()[::-1][:3]
    attrs = {
        "item_type" : i_q,
        "gender"    : g_q,
        "colour"    : c_q,
        "season"    : s_q,
        "style_top3": {ckpt["styles"][j]: float(style_q[j]) for j in top3_idx}
    }
    return paths, attrs
