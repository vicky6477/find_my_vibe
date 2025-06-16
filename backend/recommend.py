import torch, os
os.environ["OMP_NUM_THREADS"] = "1"        # avoid libomp pool
torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False      # <- critical

import numpy as np, faiss, pickle, torch
from PIL import Image, ImageFile
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor
from backend.train_fclip_multitask import FiveHead

ImageFile.LOAD_TRUNCATED_IMAGES = True   # ignore bad JPG headers

# ---------- 0. load models + index once ---------------------------------
CKPT   = Path("checkpoints/fclip_heads_5way.pth")
INDEX  = Path("data/fclip_cosine.index")
META   = Path("data/fclip_meta.pkl")

ckpt  = torch.load(CKPT, map_location="cpu")
backbone  = CLIPModel.from_pretrained(ckpt["fclip_id"])
processor = CLIPProcessor.from_pretrained(ckpt["fclip_id"])
heads = FiveHead(backbone.config.projection_dim,
                 len(ckpt["styles"]), len(ckpt["items"]),
                 len(ckpt["genders"]), len(ckpt["colours"]),
                 len(ckpt["seasons"]))
heads.load_state_dict(ckpt["heads"])
heads.eval()
backbone.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
backbone.to(DEVICE); heads.to(DEVICE)

index = faiss.read_index(str(INDEX))
meta  = pickle.load(open(META, "rb"))

# ---------- helpers -----------------------------------------------------
def _hybrid(q_vec, q_style, c_vec, c_style, alpha=0.7):
    return alpha * np.dot(q_vec, c_vec) + (1 - alpha) * np.dot(q_style, c_style)

# ---------- public API --------------------------------------------------
def recommend(img_path: str | Path, k: int = 5, return_attrs: bool = False):
    """
    Parameters
    ----------
    img_path : path to query image
    k        : number of matches
    return_attrs : if True → also return dict with 5-head prediction

    Returns
    -------
    list[str]             (if return_attrs=False)
    (list[str], dict)     (if return_attrs=True)
    """
    img = Image.open(img_path).convert("RGB")
    pix = processor(images=img, return_tensors="pt")["pixel_values"].to(DEVICE)

    with torch.inference_mode():
        feat = backbone.get_image_features(pix)        # (1,512)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        style_probs = torch.softmax(heads.h_style(feat), dim=-1).cpu().squeeze().numpy()

        item_idx   = torch.softmax(heads.h_item  (feat), dim=-1).argmax().item()
        gender_idx = torch.softmax(heads.h_gender(feat), dim=-1).argmax().item()
        colour_idx = torch.softmax(heads.h_colour(feat), dim=-1).argmax().item()
        season_idx = torch.softmax(heads.h_season(feat), dim=-1).argmax().item()

    q_vec  = feat.cpu().numpy()[0]
    q_item = ckpt["items"  ][item_idx]
    q_gender = ckpt["genders"][gender_idx]
    q_colour = ckpt["colours"][colour_idx]
    q_season = ckpt["seasons"][season_idx]

    # hard filter
    cand_ids = [i for i,(p,g,it,c,s,_) in enumerate(meta)
                if g==q_gender and it==q_item and c==q_colour and s==q_season]
    if not cand_ids:
        cand_ids = [i for i,(p,g,it,_,_,_) in enumerate(meta)
                    if g==q_gender and it==q_item]

    cand_vecs = index.reconstruct_n(0, index.ntotal)[cand_ids]
    cand_meta = [meta[i] for i in cand_ids]

    scores = [_hybrid(q_vec, style_probs, v, m[-1]) for v,m in zip(cand_vecs,cand_meta)]
    topk   = np.argsort(scores)[::-1][:k]
    matches = [cand_meta[i][0] for i in topk]

    if not return_attrs:
        return matches

    # package attribute dict
    top3 = style_probs.argsort()[::-1][:3]
    attr = {
        "style_top3": {ckpt["styles"][i]: float(style_probs[i]) for i in top3},
        "item_type" : q_item,
        "gender"    : q_gender,
        "colour"    : q_colour,
        "season"    : q_season,
    }
    matches = [m.replace("\\", "/") for m in matches]  

    return matches, attr