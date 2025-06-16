"""
Build triplet index for fashion dataset
python -m backend.build_index_proj
"""
import pickle, numpy as np, faiss, torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from backend.train_proj_head import ProjHead


# ▸ ▸ ▸ EDIT THESE PATHS IF YOUR TREE DIFFERS ◂ ◂ ◂
CKPT_HEAD = Path("checkpoints/proj_head_256.pth")
IMG_DIRS = [
    Path("fashion_dataset/images"),
    Path("our_dataset/fashion_images")
]
OUT_VEC   = Path("data/fclip_triplet.index")
META      = Path("data/fclip_triplet_meta.pkl")

clip_id = "patrickjohncyh/fashion-clip"
clip    = CLIPModel.from_pretrained(clip_id).eval()
proc    = CLIPProcessor.from_pretrained(clip_id)

head = ProjHead()
head.load_state_dict(torch.load(CKPT_HEAD, map_location="cpu"))
head.eval()

DEVICE = ("mps"  if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available()        else
          "cpu")
clip.to(DEVICE); head.to(DEVICE)
print("device:", DEVICE)

# ----------- 2.  Collect all image paths -----------------------------
paths = []
for root in IMG_DIRS:
    paths += sorted(root.rglob("*.jpg"))
    paths += sorted(root.rglob("*.png"))

assert paths, f"No images under {IMDIR}"
print("images:", len(paths))

# ----------- 3.  Embed images in batches -----------------------------
BATCH = 64 if DEVICE != "cpu" else 8
vecs, meta = [], []
pbar = tqdm(total=len(paths), ncols=80, desc="Embedding")

for s in range(0, len(paths), BATCH):
    batch_paths = paths[s:s+BATCH]
    imgs = [Image.open(p).convert("RGB") for p in batch_paths]
    pix  = proc(images=imgs, return_tensors="pt", padding=True
                )["pixel_values"].to(DEVICE)

  
    with torch.inference_mode():
        feat = clip.get_image_features(pixel_values=pix)          # (B,512)
        proj = head(feat).cpu().numpy()                           # (B,256)
 

    vecs.append(proj.astype("float32"))
    meta.extend([p.as_posix() for p in batch_paths])
    pbar.update(len(batch_paths))

pbar.close()

vecs = np.concatenate(vecs, axis=0)
print("vecs shape:", vecs.shape)

# ---------- 4.  Build FAISS index (inner-product = cosine) -----------
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)

OUT_VEC.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(OUT_VEC))
with open(META, "wb") as f:
    pickle.dump(meta, f)

print(" Triplet index saved:", OUT_VEC)
print(" Metadata saved    :", META)
