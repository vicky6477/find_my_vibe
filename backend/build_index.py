import pathlib, pickle, time, numpy as np, faiss, torch
from tqdm import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ▸ ▸ ▸ EDIT THESE PATHS IF YOUR TREE DIFFERS ◂ ◂ ◂
CKPT_PATH = pathlib.Path("checkpoints/fclip_heads_5way.pth")
IMG_ROOT  = pathlib.Path("fashion_dataset/images")
INDEX_OUT = pathlib.Path("data/fclip_cosine.index")
META_OUT  = pathlib.Path("data/fclip_meta.pkl")

# ---------- 1.  Load checkpoint, backbone, processor, heads ----------
print("• Loading checkpoint …")
ckpt      = torch.load(CKPT_PATH, map_location="cpu")
backbone  = CLIPModel.from_pretrained(ckpt["fclip_id"])
processor = CLIPProcessor.from_pretrained(ckpt["fclip_id"])

from backend.train_fclip_multitask import FiveHead     # reuse the class
heads = FiveHead(backbone.config.projection_dim,
                 len(ckpt["styles"]), len(ckpt["items"]),
                 len(ckpt["genders"]), len(ckpt["colours"]),
                 len(ckpt["seasons"]))
heads.load_state_dict(ckpt["heads"])
backbone.eval(); heads.eval()

DEVICE = "cpu"
backbone.to(DEVICE); heads.to(DEVICE)

paths  = sorted(IMG_ROOT.glob("*.jpg"))
vecs   = np.empty((len(paths), backbone.config.projection_dim), dtype="float32")
meta   = []          # [(path, gender, item, colour, season, style_vec)]

st = time.time()
for i, p in enumerate(tqdm(paths, ncols=80, desc="Embedding")):
    img = Image.open(p).convert("RGB")
    pix = processor(images=img, return_tensors="pt")["pixel_values"].to(DEVICE)

    with torch.inference_mode():
        feat  = backbone.get_image_features(pix)           # (1,512)
        feat  = feat / feat.norm(p=2, dim=-1, keepdim=True)
        vecs[i] = feat.cpu().numpy()

        # attribute logits → hard labels / soft style
        style_probs = torch.softmax(heads.h_style(feat), dim=-1).cpu().squeeze().numpy()

        item_idx    = torch.softmax(heads.h_item  (feat), dim=-1).argmax().item()
        gender_idx  = torch.softmax(heads.h_gender(feat), dim=-1).argmax().item()
        colour_idx  = torch.softmax(heads.h_colour(feat), dim=-1).argmax().item()
        season_idx  = torch.softmax(heads.h_season(feat), dim=-1).argmax().item()

    meta.append((
        str(p),
        ckpt["genders"][gender_idx],
        ckpt["items"  ][item_idx],
        ckpt["colours"][colour_idx],
        ckpt["seasons"][season_idx],
        style_probs,
    ))

print(f"• Embeddings done in {time.time()-st:.1f}s")

# ---------- 3.  Build FAISS index (inner-product = cosine) -----------
print("• Building FAISS index …")
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)

INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(INDEX_OUT))

META_OUT.parent.mkdir(parents=True, exist_ok=True)
pickle.dump(meta, open(META_OUT, "wb"))
print(f"✔ Saved {index.ntotal:,} vectors to {INDEX_OUT}")
print(f"✔ Metadata → {META_OUT}")