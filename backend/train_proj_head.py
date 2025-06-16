"""
This script fine-tunes a projection head on top of a frozen FashionCLIP image encoder using a triplet loss. 
It is designed to learn better visual embeddings for fashion images by bringing similar images closer and pushing dissimilar ones apart.

Key components:

Dataset Loader: Loads anchor, positive, and negative images from a CSV file and applies transforms.

Projection Head: A simple linear layer that maps CLIP’s image features into a lower-dimensional normalized space.

Loss Function: InfoNCE-based triplet loss that encourages anchor-positive pairs to be closer than anchor-negative pairs.

Training Loop: For each epoch, it extracts image features using the frozen CLIP encoder, projects them, calculates the loss, 
and updates only the projection head.


Example
-------
python -m backend.train_proj_head \
       --csv  our_dataset/train_triplets.csv \
       --imgs our_dataset/fashion_images \
       --val  our_dataset/val_triplets.csv \
       --out  checkpoints/proj_head_256.pth \
       --bs   16 --epochs 20 --workers 4

"""
import argparse, os, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from transformers import CLIPModel, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------------------------------------------------------ #
# 1. CLI
# ------------------------------------------------------------------ #
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  required=True, help="Path to styles.csv")
    p.add_argument("--imgs", required=True, help="Folder with *.jpg images")
    p.add_argument("--out",  default="Output checkpoint path (*.pth)")
    p.add_argument("--val",  default="fashion_dataset/val_triplets.csv",
                    help="Path to validation triplets CSV")
    p.add_argument("--bs",   type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr",   type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu",  action="store_true")
    return p.parse_args()

# ------------------------------------------------------------------ #
# 2. Triplet dataset
# ------------------------------------------------------------------ #
class TripletSet(Dataset):
    def __init__(self, csv, imgdir):
        self.tab  = pd.read_csv(csv)
        self.dir  = imgdir

    def __len__(self): return len(self.tab)

    def _path(self, name):
        for ext in (".jpg", ".png"):
            p = os.path.join(self.dir, name + ext)
            if os.path.exists(p): return p
        raise FileNotFoundError(name)

    def __getitem__(self, idx):
        a, p, n = self.tab.iloc[idx]
        return (Image.open(self._path(a)).convert("RGB"),
                Image.open(self._path(p)).convert("RGB"),
                Image.open(self._path(n)).convert("RGB"))

# ------------------------------------------------------------------ #
# 3. Projection head + loss
# ------------------------------------------------------------------ #
class ProjHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=256):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.normalize(self.fc(x), p=2, dim=-1)

def info_nce(a, p, n, t=0.07):
    sim_ap = (a * p).sum(-1) / t
    sim_an = (a * n).sum(-1) / t
    logits = torch.stack([sim_ap, sim_an], 1)
    labels = torch.zeros(a.size(0), dtype=torch.long, device=a.device)
    return F.cross_entropy(logits, labels)

#   collate function 
# ------------------------------------------------------------------ #
def triplet_collate(batch):
    """batch = [(a,p,n), (a,p,n), ...]  →  ([a1,a2,...], [p1,p2,...], [n1,n2,...])"""
    a, p, n = zip(*batch)        # tuple of tuples  → 3 tuples
    return list(a), list(p), list(n)

# ------------------------------------------------------------------ #
# 4. Train
# ------------------------------------------------------------------ #
def main(args):
    dev = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)

    MODEL_ID = "patrickjohncyh/fashion-clip"
    clip      = CLIPModel.from_pretrained(MODEL_ID).to(dev).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    for p in clip.parameters(): p.requires_grad = False

    head = ProjHead().to(dev)
    opt  = torch.optim.Adam(head.parameters(), lr=args.lr)

    def make_loader(csv, shuffle):
        return DataLoader(TripletSet(csv, args.imgs),
                          batch_size=args.bs,
                          shuffle=shuffle,
                          num_workers=args.workers,
                          pin_memory=(dev == "cuda"),
                          collate_fn=triplet_collate)

    tr_loader = make_loader(args.csv, shuffle=True)
    va_loader = make_loader(args.val, shuffle=False)

    def embed(pil_batch):
        inputs = processor(images=list(pil_batch), return_tensors="pt").to(dev)
        with torch.no_grad():
            feat = clip.get_image_features(**inputs)
        return feat

    for ep in range(args.epochs):
        head.train(); tr_loss = 0
        for a, p, n in tr_loader:
            loss = info_nce(head(embed(a)), head(embed(p)), head(embed(n)))
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()

        head.eval(); va_loss = 0
        with torch.no_grad():
            for a, p, n in va_loader:
                va_loss += info_nce(head(embed(a)), head(embed(p)), head(embed(n))).item()

        print(f"Epoch {ep+1:2d}/{args.epochs}  "
              f"train {tr_loss/len(tr_loader):.4f}  "
              f"val {va_loss/len(va_loader):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(head.state_dict(), args.out)
    print(" projection head saved →", args.out)

if __name__ == "__main__":
    main(cli())
