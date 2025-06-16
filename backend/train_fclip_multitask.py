"""
Train five independent linear heads on top of a *frozen* FashionCLIP backbone.

Heads / labels
--------------
style   ← CSV column  `usage`
item    ← CSV column  `articleType`
gender  ← CSV column  `gender`
colour  ← CSV column  `baseColour`
season  ← CSV column  `season`

Assumptions
-----------
• Images live at  {IMG_ROOT}/{id}.jpg   ( `id` column in the CSV ).
• CSV contains at least the columns listed above.
• GPU with ≥ 16 GB VRAM is available for batch=256; otherwise lower the batch.

Usage (from repo root)
----------------------
python -m backend.train_fclip_multitask \
       --csv    fashion_dataset/styles.csv \
       --imgs   fashion_dataset/images \
       --out    checkpoints/fclip_heads_5way.pth \
       --batch  256 --epochs 4 --workers 8
"""

import argparse, pathlib, time
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------- #
# 1.   CLI arguments
# ---------------------------------------------------------------------- #
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  required=True, help="Path to styles.csv")
    p.add_argument("--imgs", required=True, help="Folder with *.jpg images")
    p.add_argument("--out",  required=True, help="Output checkpoint path (*.pth)")
    p.add_argument("--batch",  type=int, default=256)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr",     type=float, default=3e-4)
    p.add_argument("--workers",type=int, default=8,
                   help="DataLoader workers (0 is safest on Windows notebooks)")
    p.add_argument("--cpu",    action="store_true", help="Force CPU training")
    return p.parse_args()

# ---------------------------------------------------------------------- #
# 2.   Dataset wrapper
# ---------------------------------------------------------------------- #
class FashionCSVDataset(Dataset):
    def __init__(self, df, processor,
                 styles, items, genders, colours, seasons):
        self.df       = df.reset_index(drop=True)
        self.processor= processor
        self.styles   = styles
        self.items    = items
        self.genders  = genders
        self.colours  = colours
        self.seasons  = seasons

    def __len__(self):                  
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        pix = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        return (
            pix,
            torch.tensor(self.styles .index(row.style)),
            torch.tensor(self.items  .index(row.item_type)),
            torch.tensor(self.genders.index(row.gender)),
            torch.tensor(self.colours.index(row.colour)),
            torch.tensor(self.seasons.index(row.season)),
        )


# ---------------------------------------------------------------------- #
# 3.   Five-head classifier
# ---------------------------------------------------------------------- #
class FiveHead(nn.Module):
    def __init__(self, dim, n_s, n_i, n_g, n_c, n_se):
        super().__init__()
        self.h_style  = nn.Linear(dim, n_s)
        self.h_item   = nn.Linear(dim, n_i)
        self.h_gender = nn.Linear(dim, n_g)
        self.h_colour = nn.Linear(dim, n_c)
        self.h_season = nn.Linear(dim, n_se)

    def forward(self, f):
        return (self.h_style (f),
                self.h_item  (f),
                self.h_gender(f),
                self.h_colour(f),
                self.h_season(f))

# ---------------------------------------------------------------------- #
# 4.   Train loop
# ---------------------------------------------------------------------- #
def main():
    args   = cli()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 4.1  Read CSV and build vocabularies
    df = (pd.read_csv(args.csv,on_bad_lines="skip")
            .rename(columns={"articleType":"item_type",
                             "usage":"style",
                             "baseColour":"colour"}))
    csv_cols = ["id","style","item_type","gender","colour","season"]
    missing  = [c for c in csv_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    img_root = pathlib.Path(args.imgs)
    df["path"] = df["id"].astype(str).apply(lambda x: img_root / f"{x}.jpg")
    df = df[df["path"].map(pathlib.Path.exists)].copy().reset_index(drop=True)
    print(f"Images found: {len(df):,}")

    global STYLES, ITEMS, GENDERS, COLOURS, SEASONS
    df[["style","item_type","gender","colour","season"]] = (
        df[["style","item_type","gender","colour","season"]].fillna('unknown').applymap(str.lower))

    STYLES  = sorted(df['style'].unique())
    ITEMS   = sorted(df.item_type.unique())
    GENDERS = sorted(df.gender.unique())
    COLOURS = sorted(df.colour.unique())
    SEASONS = sorted(df.season.unique())

    print("Vocab sizes –",
          f"styles {len(STYLES)}, items {len(ITEMS)}, gender {len(GENDERS)},",
          f"colours {len(COLOURS)}, seasons {len(SEASONS)}")

    # 4.2  Backbone & processor (frozen)
    MODEL_ID  = "patrickjohncyh/fashion-clip"
    backbone  = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    backbone.eval()

    # 4.3  Loader
    ds = FashionCSVDataset(
        df, processor,
        styles = STYLES,
        items  = ITEMS,
        genders= GENDERS,
        colours= COLOURS,
        seasons= SEASONS
     )
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=(device=="cuda"))

    # 4.4  Heads + optimiser
    heads = FiveHead(backbone.config.projection_dim,
                     len(STYLES), len(ITEMS), len(GENDERS),
                     len(COLOURS), len(SEASONS)).to(device)
    opt  = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=1e-4)
    ce   = nn.CrossEntropyLoss()

    # 4.5  Training epochs
    for epoch in range(args.epochs):
        t0, running = time.time(), 0.0
        heads.train()
        for pix, ys, yi, yg, yc, yse in tqdm(dl,
                 desc=f"epoch {epoch+1}/{args.epochs}", ncols=80):
            pix = pix.to(device, non_blocking=True)
            with torch.no_grad():
                feats = backbone.get_image_features(pix)          # frozen

            log_s, log_i, log_g, log_c, log_se = heads(feats)
            loss  = ce(log_s, ys .to(device))
            loss += ce(log_i, yi .to(device))
            loss += ce(log_g, yg .to(device))
            loss += ce(log_c, yc .to(device))
            loss += ce(log_se, yse.to(device))

            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()

        print(f"Epoch {epoch+1:2d}  loss={running/len(dl):.4f} "
              f"{time.time()-t0:.1f}s")

    # 4.6  Save checkpoint
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "fclip_id": MODEL_ID,
        "heads"   : heads.state_dict(),
        "styles"  : STYLES,
        "items"   : ITEMS,
        "genders" : GENDERS,
        "colours" : COLOURS,
        "seasons" : SEASONS,
    }, out_path)
    print("✔ saved →", out_path)

if __name__ == "__main__":
    main()