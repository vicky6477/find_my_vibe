#!/usr/bin/env python
"""
demo_find.py – CLI / notebook helper:
prints 5-head prediction and shows the k matches
"""
import argparse, pathlib, sys
from recommend_v2 import recommend
from PIL import Image

# ------------- CLI ----------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("image", help="query image path")
p.add_argument("-k", type=int, default=5, help="matches to show")
args = p.parse_args()

query = pathlib.Path(args.image)
if not query.exists():
    sys.exit(f"❌  file not found: {query}")

matches, attrs = recommend(str(query), k=args.k, return_attrs=True)

print("\nPredicted attributes")
print("--------------------")
print(" item_type :", attrs["item_type"])
print(" gender    :", attrs["gender"])
print(" colour    :", attrs["colour"])
print(" season    :", attrs["season"])
print(" style top3:")
for s,p in attrs["style_top3"].items():
    print(f"   {s:<12} {p:.2f}")

# ------------- try matplotlib grid -----------------------------------
try:
    import matplotlib.pyplot as plt

    n = len(matches)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    plt.figure(figsize=(3*ncols, 3*(nrows+1)))

    plt.subplot(nrows+1, ncols, 1)
    plt.imshow(Image.open(query))
    plt.title("Query", fontsize=9); plt.axis("off")

    for i,p in enumerate(matches, start=2):
        plt.subplot(nrows+1, ncols, i)
        plt.imshow(Image.open(p))
        plt.title(pathlib.Path(p).name, fontsize=7)
        plt.axis("off")

    plt.tight_layout(); plt.show()

except ImportError:
    print("\nMatches:")
    print("\n".join(matches))
