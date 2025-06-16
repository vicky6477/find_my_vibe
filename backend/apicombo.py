from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib, shutil, os

# two retrieval modes
from backend.recommend import recommend                      # five‑head (attribute filter)
from backend.recommend_combo import recommend_combo          # triplet + five‑head

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# static mounts
app.mount("/fashion-dataset", StaticFiles(directory="fashion_dataset/images"), name="fashion-dataset")

app.mount(
    "/our-dataset",StaticFiles(directory="our_dataset/fashion_images"),       
    name="our-dataset")

app.mount("/frontend",         StaticFiles(directory="frontend"),              name="frontend")

@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")


# ---------------- prediction endpoint ----------------
@app.post("/predict/")
async def predict_style(
    file: UploadFile = File(...),
    k:   int  = Query(3,  ge=1, le=10, description="number of matches"),
    mode:str = Query("combo", enum=["combo", "fivehead"], description="retrieval strategy")
):
    """Upload an image and get top‑k recommendations.

    * **combo**   : Triplet recall  ➜  five‑head re‑rank (default, best quality)  
    * **fivehead**: Only five‑head filter + hybrid score (faster, no Triplet)"""
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # choose retrieval pipeline
    if mode == "combo":
        matches, attrs = recommend_combo(file_path, k=k, n=500)
    else:   # pure five‑head
        matches, attrs = recommend(file_path, k=k, return_attrs=True)

    # turn local paths into URLs that the front‑end <img> can load
    match_urls = [f"/fashion-dataset/{pathlib.Path(p).name}" for p in matches]

    return {
        "mode"           : mode,
        "prediction"     : attrs,       # 5‑head attributes (+ style top‑3)
        "recommendations": match_urls   # e.g. /fashion-dataset/50556.jpg
    }