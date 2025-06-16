from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib, shutil, os
import pandas as pd

from backend.recommend        import recommend          # five-head only
from backend.recommend_combo  import recommend_combo    # triplet + five-head

# ------------- FastAPI app setup -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------- static mounts ----------
app.mount("/fashion-dataset", StaticFiles(directory="fashion_dataset/images"), name="fashion-dataset")
app.mount("/our-dataset",     StaticFiles(directory="our_dataset/fashion_images"), name="our-dataset")
app.mount("/frontend",        StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def index():
    return FileResponse("frontend/index.html")

# helper: disk path → URL
def _to_url(p: str) -> str:
    if p.startswith("fashion_dataset/images"):
        return p.replace("fashion_dataset/images", "/fashion-dataset")
    if p.startswith("our_dataset/fashion_images"):
        return p.replace("our_dataset/fashion_images", "/our-dataset")
    return "/fashion-dataset/" + pathlib.Path(p).name

# ---------- valid fashion item types ----------
def load_valid_item_types(csv_path="fashion_dataset/styles.csv", top_n=100):
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df = df.dropna(subset=["articleType"])
    df["articleType"] = df["articleType"].str.lower()
    return set(df["articleType"].value_counts().head(top_n).index)

VALID_ITEM_TYPES = load_valid_item_types()

# ---------- single prediction endpoint ----------
@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    k:   int  = Query(3,  ge=1, le=10),
    mode:str = Query("combo", enum=["combo", "fivehead"])  
):
    """mode='combo' → same style；mode='fivehead' → similarity item"""
    os.makedirs("uploads", exist_ok=True)
    qpath = f"uploads/{file.filename}"
    with open(qpath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        if mode == "combo":
            matches, attrs = recommend_combo(qpath, k=k, n=1000)
        else:
            matches, attrs = recommend(qpath, k=k, return_attrs=True)

        # validate attributes
        item_type = attrs.get("item_type", "unknown").lower()
        confidence = attrs.get("item_confidence", 0.0)
        
        print(" Predict item_type:", item_type)
        print(" Predict confidence:", confidence)
     
        
 
        if confidence < 0.01 or item_type not in VALID_ITEM_TYPES:
            print(" Blocked due to low confidence or unknown item_type")
            raise HTTPException(status_code=400, detail="This image does not appear to be a fashion item. Please upload a valid fashion product.")

    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) 

    return {
        "mode": mode,
        "prediction": attrs,
        "recommendations": [_to_url(p) for p in matches]
    }
