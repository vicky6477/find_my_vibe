from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib, shutil, os

# ← call the new helper that can return attrs + top-k
from backend.recommend import recommend

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# serve catalogue images and front-end
app.mount("/fashion-dataset", StaticFiles(directory="fashion_dataset/images"), name="fashion-dataset")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")

# ------------- prediction endpoint -----------------
@app.post("/predict/")
async def predict_style(file: UploadFile = File(...), k: int = 3):
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # get 3 matches + full 5-head prediction
    matches, attrs = recommend(file_path, k=k, return_attrs=True)

    # turn local paths into URLs that front-end <img> can load
    match_urls = [
    f"/fashion-dataset/{pathlib.Path(p).name}"   
    for p in matches
    ]


    return {
        "prediction"     : attrs,      # style_top3 + item_type + gender + colour + season
        "recommendations": match_urls  # e.g. /fashion-dataset/50556.jpg
    }
