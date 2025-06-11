from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.recommend import recommend
from fastapi.responses import FileResponse
import shutil, os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/fashion-dataset", StaticFiles(directory="fashion_dataset/images"), name="fashion-dataset")

@app.post("/predict/")
async def predict_style(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results, style, item_type = recommend(file_path)
    return {
        "predicted_style": style,
        "item_type": item_type,
        "recommendations": results
    }

# Mount frontend as static site
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")