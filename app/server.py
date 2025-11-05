import os
import io
import threading
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .ascii_model import AsciiArtModel, InferenceResult


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


app = FastAPI(title="Dot/ASCII Art Generator")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

model_path = os.path.join(MODELS_DIR, "ascii_model.pt")
ascii_model = AsciiArtModel(model_save_path=model_path)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/convert")
async def convert_image(
    file: UploadFile = File(...),
    cv_guidance: bool = Query(True),
    guidance_strength: float = Query(2.0),
    mode: str = Query("palette")
):
    content = await file.read()
    try:
        result: InferenceResult = ascii_model.infer(content, cv_guidance=cv_guidance, guidance_strength=guidance_strength, mode=mode)

        png_bytes = io.BytesIO()
        result.output_image.save(png_bytes, format="PNG")
        png_bytes.seek(0)

        threading.Thread(target=ascii_model.online_train, args=(content,), daemon=True).start()

        return JSONResponse(
            {
                "ascii_text": result.ascii_text,
                "image_png_base64": result.png_base64,
                "cols": result.cols,
                "rows": result.rows,
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/train_pair")
async def train_pair(
    input: UploadFile = File(...),
    target: UploadFile = File(...),
    epochs: int = Query(3, ge=1, le=50),
    aug_per_epoch: int = Query(8, ge=1, le=64),
    lr: float = Query(1e-3, gt=0)
):
    try:
        input_bytes = await input.read()
        target_bytes = await target.read()
        def run():
            ascii_model.supervised_train_pair(input_bytes, target_bytes, epochs=epochs, lr=lr, aug_per_epoch=aug_per_epoch)
        threading.Thread(target=run, daemon=True).start()
        return JSONResponse({"status": "training_started", "epochs": epochs, "aug_per_epoch": aug_per_epoch})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/preview.png")
async def preview_png():
    from PIL import Image

    img = Image.new("RGB", (8, 8), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# Uvicorn entrypoint: uvicorn app.server:app --reload

