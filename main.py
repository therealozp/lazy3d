from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, StreamingResponse
from PIL import Image
import io
import json
import asyncio

from test import generate_3d_assets_binary, generate_gemini

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/generate-image")
async def generate_image(
    prompt: str = Form(...),
    sketch: UploadFile = File(...),
):
    sketch_bytes = await sketch.read()
    sketch_img = Image.open(io.BytesIO(sketch_bytes)).convert("RGB").resize((512, 512))

    image_buffer = generate_gemini(prompt, sketch_img)
    return Response(content=image_buffer.getvalue(), media_type="image/png")


@app.post("/generate-model")
async def generate_model(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image_obj = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((512, 512))

    result = generate_3d_assets_binary(image_obj)
    return Response(content=result["glb"].getvalue(), media_type="model/gltf-binary")
