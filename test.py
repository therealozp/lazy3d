# test.py
import torch
from PIL import Image
import io
import imageio
import rembg

from dotenv import load_dotenv

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# -------------------- CONFIG --------------------
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_scribble"
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
TRELLIS_MODEL = "JeffreyXiang/TRELLIS-image-large"

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL, torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL, controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()
load_dotenv(dotenv_path=".env.local")  # or just load_dotenv() if using `.env`

# negative_prompt = "bad anatomy, bad hands, error, blurry, bad image, low quality, incorrect proportions, bad proportions, ugly, deformed, distorted, bad perspective, watermark"

import torch
import numpy as np
from PIL import Image
import io
import rembg
import cv2


from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

import base64
import os
import mimetypes
from google import genai
from google.genai import types


# -------- CONFIG --------
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SCRIBBLE_MODEL = "lllyasviel/control_v11p_sd15_scribble"
DEPTH_MODEL = "lllyasviel/control_v11f1p_sd15_depth"
TRELLIS_MODEL = "JeffreyXiang/TRELLIS-image-large"


# -------------------- CONTROLNET --------------------
def generate_image_with_controlnet(prompt: str, image: Image.Image) -> io.BytesIO:

    result = pipe(
        prompt=prompt,
        image=image,
        # negative_prompt=negative_prompt,
        num_inference_steps=40,
        controlnet_conditioning_scale=0.8,  # fine-tuned weights
        guidance_scale=9.0,
    ).images[0]

    # Convert to bytes
    with io.BytesIO() as input_buffer:
        result.save(input_buffer, format="PNG")
        input_buffer.seek(0)
        no_bg_bytes = rembg.remove(input_buffer.read())  # remove background
        output_buffer = io.BytesIO(no_bg_bytes)

        # no bg removal
        # output_buffer = io.BytesIO(input_buffer.read())

    # Return final image as buffer
    output_buffer.seek(0)
    return output_buffer


# generate Gemini Image from sketch and prompt
def generate_gemini(prompt: str, sketch: Image.Image) -> io.BytesIO:
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    print("client created successfully.")
    try:
        encoded_string = base64.b64encode(sketch.read()).decode("utf-8")
        image = Image.open(io.BytesIO(base64.b64decode(encoded_string)))  # loading the
        # Create a list with both the prompt and image
        contents = [prompt, image]
    except:
        contents = [prompt]

    print("contents created successfully.")
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=contents,
        config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
    )
    print(response)

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            image = io.BytesIO((part.inline_data.data))
            no_bg_bytes = rembg.remove(image.read())  # remove background
            output_buffer = io.BytesIO(no_bg_bytes)

        # no bg removal
        # output_buffer = io.BytesIO(input_buffer.read())

        # Return final image as buffer
        output_buffer.seek(0)
        return output_buffer

    print("No image found in response.")
    return None


# -------------------- TRELLIS --------------------
def generate_3d_assets_binary(image: Image.Image) -> dict:
    pipeline = TrellisImageTo3DPipeline.from_pretrained(TRELLIS_MODEL)
    pipeline.cuda()

    outputs = pipeline.run(
        image,
        seed=1,
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 3,
        },
    )

    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0], outputs["mesh"][0], simplify=0.95, texture_size=1024
    )
    glb_buffer = io.BytesIO()
    glb.export(glb_buffer, file_type="glb")
    glb_buffer.seek(0)

    return {"glb": glb_buffer}
