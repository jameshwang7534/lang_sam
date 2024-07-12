
import base64
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, File, HTTPException, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from PIL import Image, ImageFilter
from lang_sam import LangSAM
import numpy as np
import logging
import requests
import torch
import cv2


cors_options = {
    "allow_methods": ["*"],
    "allow_headers": ["*"],
    "allow_credentials": True,
    "allow_origins": [
        "https://www.photio.io",
        "https://dev.photio.io",
        "http://localhost:3000",
        "http://localhost",
        "http://172.30.1.10:3000",
    ],
}

app = FastAPI()
model = LangSAM()
app.add_middleware(CORSMiddleware, **cors_options)

class InpaintAPIRequest(BaseModel):
    image_url: str
    prompt: str

@app.get("/health")
async def health():
    return Response("pong")

@app.route("/ping", methods=["GET"])
async def ping(request):
    """
    Healthcheck function.
    """
    return Response("pong")

@app.get("/")
def read_root():
    return {"Hello": "World!"}


def get_mask_from_base_image(base_image_url, prompt):
    try:
        # Check if the input is a URL or a base64 string
        if base_image_url.startswith("http://") or base_image_url.startswith("https://"):
            response = requests.get(base_image_url)
            response.raise_for_status()
            image_pil = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image_data = base64.b64decode(base_image_url)
            image_pil = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Predict the segmentation mask using the LangSAM model
        masks, _, _, _ = model.predict(image_pil, prompt)
        
        # Aggregate all masks that correspond to the prompt
        final_mask = torch.zeros_like(masks[0])
        for mask in masks:
            final_mask = torch.max(final_mask, mask)  
        
        # Convert the mask to a numpy array and then to a PIL image
        mask_np = final_mask.numpy()  
        
        # Create an image where the object is black and the background is white
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8)).convert("L")
        
        dilated_mask_img = mask_img.filter(ImageFilter.MaxFilter(65))  # adjust the size of the mask
        
        white_background = Image.new("RGB", image_pil.size, (255, 255, 255))
        
        black_object_img = Image.composite(Image.new("RGB", image_pil.size, (0, 0, 0)), white_background, dilated_mask_img)
        
        return black_object_img

    except Exception as e:
        print(f"Error in get_mask_from_base_image: {e}")
        return None


@app.post("/generate-mask")
async def mask_image(request: InpaintAPIRequest):
    masked_image = get_mask_from_base_image(request.image_url, request.prompt)
    if masked_image is None:
        raise HTTPException(status_code=500, detail="Error processing the image")
    
    buffered = BytesIO()
    masked_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"masked_image": img_str}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)