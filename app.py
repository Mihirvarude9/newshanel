from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
model_id = "stabilityai/stable-diffusion-3.5-medium"

# Ensure dtype consistency
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # Changed to float16 to match expected dtype
)

model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.float16  # Use float16 instead of bfloat16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=model_nf4,
    torch_dtype=torch.float16  # Ensure consistency
)
pipeline.enable_model_cpu_offload()

class ImageRequest(BaseModel):
    prompt: str
    steps: int = 40
    guidance_scale: float = 4.5

@app.post("/generate")
async def generate_image(request: ImageRequest):
    try:
        image = pipeline(
            prompt=request.prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
        ).images[0]
        
        # Convert PIL image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"image": img_str}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
