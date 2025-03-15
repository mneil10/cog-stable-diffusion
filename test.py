import os
import requests
import torch
from diffusers import StableDiffusionPipeline

# Model configuration
MODEL_ID = "stabilityai/stable-diffusion-2-1"  # Using the base model

def predict(prompt: str):
    print("Checking CUDA availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        print("Loading model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,  # Always use float16 for better GPU performance
        ).to(device)  # Explicitly move to CUDA
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Generating image...")
    image = pipe(prompt).images[0]
    
    output_path = "output.png"
    image.save(output_path)
    print(f"Image saved as {output_path}")

if __name__ == "__main__":
    prompt = "A futuristic cyberpunk city at night"
    print(f"Generating image for prompt: {prompt}")
    predict(prompt) 