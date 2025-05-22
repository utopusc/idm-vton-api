#!/usr/bin/env python3
"""
IDM-VTON FastAPI Server
API wrapper for the IDM-VTON Gradio application
"""

import sys
import os
sys.path.append('./')

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import base64
import torch
import uvicorn
from typing import Optional
import asyncio
import json
import logging

# Import the IDM-VTON modules
from gradio_demo.app import start_tryon, pipe, openpose_model, parsing_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IDM-VTON API",
    description="Virtual Try-On API using IDM-VTON",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model management
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models_loaded = False

def load_models():
    """Load models to GPU if not already loaded"""
    global models_loaded
    if not models_loaded:
        logger.info("Loading models to GPU...")
        pipe.to(device)
        pipe.unet_encoder.to(device)
        openpose_model.preprocessor.body_estimation.model.to(device)
        models_loaded = True
        logger.info("Models loaded successfully!")

def unload_models():
    """Move models to CPU to free GPU memory"""
    global models_loaded
    if models_loaded:
        logger.info("Moving models to CPU...")
        pipe.to('cpu')
        pipe.unet_encoder.to('cpu')
        openpose_model.preprocessor.body_estimation.model.to('cpu')
        torch.cuda.empty_cache()
        models_loaded = False
        logger.info("Models moved to CPU!")

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()

def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("🚀 Starting IDM-VTON API Server...")
    logger.info(f"Device: {device}")
    # Don't load models immediately to save startup time
    # They will be loaded on first request

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    unload_models()
    logger.info("👋 API Server shutdown complete")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "IDM-VTON API Server is running!",
        "device": device,
        "models_loaded": models_loaded
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": models_loaded,
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    }

@app.post("/try-on")
async def try_on_endpoint(
    human_image: UploadFile = File(..., description="Human image file"),
    garment_image: UploadFile = File(..., description="Garment image file"),
    garment_description: str = Form(..., description="Description of the garment"),
    auto_mask: bool = Form(True, description="Use auto-generated mask"),
    auto_crop: bool = Form(False, description="Use auto-crop & resizing"),
    denoise_steps: int = Form(30, description="Number of denoising steps (20-40)"),
    seed: Optional[int] = Form(42, description="Random seed for reproducibility")
):
    """
    Virtual Try-On endpoint
    """
    try:
        # Load models if not already loaded
        if not models_loaded:
            load_models()
        
        # Validate inputs
        if denoise_steps < 20 or denoise_steps > 40:
            raise HTTPException(status_code=400, detail="denoise_steps must be between 20 and 40")
        
        # Read and process images
        human_img_data = await human_image.read()
        garment_img_data = await garment_image.read()
        
        human_img = Image.open(io.BytesIO(human_img_data)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_img_data)).convert("RGB")
        
        # Prepare the dictionary format expected by start_tryon
        human_dict = {
            "background": human_img,
            "layers": [Image.new("RGB", human_img.size, "black")] if not auto_mask else None,
            "composite": None
        }
        
        logger.info(f"Processing try-on request: {garment_description}")
        
        # Run the try-on process
        result_image, mask_image = start_tryon(
            dict=human_dict,
            garm_img=garment_img,
            garment_des=garment_description,
            is_checked=auto_mask,
            is_checked_crop=auto_crop,
            denoise_steps=denoise_steps,
            seed=seed
        )
        
        # Convert results to base64
        result_b64 = image_to_base64(result_image)
        mask_b64 = image_to_base64(mask_image)
        
        return {
            "success": True,
            "result_image": result_b64,
            "mask_image": mask_b64,
            "parameters": {
                "garment_description": garment_description,
                "auto_mask": auto_mask,
                "auto_crop": auto_crop,
                "denoise_steps": denoise_steps,
                "seed": seed
            }
        }
        
    except Exception as e:
        logger.error(f"Error in try-on process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Try-on failed: {str(e)}")

@app.post("/try-on-base64")
async def try_on_base64_endpoint(
    human_image_b64: str = Form(..., description="Human image as base64"),
    garment_image_b64: str = Form(..., description="Garment image as base64"),
    garment_description: str = Form(..., description="Description of the garment"),
    auto_mask: bool = Form(True, description="Use auto-generated mask"),
    auto_crop: bool = Form(False, description="Use auto-crop & resizing"),
    denoise_steps: int = Form(30, description="Number of denoising steps (20-40)"),
    seed: Optional[int] = Form(42, description="Random seed for reproducibility")
):
    """
    Virtual Try-On endpoint with base64 input/output
    """
    try:
        # Load models if not already loaded
        if not models_loaded:
            load_models()
        
        # Validate inputs
        if denoise_steps < 20 or denoise_steps > 40:
            raise HTTPException(status_code=400, detail="denoise_steps must be between 20 and 40")
        
        # Convert base64 to images
        human_img = base64_to_image(human_image_b64)
        garment_img = base64_to_image(garment_image_b64)
        
        # Prepare the dictionary format expected by start_tryon
        human_dict = {
            "background": human_img,
            "layers": [Image.new("RGB", human_img.size, "black")] if not auto_mask else None,
            "composite": None
        }
        
        logger.info(f"Processing try-on request: {garment_description}")
        
        # Run the try-on process
        result_image, mask_image = start_tryon(
            dict=human_dict,
            garm_img=garment_img,
            garment_des=garment_description,
            is_checked=auto_mask,
            is_checked_crop=auto_crop,
            denoise_steps=denoise_steps,
            seed=seed
        )
        
        # Convert results to base64
        result_b64 = image_to_base64(result_image)
        mask_b64 = image_to_base64(mask_image)
        
        return {
            "success": True,
            "result_image": result_b64,
            "mask_image": mask_b64,
            "parameters": {
                "garment_description": garment_description,
                "auto_mask": auto_mask,
                "auto_crop": auto_crop,
                "denoise_steps": denoise_steps,
                "seed": seed
            }
        }
        
    except Exception as e:
        logger.error(f"Error in try-on process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Try-on failed: {str(e)}")

@app.post("/unload-models")
async def unload_models_endpoint():
    """Unload models from GPU to free memory"""
    unload_models()
    return {"message": "Models unloaded from GPU", "models_loaded": models_loaded}

@app.post("/load-models")
async def load_models_endpoint():
    """Load models to GPU"""
    load_models()
    return {"message": "Models loaded to GPU", "models_loaded": models_loaded}

if __name__ == "__main__":
    # Configuration
    host = "0.0.0.0"
    port = 8000
    
    logger.info(f"🚀 Starting IDM-VTON API Server on {host}:{port}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )