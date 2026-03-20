"""
FastAPI Backend for Lofi Generator
Handles image uploads, MIDI generation, and file downloads
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import cv2
import os
import uuid
from pathlib import Path
from datetime import datetime
import shutil
from typing import Optional

from improved_model import ImprovedNeuralNetwork
from training_pipeline import extract_image_features
from midi_generation import generate_music_from_prediction
from midi_to_audio import convert_midi_to_wav
from sklearn.preprocessing import StandardScaler, MinMaxScaler


app = FastAPI(
    title="Lofi Generator API",
    description="Generate lofi beats from images using AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

model = None
scaler_x = None
scaler_y = None
model_loaded = False


def load_model_and_scalers():
    global model, scaler_x, scaler_y, model_loaded
    
    try:
        model_path = MODELS_DIR / "lofi_model.npy"
        if not model_path.exists():
            print("⚠️ Model file not found. Please train the model first.")
            return False
        
        model = ImprovedNeuralNetwork(
            input_size=6,
            hidden_sizes=[64, 128, 128, 64],
            output_size=20
        )
        model.load_model(str(model_path))
        
        scaler_x_path = MODELS_DIR / "scaler_x.npy"
        scaler_y_path = MODELS_DIR / "scaler_y.npy"
        
        if not scaler_x_path.exists() or not scaler_y_path.exists():
            print("Scaler files not found. Please train the model first.")
            return False
        
        scaler_x_data = np.load(scaler_x_path, allow_pickle=True).item()
        scaler_x = StandardScaler()
        scaler_x.mean_ = scaler_x_data['mean']
        scaler_x.scale_ = scaler_x_data['scale']
        scaler_x.var_ = scaler_x_data['var']
        scaler_x.n_features_in_ = len(scaler_x.mean_)
        
        scaler_y_data = np.load(scaler_y_path, allow_pickle=True).item()
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaler_y.min_ = scaler_y_data['min']
        scaler_y.scale_ = scaler_y_data['scale']
        scaler_y.data_min_ = scaler_y_data['data_min']
        scaler_y.data_max_ = scaler_y_data['data_max']
        scaler_y.n_features_in_ = len(scaler_y.min_)
        
        model_loaded = True
        print("Model and scalers loaded successfully!")
        return True
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.on_event("startup")
async def startup_event():
    load_model_and_scalers()


@app.get("/")
async def root():
    return {
        "message": "Lofi Generator API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_loaded
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/generate")
async def generate_music(
    file: UploadFile = File(...),
    duration: Optional[int] = 15,
    sa_iterations: Optional[int] = 3000
):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    request_id = str(uuid.uuid4())
    image_path = UPLOAD_DIR / f"{request_id}.jpg"

    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        image_features = extract_image_features(str(image_path))
        if image_features is None:
            raise Exception("Failed to extract features")

        image_features_scaled = scaler_x.transform(image_features.reshape(1, -1))
        predicted_music = model.predict(image_features_scaled)
        
        midi_filename = OUTPUT_DIR / f"{request_id}.mid"
        success = generate_music_from_prediction(
            predicted_music, scaler_y, str(midi_filename),
            sa_iterations=sa_iterations, target_duration=duration
        )
        
        if not success:
            raise Exception("MIDI generation failed")

        audio_filename = OUTPUT_DIR / f"{request_id}.wav"
        audio_success = False
        
        import shutil as system_shutil
        if system_shutil.which("fluidsynth"):
            try:
                print("🎵 Converting MIDI to audio...")
                audio_success = convert_midi_to_wav(str(midi_filename), str(audio_filename))
            except Exception as e:
                print(f"⚠️ Audio conversion failed: {e}")
        else:
            print("⚠️ Fluidsynth not found on server. Skipping WAV conversion.")

        if image_path.exists():
            image_path.unlink()
        
        response_data = {
            "success": True,
            "request_id": request_id,
            "midi_url": f"/outputs/{request_id}.mid",
            "filename": f"{request_id}.mid",
            "duration": duration,
            "audio_available": audio_success,
            "timestamp": datetime.now().isoformat()
        }
        
        if audio_success:
            response_data["audio_url"] = f"/outputs/{request_id}.wav"
            response_data["audio_filename"] = f"{request_id}.wav"
        
        return JSONResponse(response_data)
    
    except Exception as e:
        if image_path.exists():
            image_path.unlink()
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{request_id}")
async def download_midi(request_id: str):
    midi_path = OUTPUT_DIR / f"{request_id}.mid"
    
    if not midi_path.exists():
        raise HTTPException(
            status_code=404,
            detail="MIDI file not found"
        )
    
    return FileResponse(
        path=str(midi_path),
        media_type="audio/midi",
        filename=f"lofi_{request_id}.mid"
    )


@app.delete("/api/cleanup/{request_id}")
async def cleanup_files(request_id: str):
    midi_path = OUTPUT_DIR / f"{request_id}.mid"
    
    deleted = False
    if midi_path.exists():
        midi_path.unlink()
        deleted = True
    
    return {
        "success": True,
        "deleted": deleted,
        "request_id": request_id
    }


@app.post("/api/reload-model")
async def reload_model():
    success = load_model_and_scalers()
    
    if success:
        return {
            "success": True,
            "message": "Model reloaded successfully",
            "model_loaded": model_loaded
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to reload model"
        )


app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║            LOFI GENERATOR API SERVER                  ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Starting server on http://localhost:8000
    
    API Endpoints:
    - POST /api/generate       : Generate music from image
    - GET  /api/download/{id}  : Download generated MIDI
    - GET  /health            : Health check
    - POST /api/reload-model   : Reload trained model
    
    Documentation: http://localhost:8000/docs
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
