# FeelScape

An AI-powered LOFI music generator that analyzes the mood of an uploaded image 
and produces a matching LOFI beat as a WAV or MIDI file.

<p align="center">
  <img src="https://img.shields.io/badge/React-%2320232a.svg?logo=react&logoColor=%2361DAFB">
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=000">
  <img src="https://img.shields.io/badge/FastAPI-009485.svg?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=fff" />
</p>

## How It Works
Image Upload → Feature Extraction → Neural Network → MIDI Parameters → 
Simulated Annealing Optimization → MIDI File → WAV Audio

## Features
- Custom fully connected neural network built from scratch with NumPy
- Visual feature extraction (color, brightness, contrast, patterns)
- Simulated annealing optimization for melody refinement
- Data augmentation pipeline expanding 50 → 1000+ training pairs
- WAV and MIDI file download

## Live Demo
Link: https://feelscape-frontend.onrender.com/

## Run Locally
```bash
# Backend
cd backend
pip install fastapi uvicorn numpy opencv-python mido scikit-learn
python3 app.py

# Frontend
cd frontend
npm install && npm start
```

## Screenshots
