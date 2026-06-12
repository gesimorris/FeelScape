# FeelScape

An AI-powered music generation system that analyzes the mood of an uploaded image and generates a matching LOFI track in MIDI and WAV formats.

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
<img width="1364" height="689" alt="Screenshot 2026-06-12 at 12 41 39 PM" src="https://github.com/user-attachments/assets/c732f057-08f6-4dac-9c9a-169d20d8eb15" />
<img width="1052" height="690" alt="Screenshot 2026-06-12 at 12 42 05 PM" src="https://github.com/user-attachments/assets/70c34e0f-e6ef-4909-965a-009382a2b76d" />
<img width="1033" height="676" alt="Screenshot 2026-06-12 at 12 43 32 PM" src="https://github.com/user-attachments/assets/261a55c2-b3b1-4b0c-b9a4-fb7cf0ceae9d" />
<img width="1258" height="276" alt="Screenshot 2026-06-12 at 12 45 02 PM" src="https://github.com/user-attachments/assets/1fcdebc7-ac6e-4289-a9da-9c140ac7cc23" />

## Sample Output

🎵 [Download / Play MP3]([demofeelscape.mp3](https://github.com/user-attachments/files/28896103/demofeelscape.mp3))

## Future Improvements

- Train on a larger image-music dataset
- Replace the fully connected network with a CNN-based feature extractor
- Generate multi-instrument arrangements instead of a single melody
- Incorporate transformer-based music generation
- Improve WAV synthesis with higher-quality instrument samples
