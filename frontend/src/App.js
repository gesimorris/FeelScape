// The User Interface for the Lofi Generator application.
// Built with React, it allows users to upload an image, generate a lofi beat, and download the resulting MIDI or WAV file.
//  The UI provides feedback on the generation process and handles errors gracefully.


import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { 
  Upload, Music, Download, RefreshCw, 
  Loader, CheckCircle, XCircle, Image as ImageIcon, ArrowLeft 
} from 'lucide-react';
import './App.css';


const API_BASE = process.env.NODE_ENV === 'production' 
  ? 'https://your-lofi-backend.onrender.com'
  : 'http://localhost:8000';

function App() {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [midiUrl, setMidiUrl] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [requestId, setRequestId] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [duration, setDuration] = useState(15);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setImage(file);
      setImagePreview(URL.createObjectURL(file));
      setError(null);
      setSuccess(false);
      setMidiUrl(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp'] },
    maxFiles: 1
  });

  const generateMusic = async () => {
    if (!image) {
      setError('Please upload an image first');
      return;
    }

    setIsGenerating(true);
    setError(null);
    setSuccess(false);

    const formData = new FormData();
    formData.append('file', image);
    formData.append('duration', duration);

    try {
      const response = await axios.post(`${API_BASE}/api/generate`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (response.data.success) {
        setMidiUrl(response.data.midi_url);
        setAudioUrl(response.data.audio_available ? response.data.audio_url : null);
        setRequestId(response.data.request_id);
        setSuccess(true);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Server error. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const triggerDownload = (path, fileName) => {
    const link = document.createElement('a');
    link.href = `${API_BASE}${path}`;
    link.setAttribute('download', fileName);
    document.body.appendChild(link);
    link.click();
    link.parentNode.removeChild(link);
  };

  const goBackToUpload = () => {
    setImage(null);
    setImagePreview(null);
    setError(null);
  };

  const reset = () => {
    setImage(null);
    setImagePreview(null);
    setMidiUrl(null);
    setAudioUrl(null);
    setRequestId(null);
    setError(null);
    setSuccess(false);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Music className="logo-icon" />
            <h1>Lofi Generator</h1>
          </div>
          <p className="tagline">Transform images into lofi beats with AI</p>
        </div>
      </header>

      <main className="main-content">
        <div className="container">
          
          {!imagePreview && (
            <div className="upload-section">
              <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
                <input {...getInputProps()} />
                <div className="dropzone-content">
                  <Upload className="dropzone-icon" size={64} />
                  <h2>Upload an Image</h2>
                  <p>{isDragActive ? 'Drop it!' : 'Drag & drop an image, or click to browse'}</p>
                </div>
              </div>
            </div>
          )}

          {imagePreview && !success && (
            <div className="preview-section">
              <button className="back-button" onClick={goBackToUpload}><ArrowLeft size={20} /> Back</button>
              <div className="image-preview-container">
                <img src={imagePreview} alt="Preview" className="image-preview" />
              </div>

              <div className="controls">
                <div className="duration-control">
                  <label>Duration: {duration}s</label>
                  <input type="range" min="10" max="60" value={duration} onChange={(e) => setDuration(parseInt(e.target.value))} disabled={isGenerating} />
                </div>
                <button className="generate-button" onClick={generateMusic} disabled={isGenerating}>
                  {isGenerating ? <><Loader className="spinner" size={20} /> Mapping Pixels...</> : <><Music size={20} /> Generate Beat</>}
                </button>
              </div>
              {error && <div className="alert alert-error"><XCircle size={20} /><span>{error}</span></div>}
            </div>
          )}

          {success && (
            <div className="success-section">
              <div className="success-header">
                <CheckCircle className="success-icon" size={48} />
                <h2>Beat Generated!</h2>
              </div>

              <div className="action-buttons">
                {audioUrl ? (
                  <button className="download-button" onClick={() => triggerDownload(audioUrl, `lofi_${requestId}.wav`)}>
                    <Download size={20} /> Download Audio (WAV)
                  </button>
                ) : (
                  <button className="download-button" onClick={() => triggerDownload(midiUrl, `lofi_${requestId}.mid`)}>
                    <Download size={20} /> Download MIDI
                  </button>
                )}

                {audioUrl && (
                  <button className="secondary-button" onClick={() => triggerDownload(midiUrl, `lofi_${requestId}.mid`)}>
                    <Download size={16} /> Get MIDI
                  </button>
                )}
                
                <button className="reset-button" onClick={reset}><RefreshCw size={20} /> New Image</button>
              </div>

              <div className="midi-info">
                <p>
                  {audioUrl 
                    ? "WAV generated. MIDI also available for DAW editing." 
                    : "Note: Audio conversion requires Fluidsynth. We've provided the high-quality MIDI file for you to use in any music software."}
                </p>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;