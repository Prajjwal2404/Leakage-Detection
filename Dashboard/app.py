import os
import sys
import uvicorn
import numpy as np
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from gru_ae_infer import WaterLeakageDetector

sys.path.append('..')
from Models.data_preprocess import load_and_preprocess_data


app = FastAPI(title="Water Leakage Dashboard")

# Initialize the detector
try:
    detector = WaterLeakageDetector(checkpoint_dir='..\\Checkpoints')
except Exception as e:
    print(f"Warning: Could not load detector models on startup: {e}")
    detector = None

# Load dataset for mock data generation
mock_X = None
try:
    scada_file = '..\\Dataset\\2018_SCADA.xlsx'
    leak_file = '..\\Dataset\\2018_Leakages.csv'
    
    if os.path.exists(scada_file) and os.path.exists(leak_file):
        print("Loading dataset for mock data generation...")
        mock_X, _, is_Nighttime = load_and_preprocess_data(scada_file, leak_file, seq=True)   
        print(f"Dataset loaded. Available samples: {len(mock_X)}")
    else:
        print("Dataset files not found. Using fallback random data logic.")
except Exception as e:
    print(f"Error loading real dataset: {e}")

class InferenceRequest(BaseModel):
    features: List[float]
    is_nighttime: int

@app.post("/predict")
def predict(request: InferenceRequest):
    if not detector:
        raise HTTPException(status_code=500, detail="Detector model not initialized. Ensure checkpoints exist.")
    
    if len(request.features) != 40:
        raise HTTPException(status_code=400, detail="Expected exactly 40 features.")
        
    try:
        is_leak, mse, thresh = detector.predict(request.features, request.is_nighttime)
        return {
            "is_leak": is_leak,
            "error": float(mse),
            "threshold": float(thresh)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mock_data")
def get_mock_data():
    """Generates mock data for frontend demonstration by sampling real dataset."""
    if mock_X is not None and len(mock_X) > 0:
        # Sample a random row from the loaded dataset
        idx = np.random.randint(0, len(mock_X))
        mock = mock_X[idx].tolist()
        is_nighttime = int(is_Nighttime[idx])
        return {"features": mock, "is_nighttime": is_nighttime}
    else:
        # Fallback to random data if dataset failed to load
        mock = np.random.randn(40).tolist()
        mock[-1] = np.random.choice([0, 1]) 
        if np.random.rand() > 0.9:
            mock = (np.array(mock) * 3).tolist()
            
        return {"features": mock, "is_nighttime": mock[-1]}

# Mount static files to serve the frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
