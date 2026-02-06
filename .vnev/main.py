from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os

# Initialize App
app = FastAPI()

# Enable CORS (Allows your Android App to talk to this server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD MODEL (Global Variables) ---
# We load these once when the server starts to save time
print("Loading AI Model...")

# Helper function to find files safely
def get_model_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

try:
    # Load the specific Age Detection Model
    age_net = cv2.dnn.readNet(
        get_model_path("D:/USER/Antigravity/The_model/The_model/model/age_deploy.prototxt"), 
        get_model_path("D:/USER/Antigravity/The_model/The_model/model/age_net.caffemodel")
    )
    
    # Model Configuration Constants
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model files. Did you upload them? {e}")

@app.get("/")
def home():
    return {"status": "Online", "message": "ChronoFace AI is running"}

@app.post("/scan-face")
async def scan_face(file: UploadFile = File(...)):
    try:
        # 1. READ IMAGE
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Could not decode image"}

        # 2. PREPROCESS (Resize to 227x227 for the AI)
        blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # 3. PREDICT
        age_net.setInput(blob)
        predictions = age_net.forward()
        
        # 4. GET RESULT
        i = predictions[0].argmax()
        age = AGE_LIST[i]
        confidence = float(predictions[0][i])
        

        return {
            "success": True,
            "age": age,
            "confidence": confidence
        }

    except Exception as e:
        return {"success": False, "error": str(e)}