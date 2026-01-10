from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from transformers import pipeline
import torch
import os
from typing import List
from PIL import Image
import io
import tempfile
import librosa

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_TEXT = "IoanRoume/twitter-misinformation-bertweet"
MODEL_IMAGE = "IoanRoume/ai-image-classifier"
MODEL_AUDIO = "IoanRoume/ai-audio-classifier"

device = 0 if torch.cuda.is_available() else -1

print("Loading classification pipeline...")
classifier = pipeline("text-classification", model=MODEL_TEXT, device=device)
print("Pipeline loaded.")


print("Loading image classification pipeline...")
image_classifier = pipeline("image-classification", model=MODEL_IMAGE, device=device)
print("Image pipeline loaded.")



print("Loading audio classification pipeline...")
audio_classifier = pipeline("audio-classification", model=MODEL_AUDIO, device=device)
print("Audio pipeline loaded.")

app = FastAPI(
    title="Multimodal Misinformation Detection API",
    description="API for detecting misinformation in text, images, and audio.",
    version="3.0.0"
)
class TweetRequest(BaseModel):
    text: List[str]


class PredictionItem(BaseModel):
    text: str
    prediction: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]


@app.post("/predict/text", response_model=PredictionResponse)
async def predict_misinformation(request: TweetRequest):
    try:
        results = classifier(request.text, truncation=True, max_length=128)

        predictions = [
            PredictionItem(
                text=text,
                prediction=result['label'],
                confidence=result['score']
            ) for text, result in zip(request.text, results)
        ]
        
        return PredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


class ImagePredictionItem(BaseModel):
    filename: str
    prediction: str
    confidence: float


class ImagePredictionResponse(BaseModel):
    predictions: List[ImagePredictionItem]



@app.post("/predict/image")
async def predict_image(files: List[UploadFile] = File(...)):
    try:
        predictions = []

        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            result = image_classifier(image)[0]

            predictions.append(
                ImagePredictionItem(
                    filename=file.filename,
                    prediction=result['label'],
                    confidence=result['score']
                )
            )
        
        return ImagePredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image prediction failed: {str(e)}")


@app.post("/predict/image/base64")
async def predict_image_base64(request: dict):
    import base64
    
    try:
        predictions = []
        
        for img_data in request.get("images", []):
            # Decode base64 image
            image_bytes = base64.b64decode(img_data["data"])
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Classify image
            result = image_classifier(image)[0]
            
            predictions.append(
                ImagePredictionItem(
                    filename=img_data.get("name", "unknown"),
                    prediction=result['label'],
                    confidence=result['score']
                )
            )
        
        return ImagePredictionResponse(predictions=predictions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base64 image prediction failed: {str(e)}")


class AudioPredictionItem(BaseModel):
    filename: str
    prediction: str
    confidence: float
    duration: float

class AudioPredictionResponse(BaseModel):
    predictions: List[AudioPredictionItem]

@app.post("/predict/audio")
async def predict_audio(files: List[UploadFile] = File(...)):
    try:
        predictions = []
        for file in files:
            audio_bytes = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            try:
                audio_array, sampling_rate = librosa.load(tmp_path, sr= 16000)
                duration = len(audio_array)/ sampling_rate
                result = audio_classifier(audio_array,sampling_rate=sampling_rate)[0]
                predictions.append(
                    AudioPredictionItem(
                        filename=file.filename,
                        prediction=result['label'],
                        confidence=result['score'],
                        duration=round(duration, 2)
                    )
                )
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
        
        return AudioPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    #Run the FastAPI app with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
