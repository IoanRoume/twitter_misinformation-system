from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from transformers import pipeline
import torch
import os
from typing import List
from PIL import Image
import io

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_TEXT = "IoanRoume/twitter-misinformation-bertweet"
MODEL_IMAGE = "IoanRoume/ai-image-classifier"
device = 0 if torch.cuda.is_available() else -1

print("Loading classification pipeline...")
classifier = pipeline("text-classification", model=MODEL_TEXT, device=device)
print("Pipeline loaded.")


print("Loading image classification pipeline...")
image_classifier = pipeline("image-classification", model=MODEL_IMAGE, device=device)
print("Image pipeline loaded.")

app = FastAPI(title="Misinformation Detection API", description="API for detecting misinformation in tweets using a fine-tuned BERTweet model.", version="1.0.1")

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

if __name__ == "__main__":
    import uvicorn

    #Run the FastAPI app with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
