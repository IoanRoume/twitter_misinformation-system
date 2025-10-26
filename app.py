from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch
import os
from typing import List


HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "IoanRoume/twitter-misinformation-bertweet"

device = 0 if torch.cuda.is_available() else -1

print("Loading classification pipeline...")
classifier = pipeline("text-classification", model=MODEL, device=device)
print("Pipeline loaded.")


app = FastAPI(title="Misinformation Detection API", description="API for detecting misinformation in tweets using a fine-tuned BERTweet model.", version="1.0.1")

class TweetRequest(BaseModel):
    text: List[str]


class PredictionItem(BaseModel):
    text: str
    prediction: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]


@app.post("/predict", response_model=PredictionResponse)
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


if __name__ == "__main__":
    import uvicorn

    #Run the FastAPI app with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
