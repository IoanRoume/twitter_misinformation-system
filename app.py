from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch
import os


HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "IoanRoume/twitter-misinformation-bertweet"

device = 0 if torch.cuda.is_available() else -1

print("Loading classification pipeline...")
classifier = pipeline("text-classification", model=MODEL, device=device)
print("Pipeline loaded.")


app = FastAPI(title="Misinformation Detection API", description="API for detecting misinformation in tweets using a fine-tuned BERTweet model.", version="1.0.0")

class TweetRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


@app.post("/predict", response_model=PredictionResponse)
async def predict_misinformation(request: TweetRequest):
    try:
        result = classifier(request.text, truncation=True, max_length=128)[0]
        
        return PredictionResponse(
            prediction=result['label'],
            confidence=result['score']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    #Run the FastAPI app with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
