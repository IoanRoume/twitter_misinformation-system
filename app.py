from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch
import os
# get path from env
SAVING_PATH = os.getenv("SAVING_PATH", "./finetuned_bertweet_model")

device = 0 if torch.cuda.is_available() else -1

print("Loading classification pipeline...")
classifier = pipeline("text-classification", model=SAVING_PATH, device=device)
print("Pipeline loaded.")


app = FastAPI(title="Misinformation Detection API", description="API for detecting misinformation in tweets using a fine-tuned BERTweet model.", version="1.0.0")

class TweetRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
#WITH BASIC AUTHENTICATION
@app.post("/predict", response_model=PredictionResponse, summary="Predict Misinformation in Tweet", description="Classify a tweet as misinformation or not.")
async def predict_misinformation(request: TweetRequest):
    result = classifier(request.text, truncation=True, padding=True)[0]
    prediction = "Misinformation" if result['label'] == 'LABEL_1' else "Legitimate"

    return PredictionResponse(
        prediction=prediction,
        confidence=result['score']
    )


if __name__ == "__main__":
    import uvicorn

    #Run the FastAPI app with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
