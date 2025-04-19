import uvicorn
import os
import pandas as pd
import nltk
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api import app as api_app
from models import create_sample_dataset, train_model_from_data

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Create sample dataset and train model if they don't exist
SAMPLE_DATA_PATH = "sample_data.csv"
MODEL_PATH = "email_classifier_model.joblib"

if not os.path.exists(SAMPLE_DATA_PATH):
    print("Creating sample dataset...")
    create_sample_dataset(SAMPLE_DATA_PATH)

if not os.path.exists(MODEL_PATH):
    print("Training model...")
    classifier = train_model_from_data(SAMPLE_DATA_PATH, MODEL_PATH)

# Add CORS middleware
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add startup event
@api_app.on_event("startup")
async def startup_event():
    print("Email Classification API is starting up...")
    print(f"API is available at: http://localhost:8000")
    print(f"API documentation is available at: http://localhost:8000/docs")

# Main entry point
if __name__ == "__main__":
    uvicorn.run("app:api_app", host="0.0.0.0", port=8000, reload=True)
