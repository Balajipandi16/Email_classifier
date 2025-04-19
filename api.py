from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import joblib
import os
from models import EmailClassifier
from utils import PIIMasker

# Define request and response models
class EmailRequest(BaseModel):
    email_body: str

class MaskedEntity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[MaskedEntity]
    masked_email: str
    category_of_the_email: str

# Initialize FastAPI app
app = FastAPI(
    title="Email Classification API",
    description="API for classifying support emails and masking PII",
    version="1.0.0"
)

# Load the classifier model
MODEL_PATH = "email_classifier_model.joblib"

# Initialize classifier and masker
classifier = EmailClassifier()
masker = PIIMasker()

# Load model if it exists
if os.path.exists(MODEL_PATH):
    classifier.load_model(MODEL_PATH)
else:
    # If model doesn't exist, create a sample dataset and train a model
    from models import create_sample_dataset
    
    SAMPLE_DATA_PATH = "sample_data.csv"
    if not os.path.exists(SAMPLE_DATA_PATH):
        create_sample_dataset(SAMPLE_DATA_PATH)
    
    from models import train_model_from_data
    classifier = train_model_from_data(SAMPLE_DATA_PATH, MODEL_PATH)

@app.post("/classify", response_model=EmailResponse)
async def classify_email(request: EmailRequest):
    """
    Classify an email and mask PII.
    
    Args:
        request (EmailRequest): Request containing the email body
        
    Returns:
        EmailResponse: Response with masked email and classification
    """
    try:
        email_body = request.email_body
        
        # Mask PII
        masked_email, masked_entities = masker.mask_pii(email_body)
        
        # Classify the masked email
        category = classifier.predict(masked_email)
        
        # Prepare response
        response = {
            "input_email_body": email_body,
            "list_of_masked_entities": masked_entities,
            "masked_email": masked_email,
            "category_of_the_email": category
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing email: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint that provides API information.
    
    Returns:
        Dict: API information
    """
    return {
        "message": "Email Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/classify": "POST endpoint to classify emails and mask PII"
        }
    }
