---
title: Email Classification API
emoji: 📧
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.22.0
app_file: app_streamlit.py
pinned: false
license: mit
---

# Email Classification System for Support Team

This project implements an email classification system that categorizes support emails into predefined categories while ensuring personal information (PII) is masked before processing.

## Features

1. **Email Classification**: Classifies support emails into different categories (e.g., Billing Issues, Technical Support, Account Management, etc.)
2. **Personal Information Masking**: Masks PII and PCI data before processing, including:
   - Full Name
   - Email Address
   - Phone number
   - Date of birth
   - Aadhar card number
   - Credit/Debit Card Number
   - CVV number
   - Card expiry number
3. **API Deployment**: Exposes the solution as an API

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd email_classifier
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Download required NLTK data**:
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

4. **Download required Spacy model**:
   ```
   python -m spacy download en_core_web_sm
   ```

5. **Run the API server**:
   ```
   uvicorn app:app --reload
   ```

## API Usage

The API accepts POST requests with email content and returns the classified category along with masked entities.

**Endpoint**: `/classify`

**Request Format**:
```json
{
  "email_body": "string containing the email"
}
```

**Response Format**:
```json
{
  "input_email_body": "string containing the email",
  "list_of_masked_entities": [
    {
      "position": [start_index, end_index],
      "classification": "entity_type",
      "entity": "original_entity_value"
    }
  ],
  "masked_email": "string containing the masked email",
  "category_of_the_email": "string containing the class"
}
```

## Project Structure

- `app.py`: Main application file with FastAPI implementation
- `models.py`: Contains model training and utility functions
- `utils.py`: Contains utility functions for PII masking and other operations
- `api.py`: API implementation
- `requirements.txt`: List of required packages

## Training Data

The system is trained on a dataset of support emails containing different types of requests and incidents.
