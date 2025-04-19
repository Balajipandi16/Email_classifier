import streamlit as st
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk
import os
import json

# Set page configuration
st.set_page_config(
    page_title="Email Classification with PII Masking",
    page_icon="📧",
    layout="wide"
)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

download_nltk_data()

# Define regex patterns for PII detection
patterns = {
    "full_name": r'(?i)(?:my name is|I am|This is|name\'s|name is) ([A-Z][a-z]+ [A-Z][a-z]+)',
    "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "phone_number": r'(?:\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}|\+\d{1,3}[-\s]?\d{2}[-\s]?\d{3}[-\s]?\d{4}',
    "dob": r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2})\b',
    "aadhar_num": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    "credit_debit_no": r'\b(?:\d{4}[-\s]?){4}\b',
    "cvv_no": r'\bCVV:?\s*\d{3,4}\b|\b[Cc]vv\s*(?:number|code|no)?:?\s*\d{3,4}\b',
    "expiry_no": r'\b(?:0[1-9]|1[0-2])/\d{2,4}\b|\b(?:0[1-9]|1[0-2])-\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b'
}

# Sample data for training
sample_data = [
    {"email": "Subject: Unvorhergesehener Absturz der Datenanalyse-Plattform\n\nDie Datenanalyse-Plattform brach unerwartet ab, da die Speicheroberfläche zu gering war My name is Sophia Rossi.. Ich habe versucht, Laravel 8 und meinen MacBook Pro neu zu starten, aber das Problem behält sich bei. Ich benötige Ihre Unterstützung, um diesen Fehler zu beheben. You can reach me at janesmith@company.com.", "type": "Incident"},
    {"email": "Subject: Customer Support Inquiry\n\nSeeking information on digital strategies that can aid in brand growth and details on the available services. Looking forward to learning more to help our business grow My name is Elena Ivanova.. Thank you, and I look forward to hearing from you soon. You can reach me at fatima.farsi@help.com.", "type": "Request"},
    {"email": "Subject: Data Analytics for Investment\n\nI am contacting you to request information on data analytics tools that can be utilized with the Eclipse IDE for enhancing investment optimization. I am seeking suggestions for tools that can aid in making data-driven decisions. Particularly, I am interested in tools that can manage large datasets and offer advanced analytics features. These tools should be compatible with the Eclipse IDE and can smoothly integrate into my workflow You can reach me at liuwei@business.cn.. Key features I am interested in include data visualization, predictive modeling, and machine learning capabilities. I would greatly appreciate any recommendations or advice on how to begin with data analytics for investment optimization using the Eclipse IDE. My name is Elena Ivanova.", "type": "Request"},
    {"email": "Subject: Problem with Integration\n\nThe integration stopped working unexpectedly, causing synchronization errors between our platforms My name is Sophia Rossi.. This might be due to recent modifications to the API. Despite attempting to restart services and examining logs, the problem remains unresolved You can reach me at liuwei@business.cn.. I would be grateful for your help in addressing this issue.", "type": "Problem"},
    {"email": "Subject: Assistance Request\n\nDear Customer Support, I am writing in regards to a recently detected data blocker in our medical data sets. This issue might stem from outdated PHP versions My name is John Doe.. Despite updating the Elasticsearch and Terraform configurations, the problem persists. I would appreciate any assistance you can provide at your earliest convenience to resolve this matter. Kindly inform me of any additional information you require to continue the investigation and resolution. I look forward to hearing from you soon. Thank you for your time and support. Kind regards, [Your Name] You can reach me at maria.gonzalez@shop.es.", "type": "Problem"}
]

# Create and train a simple classifier
class SimpleEmailClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classes = []
        
    def train(self, emails, labels):
        # Extract text from emails
        texts = [self._preprocess_email(email) for email in emails]
        # Vectorize
        X = self.vectorizer.fit_transform(texts)
        # Train classifier
        self.classifier.fit(X, labels)
        # Store classes
        self.classes = list(set(labels))
        
    def predict(self, email_text):
        # Preprocess
        text = self._preprocess_email(email_text)
        # Vectorize
        X = self.vectorizer.transform([text])
        # Predict
        return self.classifier.predict(X)[0]
    
    def _preprocess_email(self, email_text):
        # Simple preprocessing
        text = email_text.lower()
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# Function to mask PII
def mask_pii(text):
    entities = []
    masked_text = text
    
    # Find matches for each pattern
    for entity_type, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            # For full_name, extract only the name part
            if entity_type == "full_name":
                # Extract the captured group (the actual name)
                start, end = match.span(1)
                entity_value = match.group(1)
            else:
                start, end = match.span()
                entity_value = match.group()
            
            entities.append({
                "position": [start, end],
                "classification": entity_type,
                "entity": entity_value
            })
    
    # Sort entities by start position in reverse order to avoid position shifts
    entities.sort(key=lambda x: x["position"][0], reverse=True)
    
    # Create masked text by replacing entities with placeholders
    for entity in entities:
        start, end = entity["position"]
        entity_type = entity["classification"]
        masked_text = masked_text[:start] + f"[{entity_type}]" + masked_text[end:]
    
    return masked_text, entities

# Initialize and train the classifier
@st.cache_resource
def initialize_classifier():
    classifier = SimpleEmailClassifier()
    
    # Try to load the original dataset first, fall back to sample data if not available
    CSV_FILE_PATH = "combined_emails_with_natural_pii.csv"
    
    if os.path.exists(CSV_FILE_PATH):
        try:
            st.write(f"Loading data from {CSV_FILE_PATH}")
            df = pd.read_csv(CSV_FILE_PATH)
            if 'email' in df.columns and 'type' in df.columns:
                emails = df['email'].tolist()
                labels = df['type'].tolist()
                st.write(f"Training on {len(emails)} emails from CSV file")
            else:
                st.write(f"CSV file doesn't have required columns. Using sample data instead.")
                emails = [item["email"] for item in sample_data]
                labels = [item["type"] for item in sample_data]
        except Exception as e:
            st.write(f"Error loading CSV file: {e}. Using sample data instead.")
            emails = [item["email"] for item in sample_data]
            labels = [item["type"] for item in sample_data]
    else:
        st.write("CSV file not found. Using sample data instead.")
        emails = [item["email"] for item in sample_data]
        labels = [item["type"] for item in sample_data]
    
    # Train the classifier
    classifier.train(emails, labels)
    return classifier

# Process email
def process_email(email_text):
    """Process email through classification and PII masking"""
    # Mask PII
    masked_email, masked_entities = mask_pii(email_text)
    
    # Classify the masked email
    category = classifier.predict(masked_email)
    
    # Create JSON output
    json_output = {
        "input_email_body": email_text,
        "list_of_masked_entities": masked_entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }
    
    return json_output

# Main Streamlit app
st.title("Email Classification with PII Masking")
st.markdown("""
This application classifies support emails into categories and masks personally identifiable information (PII).
""")

# Initialize classifier
classifier = initialize_classifier()

# Example emails
example_emails = [
    "Subject: Issue with Data Analytics Platform\n\nI am reaching out to report an issue with our data analytics platform. The platform has crashed, and we believe it might be due to inadequate RAM allocation. My name is John Smith. We have already tried restarting the server and reviewing the logs, but the problem still exists. You can reach me at john.smith@example.com.",
    "Subject: Customer Support Inquiry\n\nSeeking information on digital strategies that can aid in brand growth and details on the available services. Looking forward to learning more to help our business grow. My name is Elena Ivanova. Thank you, and I look forward to hearing from you soon. You can reach me at elena.ivanova@help.com."
]

# Create tabs
tab1, tab2 = st.tabs(["Classify Email", "About"])

with tab1:
    # Input area
    st.subheader("Enter Email Text")
    
    # Example selector
    example_option = st.selectbox(
        "Or select an example email:",
        ["Custom Input"] + example_emails
    )
    
    if example_option == "Custom Input":
        email_text = st.text_area("Email Content", height=200)
    else:
        email_text = example_option
        st.text_area("Email Content", value=email_text, height=200)
    
    # Process button
    if st.button("Classify Email"):
        if email_text:
            with st.spinner("Processing email..."):
                result = process_email(email_text)
                
                # Display results
                st.subheader("Classification Results")
                st.success(f"Category: {result['category_of_the_email']}")
                
                st.subheader("Masked Email")
                st.text_area("", value=result["masked_email"], height=200, disabled=True)
                
                st.subheader("Detected Entities")
                for entity in result["list_of_masked_entities"]:
                    st.markdown(f"- **{entity['classification']}**: {entity['entity']} at position {entity['position']}")
                
                st.subheader("JSON Output")
                st.json(result)
        else:
            st.error("Please enter an email text to classify.")

with tab2:
    st.subheader("About This Application")
    st.markdown("""
    ### Email Classification System with PII Masking
    
    This application classifies support emails into different categories while masking personally identifiable information (PII).
    
    #### Features
    
    1. **Email Classification**: Classifies emails into categories like Request, Problem, Incident
    2. **PII Masking**: Masks personal information like:
       - Full names
       - Email addresses
       - Phone numbers
       - Dates of birth
       - Aadhar card numbers
       - Credit/debit card numbers
       - CVV numbers
       - Card expiry dates
    
    #### API Format
    
    The application returns JSON in the following format:
    
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
    
    #### GitHub Repository
    
    The complete code for this project is available at: [https://github.com/Balajipandi16/email-classifier](https://github.com/Balajipandi16/email-classifier)
    """)
