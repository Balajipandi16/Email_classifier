import re
import spacy
import pandas as pd
from typing import List, Dict, Tuple, Any

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model not found, provide instructions
    print("Please download the spaCy model with: python -m spacy download en_core_web_sm")
    nlp = None

class PIIMasker:
    """
    Class for masking Personally Identifiable Information (PII) in text.
    Uses regex patterns and NER to identify and mask PII.
    """
    
    def __init__(self):
        # Define regex patterns for different types of PII
        self.patterns = {
            "full_name": r'(?i)(?:my name is|I am|This is|name\'s|name is) ([A-Z][a-z]+ [A-Z][a-z]+)',
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "phone_number": r'(?:\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}|\+\d{1,3}[-\s]?\d{2}[-\s]?\d{3}[-\s]?\d{4}',
            "dob": r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2})\b',
            "aadhar_num": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "credit_debit_no": r'\b(?:\d{4}[-\s]?){4}\b',
            "cvv_no": r'\bCVV:?\s*\d{3,4}\b|\b[Cc]vv\s*(?:number|code|no)?:?\s*\d{3,4}\b',
            "expiry_no": r'\b(?:0[1-9]|1[0-2])/\d{2,4}\b|\b(?:0[1-9]|1[0-2])-\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b'
        }
    
    def _find_matches(self, text: str) -> List[Dict[str, Any]]:
        """
        Find all PII matches in the text using regex patterns.
        
        Args:
            text (str): Input text to search for PII
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing entity information
        """
        entities = []
        
        # Find matches for each pattern
        for entity_type, pattern in self.patterns.items():
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
        
        # Sort entities by start position
        entities.sort(key=lambda x: x["position"][0])
        
        return entities
    
    def mask_pii(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Mask PII in the given text.
        
        Args:
            text (str): Input text to mask
            
        Returns:
            Tuple[str, List[Dict[str, Any]]]: Masked text and list of masked entities
        """
        entities = self._find_matches(text)
        
        # Create masked text by replacing entities with placeholders
        masked_text = text
        # Sort entities in reverse order to avoid position shifts
        for entity in sorted(entities, key=lambda x: x["position"][0], reverse=True):
            start, end = entity["position"]
            entity_type = entity["classification"]
            masked_text = masked_text[:start] + f"[{entity_type}]" + masked_text[end:]
        
        return masked_text, entities

class EmailPreprocessor:
    """
    Class for preprocessing email text for classification.
    """
    
    @staticmethod
    def extract_subject_body(email_text: str) -> Tuple[str, str]:
        """
        Extract subject and body from an email text.
        
        Args:
            email_text (str): Full email text
            
        Returns:
            Tuple[str, str]: Subject and body of the email
        """
        # Simple extraction based on "Subject:" prefix
        lines = email_text.strip().split('\n')
        subject = ""
        body = ""
        
        if lines and lines[0].startswith("Subject:"):
            subject = lines[0].replace("Subject:", "").strip()
            body = '\n'.join(lines[1:]).strip()
        else:
            # If no subject line is found, treat the whole text as body
            body = email_text
        
        return subject, body
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing extra whitespace and normalizing.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load email data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the email data
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()
