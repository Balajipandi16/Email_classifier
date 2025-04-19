import pandas as pd
import numpy as np
import re
import pickle
import joblib
from typing import Dict, List, Any, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from utils import EmailPreprocessor

class EmailClassifier:
    """
    Class for classifying emails into predefined categories.
    Uses TF-IDF vectorization and Random Forest classifier.
    """
    
    def __init__(self):
        """Initialize the email classifier with a pipeline of TF-IDF vectorizer and Random Forest classifier."""
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        self.classes = []
        self.preprocessor = EmailPreprocessor()
    
    def preprocess_email(self, email_text: str) -> str:
        """
        Preprocess email text for classification.
        
        Args:
            email_text (str): Raw email text
            
        Returns:
            str: Preprocessed email text
        """
        subject, body = self.preprocessor.extract_subject_body(email_text)
        
        # Combine subject and body for classification
        combined_text = f"{subject} {body}"
        
        # Clean the text
        cleaned_text = self.preprocessor.clean_text(combined_text)
        
        return cleaned_text
    
    def train(self, emails: List[str], labels: List[str]) -> None:
        """
        Train the classifier on the provided emails and labels.
        
        Args:
            emails (List[str]): List of email texts
            labels (List[str]): List of corresponding labels
        """
        # Preprocess emails
        preprocessed_emails = [self.preprocess_email(email) for email in emails]
        
        # Store unique classes
        self.classes = list(set(labels))
        
        # Train the pipeline
        self.pipeline.fit(preprocessed_emails, labels)
        
        # Print training results
        print(f"Trained classifier with {len(emails)} emails and {len(self.classes)} classes.")
    
    def predict(self, email_text: str) -> str:
        """
        Predict the category of an email.
        
        Args:
            email_text (str): Email text to classify
            
        Returns:
            str: Predicted category
        """
        # Preprocess the email
        preprocessed_email = self.preprocess_email(email_text)
        
        # Make prediction
        prediction = self.pipeline.predict([preprocessed_email])[0]
        
        return prediction
    
    def evaluate(self, test_emails: List[str], test_labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate the classifier on test data.
        
        Args:
            test_emails (List[str]): List of test email texts
            test_labels (List[str]): List of corresponding test labels
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        # Preprocess test emails
        preprocessed_emails = [self.preprocess_email(email) for email in test_emails]
        
        # Make predictions
        predictions = self.pipeline.predict(preprocessed_emails)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            model_path (str): Path to save the model
        """
        joblib.dump(self.pipeline, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.pipeline = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

def train_model_from_data(data_path: str, model_save_path: str = 'email_classifier_model.joblib') -> EmailClassifier:
    """
    Train a model from a CSV file containing email data.
    
    Args:
        data_path (str): Path to the CSV file with email data
        model_save_path (str): Path to save the trained model
        
    Returns:
        EmailClassifier: Trained classifier
    """
    # Load data
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Check if required columns exist
    if 'email' not in df.columns or 'type' not in df.columns:
        print("Data must contain 'email' and 'type' columns")
        return None
    
    # Create and train classifier
    classifier = EmailClassifier()
    
    # Split data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Train the classifier
    classifier.train(train_df['email'].tolist(), train_df['type'].tolist())
    
    # Evaluate the classifier
    eval_metrics = classifier.evaluate(test_df['email'].tolist(), test_df['type'].tolist())
    print(f"Model accuracy: {eval_metrics['accuracy']:.4f}")
    
    # Save the model
    classifier.save_model(model_save_path)
    
    return classifier

# Function to create a sample dataset from the provided examples
def create_sample_dataset(output_path: str = 'sample_data.csv') -> None:
    """
    Create a sample dataset from the provided examples.
    
    Args:
        output_path (str): Path to save the sample dataset
    """
    # Sample data from the assignment
    sample_data = [
        {"email": "Subject: Unvorhergesehener Absturz der Datenanalyse-Plattform\n\nDie Datenanalyse-Plattform brach unerwartet ab, da die Speicheroberfläche zu gering war My name is Sophia Rossi.. Ich habe versucht, Laravel 8 und meinen MacBook Pro neu zu starten, aber das Problem behält sich bei. Ich benötige Ihre Unterstützung, um diesen Fehler zu beheben. You can reach me at janesmith@company.com.", "type": "Incident"},
        {"email": "Subject: Customer Support Inquiry\n\nSeeking information on digital strategies that can aid in brand growth and details on the available services. Looking forward to learning more to help our business grow My name is Elena Ivanova.. Thank you, and I look forward to hearing from you soon. You can reach me at fatima.farsi@help.com.", "type": "Request"},
        {"email": "Subject: Data Analytics for Investment\n\nI am contacting you to request information on data analytics tools that can be utilized with the Eclipse IDE for enhancing investment optimization. I am seeking suggestions for tools that can aid in making data-driven decisions. Particularly, I am interested in tools that can manage large datasets and offer advanced analytics features. These tools should be compatible with the Eclipse IDE and can smoothly integrate into my workflow You can reach me at liuwei@business.cn.. Key features I am interested in include data visualization, predictive modeling, and machine learning capabilities. I would greatly appreciate any recommendations or advice on how to begin with data analytics for investment optimization using the Eclipse IDE. My name is Elena Ivanova.", "type": "Request"},
        {"email": "Subject: Krankenhaus-Dienstleistung-Problem\n\nEin Medien-Daten-Sperrverhalten trat aufgrund unerlaubten Zugriffes auf You can reach me at fatima.farsi@help.com.. Eine schwache Passwortauswahl ist wahrscheinlich die Ursache My name is David Kim.. Nach dem Versuch, das Passwort zu ändern, behält das Problem leider anhaltend Bestand.", "type": "Incident"},
        {"email": "Subject: Security\n\nDear Customer Support, I am reaching out to inquire about the security protocols you have in place to protect medical data. As a valued customer, I want to ensure that my sensitive health information is handled with the utmost care and confidentiality My name is Maria Gonzalez.. Could you please provide me with information on the measures you take to safeguard medical data? Additionally, I would appreciate it if you could inform me about any certifications or compliances your company adheres to, such as HIPAA You can reach me at fatima.farsi@help.com.. I am also keen to know if you have any specific measures to prevent data breaches and unauthorized access. Thank you for your time and assistance. I look forward to your prompt response.", "type": "Request"},
        {"email": "Subject: Concerns About Securing Medical Data on 2-in-1 Convertible Laptop with Norton 360\n\nInquiring about best practices for securing medical data on a 2-in-1 Convertible Laptop. Currently have Norton 360 installed and need guidance on how to properly configure it to protect sensitive medical information You can reach me at fatima.farsi@help.com.. Would greatly appreciate any tips or recommendations you can offer. My name is Jane Smith.", "type": "Request"},
        {"email": "Subject: Ratung für Sicherung medizinischer Daten in HubSpot CRM PostgreSQL-Umgebungen\n\nRatung, ob es möglich ist, Sicherung medizinischer Daten in HubSpot CRM PostgreSQL-Umgebungen durchzuführen? Danke.. My contact number is +971-50-123-4567.", "type": "Request"},
        {"email": "Subject: Problem with Integration\n\nThe integration stopped working unexpectedly, causing synchronization errors between our platforms My name is Sophia Rossi.. This might be due to recent modifications to the API. Despite attempting to restart services and examining logs, the problem remains unresolved You can reach me at liuwei@business.cn.. I would be grateful for your help in addressing this issue.", "type": "Problem"},
        {"email": "Subject: Assistance Request\n\nDear Customer Support, I am writing in regards to a recently detected data blocker in our medical data sets. This issue might stem from outdated PHP versions My name is John Doe.. Despite updating the Elasticsearch and Terraform configurations, the problem persists. I would appreciate any assistance you can provide at your earliest convenience to resolve this matter. Kindly inform me of any additional information you require to continue the investigation and resolution. I look forward to hearing from you soon. Thank you for your time and support. Kind regards, [Your Name] You can reach me at maria.gonzalez@shop.es.", "type": "Problem"},
        {"email": "Subject: Support Request\n\nThe latest data analysis reports are inconsistent. Potential causes might be input errors due to recent software updates. Please check the input data and re-run the analysis You can reach me at sophia.rossi@service.it.. The issue persists. My name is David Kim.", "type": "Problem"},
        {"email": "Subject: Issue with Data Analytics Platform - Insufficient RAM Allocation\n\nI am reaching out to report an issue with our data analytics platform. The platform has crashed, and we believe it might be due to inadequate RAM allocation My name is Sophia Rossi.. We have already tried restarting the server and reviewing the logs, but the problem still exists You can reach me at david.kim@corp.kr.. We kindly request your assistance in investigating this matter and providing a resolution at your earliest convenience. Please inform us if any additional information from our side is necessary to address this issue. The platform is currently unavailable, and we urgently need it to be operational. Thank you for your consideration and support.", "type": "Problem"},
        {"email": "Subject: Reported Issue with Project Sync Resulting in Data Loss\n\nDear Customer Support,\n\nI am reporting an issue with our project sync, which has led to data loss. This sync failure occurred after recent software updates, possibly due to incompatibility issues My name is Maria Gonzalez.. Despite restarting our systems and reviewing firewall settings, the problem remains unresolved.\n\nThe software updates were installed recently, and the sync failure happened shortly thereafter. Our team has attempted to troubleshoot the issue by restarting the systems and reviewing the firewall settings, but these efforts have been unsuccessful.\n\nI would greatly appreciate it if you could investigate this matter and provide a solution to resolve the sync issue and recover the lost data You can reach me at liuwei@business.cn.. If there is any additional information or logs you require from our side, please let me know.\n\nThank you for your time and assistance.\n\nBest regards,\n[Your Name]", "type": "Incident"}
    ]
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created and saved to {output_path}")
