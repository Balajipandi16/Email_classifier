import requests
import json

def test_classification_api():
    """
    Test the email classification API with a sample email.
    """
    # API endpoint (update this if deployed elsewhere)
    url = "http://localhost:8000/classify"
    
    # Sample email with PII
    test_email = """Subject: Issue with Data Analytics Platform

I am reaching out to report an issue with our data analytics platform. The platform has crashed, 
and we believe it might be due to inadequate RAM allocation. My name is John Smith. 
We have already tried restarting the server and reviewing the logs, but the problem still exists. 
You can reach me at john.smith@example.com or call me at +1-555-123-4567.
My credit card number is 4111-1111-1111-1111 with CVV 123 and expiry date 05/2025.

We kindly request your assistance in investigating this matter and providing a resolution 
at your earliest convenience. Please inform us if any additional information from our side 
is necessary to address this issue. The platform is currently unavailable, and we urgently 
need it to be operational.

Thank you for your consideration and support."""
    
    # Prepare request payload
    payload = {"email_body": test_email}
    
    try:
        # Send POST request to API
        response = requests.post(url, json=payload)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            print("API Test Successful!")
            print("\nInput Email (truncated):")
            print(test_email[:100] + "...")
            
            print("\nMasked Email (truncated):")
            print(result["masked_email"][:100] + "...")
            
            print("\nDetected Entities:")
            for entity in result["list_of_masked_entities"]:
                print(f"- {entity['classification']}: {entity['entity']} at position {entity['position']}")
            
            print(f"\nClassified Category: {result['category_of_the_email']}")
            
            return True
        else:
            print(f"API Test Failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    except Exception as e:
        print(f"Error during API test: {str(e)}")
        return False

if __name__ == "__main__":
    test_classification_api()
