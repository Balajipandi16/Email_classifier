# Email Classification System for Support Team - Technical Report

## 1. Introduction to the Problem Statement

This project implements an email classification system for a company's support team. The system categorizes incoming support emails into predefined categories (such as Request, Problem, Incident) while ensuring that personally identifiable information (PII) is masked before processing. After classification, the masked data is restored to its original form.

The key functionalities of the system include:
- Email classification using machine learning techniques
- Personal information masking without using LLMs
- API deployment for easy integration

The system is designed to help support teams efficiently process and route incoming emails while maintaining data privacy and security.

## 2. Approach for PII Masking and Classification

### 2.1 PII Masking Approach

For PII masking, we implemented a rule-based approach using regular expressions (regex) to identify and mask sensitive information. This approach was chosen because:

1. **Efficiency**: Regex patterns can quickly identify common PII patterns without requiring complex models.
2. **No LLM dependency**: As per requirements, the masking process doesn't rely on Large Language Models.
3. **Customizability**: Patterns can be easily updated to accommodate new types of PII.

The system identifies and masks the following types of PII:
- Full names
- Email addresses
- Phone numbers
- Dates of birth
- Aadhar card numbers
- Credit/debit card numbers
- CVV numbers
- Card expiry dates

Each detected entity is replaced with a placeholder (e.g., `[full_name]`) in the masked text, while the original information is stored for later restoration.

### 2.2 Classification Approach

For email classification, we implemented a machine learning approach using:
1. **TF-IDF Vectorization**: Converts text into numerical features by calculating term frequency-inverse document frequency.
2. **Random Forest Classifier**: An ensemble learning method that builds multiple decision trees and merges their predictions.

This approach was chosen because:
1. **Effectiveness**: Random Forest works well with high-dimensional text data.
2. **Interpretability**: The model provides feature importance, helping understand what words contribute to classification decisions.
3. **Robustness**: Less prone to overfitting compared to single decision trees.

## 3. Model Selection and Training Details

### 3.1 Model Architecture

The classification pipeline consists of:
1. **Text Preprocessing**: Extracting subject and body, cleaning text, and normalizing.
2. **Feature Extraction**: Using TF-IDF vectorization with n-grams (1-2) to capture word sequences.
3. **Classification**: Random Forest classifier with 100 estimators.

### 3.2 Training Process

The model is trained using the following steps:
1. **Data Loading**: Loading email data from CSV file.
2. **Data Splitting**: Splitting into 80% training and 20% testing sets.
3. **Model Training**: Fitting the TF-IDF vectorizer and Random Forest classifier on the training data.
4. **Model Evaluation**: Assessing performance on the test set using accuracy and classification report.
5. **Model Persistence**: Saving the trained model for later use.

### 3.3 Alternative Models Considered

While we implemented a Random Forest classifier, we also considered:
1. **Naïve Bayes**: Simple probabilistic classifier based on Bayes' theorem.
2. **Support Vector Machines (SVM)**: Effective for text classification but can be slower to train.
3. **Deep Learning Models**: Such as LSTM or Transformer-based models, which could capture more complex patterns but require more data and computational resources.

## 4. Challenges Faced and Solutions Implemented

### 4.1 Challenges in PII Detection

**Challenge**: Creating regex patterns that can identify various formats of PII without excessive false positives.
**Solution**: Developed carefully crafted patterns with context awareness (e.g., looking for phrases like "my name is" before names) and implemented validation checks.

### 4.2 Handling Multilingual Content

**Challenge**: The dataset contains emails in multiple languages (English and German).
**Solution**: Focused on language-agnostic features for classification and ensured PII detection patterns work across languages where possible.

### 4.3 Maintaining Original Text Structure

**Challenge**: Ensuring that the masked text maintains the same structure as the original for accurate classification.
**Solution**: Implemented a position-based replacement system that preserves text structure while masking sensitive information.

### 4.4 API Performance

**Challenge**: Ensuring the API can handle multiple requests efficiently.
**Solution**: Implemented asynchronous request handling with FastAPI and optimized the classification pipeline for speed.

## 5. Conclusion and Future Improvements

The implemented email classification system successfully categorizes support emails while protecting sensitive information. The system achieves this through a combination of regex-based PII masking and machine learning-based classification.

Future improvements could include:
1. **Enhanced PII Detection**: Incorporating more sophisticated techniques like conditional random fields (CRFs) for better entity recognition.
2. **Multilingual Support**: Adding language detection and language-specific processing.
3. **Model Retraining**: Implementing periodic model retraining as new data becomes available.
4. **Feedback Loop**: Adding a mechanism for support staff to provide feedback on classification results to improve the model over time.

The system is designed to be maintainable and extensible, allowing for easy updates and improvements as requirements evolve.
