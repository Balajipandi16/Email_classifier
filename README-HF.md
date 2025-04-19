# Email Classification API

This API classifies support emails into different categories while masking personally identifiable information (PII).

## API Usage

### Endpoint: `/classify`

**Request Format**:
```json
{
  "email_body": "Your email content here"
}
```

**Response Format**:
```json
{
  "input_email_body": "Original email content",
  "list_of_masked_entities": [
    {
      "position": [start_index, end_index],
      "classification": "entity_type",
      "entity": "original_entity_value"
    }
  ],
  "masked_email": "Email with PII masked",
  "category_of_the_email": "Classified category"
}
```

## Features

1. **Email Classification**: Classifies emails into categories like Request, Problem, Incident
2. **PII Masking**: Masks personal information like names, emails, phone numbers, etc.
3. **API Interface**: Easy-to-use REST API

## Example

Try sending a POST request to the `/classify` endpoint with an email containing personal information to see how it works!
