# Part 4 – LLM-Powered Feature: Model Prediction Explanation Pipeline

## Student Information

**Name:** Debojit Lahiri

---

# Chosen Track

**Track C – Model Prediction Explanation Pipeline**

This track extends the machine learning model developed in Part 3 by using a Large Language Model (LLM) to generate structured, human-readable explanations for model predictions. The best-performing model is loaded, predictions are generated, and the LLM explains each prediction in a validated JSON format.

---

# Objective

The objective of Part 4 is to integrate an LLM with the trained machine learning pipeline. The LLM receives feature values, the predicted class, and the prediction probability, then returns a structured JSON explanation describing the prediction. JSON schema validation and PII guardrails are implemented to ensure safe and reliable outputs.

---

# API Configuration

The LLM was accessed using the **OpenRouter API** through the Python `requests` library.

The API key was stored securely using an environment variable:

```python
os.environ["LLM_API_KEY"] = getpass("Enter your API Key: ")
```

No API keys were hardcoded in the notebook or committed to GitHub.

---

# Reusable LLM Function

A reusable function named `call_llm()` was implemented.

The function:

- Builds the HTTP POST request.
- Sends the system and user prompts.
- Sets the model, temperature, and maximum tokens.
- Checks the HTTP response status.
- Returns the assistant response.

A simple test prompt:

```
Reply with only the word: hello
```

successfully returned:

```
hello
```

confirming that the API connection worked correctly.

---

# System Prompt

The following system prompt was used.

```
You are an AI model explanation assistant.

Given:
- feature values
- predicted class
- prediction probability

Return ONLY valid JSON.

The JSON must contain exactly these fields:

{
"prediction_label":"",
"confidence_level":"",
"top_reason":"",
"second_reason":"",
"next_step":""
}

Do not include markdown.
Do not include explanations outside JSON.
```

---

# User Prompt Template

For each prediction the following information was sent to the LLM.

```
Feature values:
{feature_values}

Predicted Class:
{prediction}

Prediction Probability:
{probability}
```

The feature values were automatically converted into JSON before being included in the prompt.

---

# Why Temperature = 0?

Temperature was set to **0** because this task requires deterministic structured outputs.

At temperature 0 the model consistently selects the highest-probability token, producing predictable JSON responses that are easier to validate automatically.

Higher temperatures introduce randomness, making structured JSON generation less reliable.

---

# Temperature Comparison

Each prediction was evaluated twice.

| Temperature | Behaviour |
|------------|-----------|
| **0.0** | Deterministic, highly consistent JSON output |
| **0.7** | More varied wording and reasoning while maintaining the same prediction |

Temperature 0 consistently produced outputs that matched the expected schema with minimal variation, making it the preferred setting for structured explanation tasks.

---

# JSON Output Schema

The expected JSON schema contains five required scalar fields.

```json
{
  "prediction_label": "string",
  "confidence_level": "string",
  "top_reason": "string",
  "second_reason": "string",
  "next_step": "string"
}
```

After every LLM response:

1. Leading and trailing whitespace was removed.
2. The response was parsed using `json.loads()`.
3. The parsed object was validated using `jsonschema.validate()`.
4. Validation errors were handled using `try-except`.
5. If validation failed, a fallback dictionary containing null values was returned.

---

# Prediction Pipeline

The prediction pipeline consisted of the following steps.

1. Load the trained model using:

```python
joblib.load("best_model.pkl")
```

2. Generate predictions using:

```python
predict()
```

3. Generate prediction probabilities using:

```python
predict_proba()
```

4. Construct the LLM prompt.

5. Call the OpenRouter API.

6. Parse the returned JSON.

7. Validate the JSON against the predefined schema.

---

# PII Guardrail

Before every API request a regular expression checked the prompt for personally identifiable information.

The guardrail detects:

- Email addresses
- Phone numbers

Example:

| Input | Result |
|-------|--------|
| "Contact me at student@gmail.com" | **Blocked** |
| "This house has four bedrooms." | **Allowed** |

If PII is detected, the request is blocked and the following message is displayed:

```
Input blocked: PII detected.
```

This prevents sensitive personal information from being transmitted to the LLM.

---

# Demonstration Results

Three feature vectors were selected from the cleaned Ames Housing dataset.

For each sample:

- The trained model predicted the class.
- The prediction probability was computed.
- The LLM generated a structured explanation.
- The JSON schema validation succeeded.

| Sample | Prediction | JSON Validation | Guardrail |
|--------|-----------|----------------|-----------|
| Sample 1 | Successful | Pass | Allowed |
| Sample 2 | Successful | Pass | Allowed |
| Sample 3 | Successful | Pass | Allowed |

---

# Structured Output Validation

Each LLM response was successfully parsed into JSON.

All required fields were present.

Schema validation completed successfully for all three examples.

No fallback responses were required.

---

# Safety Considerations

Several safety measures were implemented.

- API keys stored securely using environment variables.
- No API keys committed to GitHub.
- PII detection before every API call.
- Strict JSON validation.
- Fallback handling for malformed responses.
- Deterministic prompting using temperature = 0.

These measures improve both reliability and security of the deployed pipeline.

---

# Conclusion

Part 4 successfully integrated a Large Language Model with the machine learning pipeline developed in Part 3.

The trained model generated predictions and prediction probabilities, while the LLM transformed those outputs into structured human-readable explanations. JSON schema validation ensured reliable machine-readable outputs, and the implemented PII guardrail prevented sensitive information from being transmitted.

This demonstrates how traditional machine learning models can be combined with modern LLMs to produce explainable, safe, and production-ready AI systems.
