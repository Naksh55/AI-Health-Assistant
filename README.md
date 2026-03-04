# AI Health Assistant

This repository contains a simple medical chatbot built with Python and NLTK.
It uses a JSON-based disease knowledge base to match user-described symptoms
against conditions and provide advice. The chatbot has been enhanced to handle
synonyms, fuzzy matching, and a comprehensive list of diseases including mental
health conditions.

## Features

- Symptom preprocessing using NLTK tokenizer
- Synonym dictionary and fuzzy matching for improved symptom detection
- Weighted disease matching based on symptom count, percentage, and severity
- Extensive `diseases.json` with 100+ conditions (including mental illnesses)
- CLI chatbot interface and test scripts

## Files

- `Main.py` - core chatbot logic
- `diseases.json` - structured database of diseases, symptoms, advice, severity
- `test_chatbot.py` - general symptom matching tests
- `test_mental_health.py` - tests focusing on mental health conditions

## Usage

1. Install dependencies (e.g., `pip install nltk`).
2. Run `python Main.py` to start the CLI chatbot.
3. Enter symptoms; type `exit` or `quit` to stop.
4. Use test scripts to verify matching behavior.

## Extending

- Add or modify entries in `diseases.json`.
- Update `SYMPTOM_SYNONYMS` in `Main.py` for better recognition.
- Improve NLP using spaCy or larger language models.

## Disclaimer

This chatbot is for educational purposes only and is not a substitute for
professional medical advice. Users should always consult a licensed healthcare
provider for diagnosis and treatment.

---
