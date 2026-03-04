import json
from Main import extract_symptoms, match_disease

# Load database
with open("diseases.json", "r") as file:
    database = json.load(file)

# Create symptom list
symptom_list = []
for disease in database:
    symptom_list.extend(database[disease]["symptoms"])

symptom_list = list(set(symptom_list))

# Test cases
test_inputs = [
    "I have a fever, cough, and I'm feeling exhausted",
    "My throat is burning and I'm throwing up",
    "I can't breathe and have chest pain",
    "My head hurts and I'm dizzy",
    "I have a runny nose and I'm sneezing"
]

print("=" * 70)
print("CHATBOT DIAGNOSIS TEST")
print("=" * 70)

for test in test_inputs:
    print(f"\nUser Input: {test}")
    print("-" * 70)
    
    detected = extract_symptoms(test, symptom_list)
    print(f"Detected symptoms: {detected}\n")
    
    ranked = match_disease(detected, database)
    if ranked:
        for i, (disease, score_data) in enumerate(ranked, 1):
            info = database[disease]
            print(f"{i}. {disease.upper()}")
            print(f"   Match: {score_data['matches']} symptoms ({score_data['percentage']}%)")
            print(f"   Severity: {info['severity']}")
            print(f"   Advice: {info['advice']}\n")
    else:
        print("No diseases matched.")
    print("=" * 70)
