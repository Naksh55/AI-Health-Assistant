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

# Mental health test cases
mental_health_tests = [
    "I'm constantly sad and have no interest in anything, feeling hopeless",
    "I have extreme mood swings, going from very happy to very depressed",
    "I keep having intrusive thoughts and obsessive behaviors, it's consuming me",
    "I have nightmares and flashbacks from a traumatic event",
    "I'm anxious all the time, can't focus, and my heart races",
    "I can't eat normally and have distorted body image"
]

print("=" * 80)
print("MENTAL HEALTH SYMPTOM DETECTION TEST")
print("=" * 80)

for test in mental_health_tests:
    print(f"\nUser Input: {test}")
    print("-" * 80)
    
    detected = extract_symptoms(test, symptom_list)
    print(f"Detected symptoms: {detected}\n")
    
    ranked = match_disease(detected, database)
    if ranked:
        for i, (disease, score_data) in enumerate(ranked, 1):
            info = database[disease]
            print(f"{i}. {disease.upper()}")
            print(f"   Matched: {score_data['matches']} symptoms ({score_data['percentage']}%)")
            print(f"   Severity: {info['severity']}")
            print(f"   Advice: {info['advice']}\n")
    else:
        print("No conditions matched.")
    print("=" * 80)

print(f"\nTotal mental health conditions in database: {sum(1 for d in database if any(word in d for word in ['disorder', 'illness', 'anxiety', 'depression', 'phobia', 'ptsd', 'ocd', 'bipolar', 'grief', 'burnout', 'insomnia', 'sleep', 'eating', 'autism', 'adhd', 'personality', 'dissociative', 'adjustment', 'stress', 'perinatal', 'premenstrual']))}")

total = len(database)
print(f"Total diseases in database: {total}")
