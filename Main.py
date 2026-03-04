import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

import string
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher
import json

# Symptom synonym dictionary for better matching
SYMPTOM_SYNONYMS = {
    "vomit": ["throw up", "sick", "puking", "nausea"],
    "fever": ["high temperature", "temp", "hot"],
    "cough": ["coughing"],
    "pain": ["ache", "hurt", "hurting", "sore", "tender"],
    "headache": ["head pain", "migraine"],
    "sore throat": ["throat pain", "hurts to swallow"],
    "runny nose": ["nose running", "nasal discharge"],
    "diarrhea": ["loose stool", "loose bowel", "runs"],
    "vomiting": ["throw up", "sick", "puking"],
    "nausea": ["feeling sick", "queasy"],
    "fatigue": ["tired", "exhaustion", "exhausted", "sleepy"],
    "weakness": ["weak", "lack of strength"],
    "swelling": ["swollen", "puffed", "bloated"],
    "rash": ["skin irritation", "skin outbreak"],
    "itching": ["itchy", "itch"],
    "numbness": ["numb", "no feeling"],
    "tingling": ["prickling", "pins and needles"],
    "shortness of breath": ["breathless", "cant breathe", "breathing difficulty", "hard to breathe"],
    "chest pain": ["chest hurt", "chest discomfort", "pain in chest"],
    "dizziness": ["dizzy", "vertigo", "lightheaded"],
    "loss of taste": ["cant taste", "no taste"],
    "loss of smell": ["cant smell", "no smell"],
    "blurred vision": ["vision blur", "cant see clearly"],
    "sore": ["pain", "ache"],
    "inflammation": ["inflamed", "swelling"],
    "chills": ["shivering", "cold"],
    "sweating": ["perspiring", "sweaty"],
    "difficulty breathing": ["breathlessness", "struggling to breathe"],
    # Mental health synonyms
    "persistent sadness": ["constantly sad", "always sad", "sadness", "feeling down", "depressed"],
    "loss of interest": ["no interest", "disinterest", "not interested", "apathy"],
    "sleep changes": ["insomnia", "sleep problems", "cant sleep", "sleeping too much", "oversleeping"],
    "difficulty concentrating": ["cant focus", "cant concentrate", "lack of focus", "unable to focus"],
    "anhedonia": ["loss of interest", "no pleasure", "nothing feels good"],
    "hopelessness": ["no hope", "hopeless", "despair", "everything is hopeless"],
    "suicidal thoughts": ["suicide", "want to die", "suicidal ideation", "end it all"],
    "mood swings": ["extreme mood swings", "mood changes", "emotional changes"],
    "excessive worry": ["worry all the time", "cant stop worrying", "anxious"],
    "restlessness": ["cant sit still", "restless", "agitated"],
    "muscle tension": ["tense muscles", "tightness", "tension"],
    "hypervigilance": ["always alert", "constantly watchful", "on edge"],
    "intrusive memories": ["intrusive thoughts", "unwanted memories", "cant get thoughts out"],
    "nightmares": ["bad dreams", "frightening dreams"],
    "flashbacks": ["vivid flashbacks", "reliving trauma"],
    "avoidance behavior": ["avoiding", "avoiding situations"],
    "compulsive behaviors": ["compulsions", "cant stop behaviors"],
    "obsessive thoughts": ["intrusive thoughts", "obsessions", "stuck thoughts"],
    "fear of contamination": ["germaphobia", "fear of germs", "contamination fears"],
    "need for order": ["need for symmetry", "perfectionism", "order and control"],
    "blushing": ["red face", "flushed"],
    "trembling": ["shaking", "tremors"],
    "palpitations": ["heart racing", "racing heart", "heart pounding", "fast heartbeat"],
    "impulsive behavior": ["impulsivity", "acting without thinking", "reckless"],
    "hallucinations": ["seeing things", "hearing voices", "visual hallucinations"],
    "delusions": ["false beliefs", "paranoia"],
    "disorganized speech": ["incoherent", "speech problems", "rambling"],
    "withdrawn behavior": ["withdrawn", "isolation", "isolating"],
    "reduced emotional expression": ["flat affect", "emotionless", "no emotion"],
    "extreme fatigue": ["exhaustion", "utterly tired"],
    "cognitive difficulties": ["brain fog", "confusion", "memory problems"],
    "emotional instability": ["emotional instability", "emotional dysregulation"],
    "self-harm": ["cutting", "self injury", "hurting myself"],
    "lack of empathy": ["no empathy", "cant feel for others"],
    "manipulative behavior": ["manipulation", "deceitful"],
    "emotional exhaustion": ["burnt out", "overworked", "exhausted emotionally"],
    "cynicism": ["cynical", "disillusioned", "negative outlook"],
    "depersonalization": ["feeling detached", "not real", "out of body"],
    "derealization": ["world not real", "surroundings unreal"],
    "severe food restriction": ["not eating", "refusing food"],
    "extreme weight loss": ["losing weight rapidly", "very thin"],
    "preoccupation with weight": ["obsessed with weight", "weight obsession"],
    "distorted body image": ["see body wrong", "body dysmorphia", "think im fat"],
    "excessive exercise": ["overexercise", "compulsive exercise"],
    "binge eating episodes": ["binge eating", "overeating"],
    "purging behaviors": ["self induced vomiting", "laxatives", "purging"],
    "uncontrollable eating": ["cant control eating", "eat without stopping"],
    "shame": ["embarrassed", "ashamed"],
    "inattention": ["can't pay attention", "forgetful", "distracted"],
    "hyperactivity": ["overactive", "always moving"],
    "impulsivity": ["impulsive", "act without thinking"],
    "difficulty organizing": ["cant organize", "disorganized"],
    "forgetfulness": ["forget things", "memory issues"],
    "social interaction difficulties": ["social problems", "difficulty with people"],
    "repetitive behaviors": ["repetitive actions", "stimming", "repetitive movements"],
    "sensory sensitivities": ["sensitive to sound", "bright light bothers", "sensory issues"],
    "inappropriate anger": ["anger issues", "rage", "uncontrollable anger"],
    "post traumatic stress": ["trauma symptoms", "traumatized"],
    "panic attacks": ["sudden panic", "panic"],
    "racing thoughts": ["thoughts racing", "cant slow down thoughts"],
    "cravings": ["strong urges", "substance cravings"],
    "tolerance": ["need more substances", "increasing amounts"],
    "withdrawal symptoms": ["withdrawal", "detox symptoms"],
    "intense fear of abandonment": ["fear of rejection", "fear of abandonment"],
    "unstable relationships": ["relationship problems", "dramatic relationships"],
    "denial": ["denying issues", "denial"],
    "emotional distress": ["very upset", "extreme emotional pain"],
    "identity confusion": ["confused identity", "not knowing who i am"],
    "loss of purpose": ["no purpose", "lost meaning"],
    "year": ["continued grief"],
    "crying": ["crying a lot", "constant crying"],
    "guilt feelings": ["feeling guilty", "guilt"],
}


def fuzzy_match(input_text, symptom, threshold=0.75):
    """Check similarity between input and symptom using fuzzy matching"""
    ratio = SequenceMatcher(None, input_text, symptom).ratio()
    return ratio >= threshold

def normalize_symptom(detected_symptom, input_text):
    """Find the best matching canonical symptom using synonyms"""
    input_words = input_text.lower().split()
    
    # Check direct match first
    best_match = detected_symptom
    
    # Check if any synonym maps to this symptom
    for canonical, synonyms in SYMPTOM_SYNONYMS.items():
        if detected_symptom in [canonical] + synonyms:
            return canonical
    
    # Fuzzy match against all symptoms
    for canonical, synonyms in SYMPTOM_SYNONYMS.items():
        for variant in [canonical] + synonyms:
            for word in input_words:
                if fuzzy_match(word, variant, 0.7):
                    return canonical
    
    return detected_symptom

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return tokens

def extract_symptoms(user_input, symptom_list):
    tokens = preprocess(user_input)
    detected = []
    input_lower = user_input.lower()
    
    for symptom in symptom_list:
        symptom_lower = symptom.lower()
        symptom_tokens = symptom_lower.split()
        
        # Method 1: Exact multi-word match
        if all(word in tokens for word in symptom_tokens):
            detected.append(symptom)
            continue
        
        # Method 2: Fuzzy match for the whole symptom
        if fuzzy_match(input_lower, symptom_lower, 0.65):
            detected.append(symptom)
            continue
        
        # Method 3: Check synonyms
        for canonical, synonyms in SYMPTOM_SYNONYMS.items():
            if symptom_lower == canonical or symptom_lower in synonyms:
                for syn in [canonical] + synonyms:
                    if fuzzy_match(input_lower, syn, 0.65) or all(w in tokens for w in syn.split()):
                        detected.append(symptom)
                        break
                if symptom in detected:
                    break
    
    return list(set(detected))  # Remove duplicates


def match_disease(detected_symptoms, database):
    """
    Match diseases based on detected symptoms with weighted scoring.
    Prioritizes:
    1. Number of matched symptoms (most important)
    2. Percentage of disease symptoms matched
    3. Disease severity (higher severity diseases ranked higher if scores tie)
    """
    scores = {}
    severity_weight = {"low": 1, "medium": 2, "high": 3}
    
    for disease, info in database.items():
        matched_symptoms = set(detected_symptoms) & set(info["symptoms"])
        match_count = len(matched_symptoms)
        
        if match_count > 0:
            total_symptoms = len(info["symptoms"])
            match_percentage = match_count / total_symptoms
            severity_score = severity_weight.get(info.get("severity", "low"), 1)
            
            # Weighted score: matches (40%) + match percentage (30%) + severity (30%)
            final_score = (match_count * 0.4) + (match_percentage * 30 * 0.3) + (severity_score * 0.3)
            
            scores[disease] = {
                "score": final_score,
                "matches": match_count,
                "percentage": round(match_percentage * 100, 1)
            }
    
    if not scores:
        return []
    
    # Sort diseases by score
    ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    
    return ranked[:3]  # top 3 matches

def chatbot_response(user_input, database, symptom_list):
    detected = extract_symptoms(user_input, symptom_list)
    
    if not detected:
        return "I couldn't detect specific symptoms. Please describe more clearly. Try mentioning: fever, cough, pain, nausea, etc."
    
    ranked_diseases = match_disease(detected, database)
    
    if ranked_diseases:
        response = f"Detected symptoms: {', '.join(detected)}\n\n"
        response += "Possible conditions:\n\n"
        
        for disease, score_data in ranked_diseases:
            info = database[disease]
            matches = score_data["matches"]
            percentage = score_data["percentage"]
            response += f"1. {disease.upper()}\n"
            response += f"   - Matched {matches} symptoms ({percentage}% match)\n"
            response += f"   - Severity: {info['severity'].upper()}\n"
            response += f"   - Advice: {info['advice']}\n\n"
        
        response += "⚠️ DISCLAIMER: This is NOT a medical diagnosis. Please consult a doctor immediately."
        return response
    
    return "Symptoms detected but no clear match found. Please consult a doctor."

def load_database():
    with open("diseases.json", "r") as file:
        return json.load(file)

if __name__ == "__main__":

    # 1️⃣ Load JSON file
    with open("diseases.json", "r") as file:
        database = json.load(file)

    # 2️⃣ Create symptom list
    symptom_list = []
    for disease in database:
        symptom_list.extend(database[disease]["symptoms"])

    # Remove duplicates
    symptom_list = list(set(symptom_list))

    print("Medical Chatbot Started (type 'exit' to quit)\n")

    #  Chat loop
    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Take care! Stay healthy.")
            break

        response = chatbot_response(user_input, database, symptom_list)
        print("Chatbot:", response)
