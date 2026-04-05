"""
ml/predictor.py — ML Disease Prediction Class
===============================================
Loads the trained RandomForest model and provides predictions.

USAGE:
  from ml.predictor import MLDiseasePredictor

  predictor = MLDiseasePredictor()
  results = predictor.predict(["fever", "headache", "chills"])
  # → [{"name": "malaria", "probability": 0.72}, ...]

KEY FEATURE:
  Includes a medical synonym dictionary that maps normalized
  medical terms (e.g., "dysuria", "pyrexia") back to Kaggle
  dataset feature names (e.g., "burning_micturition", "high_fever").
"""

import pickle
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher


ML_DIR = Path(__file__).resolve().parent
MODEL_PATH = ML_DIR / "disease_model.pkl"
FEATURES_PATH = ML_DIR / "feature_names.pkl"
ENCODER_PATH = ML_DIR / "label_encoder.pkl"


# ── Medical Synonym Dictionary ───────────────────────────────────────────────
# Maps medical/normalized terms → Kaggle feature names
# This bridges the gap between the LLM's medical vocabulary and the ML model's
# training data vocabulary.
MEDICAL_SYNONYMS = {
    # Fever-related
    "pyrexia": "high_fever",
    "fever": "high_fever",
    "febrile": "high_fever",
    "low grade fever": "mild_fever",
    "subfebril": "mild_fever",
    "hyperthermia": "high_fever",

    # Pain-related
    "cephalalgia": "headache",
    "headache": "headache",
    "migraine": "headache",
    "lumbago": "back_pain",
    "lower back pain": "back_pain",
    "dorsalgia": "back_pain",
    "arthralgia": "joint_pain",
    "joint pain": "joint_pain",
    "myalgia": "muscle_pain",
    "muscle pain": "muscle_pain",
    "body ache": "muscle_pain",
    "body aches": "muscle_pain",
    "abdominal pain": "abdominal_pain",
    "stomach pain": "stomach_pain",
    "epigastric pain": "stomach_pain",
    "retro-orbital pain": "pain_behind_the_eyes",
    "pain behind eyes": "pain_behind_the_eyes",
    "thoracic pain": "chest_pain",
    "chest pain": "chest_pain",
    "cervicalgia": "neck_pain",
    "neck pain": "neck_pain",
    "gonalgia": "knee_pain",
    "knee pain": "knee_pain",
    "coxalgia": "hip_joint_pain",
    "hip pain": "hip_joint_pain",
    "anal pain": "pain_in_anal_region",
    "rectal pain": "pain_in_anal_region",
    "painful defecation": "pain_during_bowel_movements",
    "painful urination": "burning_micturition",

    # Urinary
    "dysuria": "burning_micturition",
    "burning micturition": "burning_micturition",
    "burning sensation": "burning_micturition",
    "frequent urination": "continuous_feel_of_urine",
    "urinary frequency": "continuous_feel_of_urine",
    "polyuria": "polyuria",
    "hematuria": "spotting_ urination",
    "foul smelling urine": "foul_smell_of urine",
    "bladder discomfort": "bladder_discomfort",

    # GI symptoms
    "emesis": "vomiting",
    "vomiting": "vomiting",
    "nausea": "nausea",
    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",
    "loose stools": "diarrhoea",
    "constipation": "constipation",
    "dyspepsia": "indigestion",
    "indigestion": "indigestion",
    "heartburn": "acidity",
    "acid reflux": "acidity",
    "acidity": "acidity",
    "flatulence": "passage_of_gases",
    "bloating": "passage_of_gases",
    "hematemesis": "stomach_bleeding",
    "rectal bleeding": "bloody_stool",
    "bloody stool": "bloody_stool",
    "melena": "bloody_stool",

    # Respiratory
    "dyspnea": "breathlessness",
    "breathlessness": "breathlessness",
    "shortness of breath": "breathlessness",
    "cough": "cough",
    "tussis": "cough",
    "productive cough": "phlegm",
    "sputum": "phlegm",
    "phlegm": "phlegm",
    "hemoptysis": "blood_in_sputum",
    "coughing blood": "blood_in_sputum",
    "wheezing": "breathlessness",
    "sneezing": "continuous_sneezing",
    "rhinorrhea": "runny_nose",
    "runny nose": "runny_nose",
    "nasal congestion": "congestion",
    "congestion": "congestion",

    # Skin
    "pruritus": "itching",
    "itching": "itching",
    "exanthem": "skin_rash",
    "rash": "skin_rash",
    "skin rash": "skin_rash",
    "jaundice": "yellowish_skin",
    "icterus": "yellowish_skin",
    "yellowish skin": "yellowish_skin",
    "yellowing of eyes": "yellowing_of_eyes",
    "scleral icterus": "yellowing_of_eyes",
    "desquamation": "skin_peeling",
    "skin peeling": "skin_peeling",
    "acne": "pus_filled_pimples",
    "pimples": "pus_filled_pimples",
    "nodules": "nodal_skin_eruptions",

    # Neurological
    "vertigo": "dizziness",
    "dizziness": "dizziness",
    "lightheadedness": "dizziness",
    "syncope": "loss_of_balance",
    "loss of balance": "loss_of_balance",
    "ataxia": "unsteadiness",
    "unsteadiness": "unsteadiness",
    "paresthesia": "tingling",
    "numbness": "numbness",
    "altered sensorium": "altered_sensorium",
    "confusion": "altered_sensorium",
    "coma": "coma",
    "unconscious": "coma",
    "dysarthria": "slurred_speech",
    "slurred speech": "slurred_speech",
    "visual disturbance": "visual_disturbances",
    "blurred vision": "blurred_and_distorted_vision",

    # General / Constitutional
    "fatigue": "fatigue",
    "malaise": "malaise",
    "asthenia": "fatigue",
    "exhaustion": "fatigue",
    "tiredness": "fatigue",
    "lethargy": "lethargy",
    "anorexia": "loss_of_appetite",
    "loss of appetite": "loss_of_appetite",
    "decreased appetite": "loss_of_appetite",
    "weight loss": "weight_loss",
    "unintentional weight loss": "weight_loss",
    "weight gain": "weight_gain",
    "obesity": "obesity",
    "excessive hunger": "excessive_hunger",
    "polyphagia": "excessive_hunger",
    "increased appetite": "increased_appetite",
    "dehydration": "dehydration",
    "diaphoresis": "sweating",
    "sweating": "sweating",
    "night sweats": "sweating",
    "rigors": "chills",
    "chills": "chills",
    "shivering": "shivering",
    "restlessness": "restlessness",
    "anxiety": "anxiety",
    "insomnia": "restlessness",

    # Musculoskeletal
    "muscle weakness": "muscle_weakness",
    "weakness": "weakness_in_limbs",
    "limb weakness": "weakness_in_limbs",
    "hemiparesis": "weakness_of_one_body_side",
    "one-sided weakness": "weakness_of_one_body_side",
    "stiffness": "movement_stiffness",
    "joint stiffness": "movement_stiffness",
    "stiff neck": "stiff_neck",
    "neck stiffness": "stiff_neck",
    "swollen joints": "swelling_joints",
    "joint swelling": "swelling_joints",
    "difficulty walking": "painful_walking",
    "limping": "painful_walking",
    "muscle cramps": "cramps",
    "cramps": "cramps",
    "muscle wasting": "muscle_wasting",

    # Cardiovascular
    "tachycardia": "fast_heart_rate",
    "rapid heartbeat": "fast_heart_rate",
    "palpitations": "palpitations",
    "edema": "swollen_legs",
    "swelling": "swelling_of_stomach",
    "varicose veins": "swollen_blood_vessels",
    "prominent veins": "prominent_veins_on_calf",
    "bruising": "bruising",

    # Eyes / ENT
    "photophobia": "visual_disturbances",
    "red eyes": "redness_of_eyes",
    "conjunctival redness": "redness_of_eyes",
    "lacrimation": "watering_from_eyes",
    "watery eyes": "watering_from_eyes",
    "anosmia": "loss_of_smell",
    "loss of smell": "loss_of_smell",
    "throat irritation": "throat_irritation",
    "sore throat": "throat_irritation",
    "pharyngitis": "throat_irritation",
    "sinus pressure": "sinus_pressure",
    "sinusitis": "sinus_pressure",
    "sunken eyes": "sunken_eyes",

    # Metabolic / Endocrine
    "irregular sugar": "irregular_sugar_level",
    "hyperglycemia": "irregular_sugar_level",
    "hypoglycemia": "irregular_sugar_level",
    "cold extremities": "cold_hands_and_feets",
    "cold hands": "cold_hands_and_feets",
    "cold feet": "cold_hands_and_feets",
    "mood swings": "mood_swings",
    "depression": "depression",
    "irritability": "irritability",
    "puffy face": "puffy_face_and_eyes",
    "periorbital edema": "puffy_face_and_eyes",
    "goiter": "enlarged_thyroid",
    "enlarged thyroid": "enlarged_thyroid",
    "thyroid enlargement": "enlarged_thyroid",
    "brittle nails": "brittle_nails",
    "menstrual irregularity": "abnormal_menstruation",
    "irregular periods": "abnormal_menstruation",
    "amenorrhea": "abnormal_menstruation",

    # Hepatic
    "dark urine": "dark_urine",
    "dark colored urine": "dark_urine",
    "hepatomegaly": "swelling_of_stomach",
    "ascites": "distention_of_abdomen",
    "abdominal distension": "distention_of_abdomen",
    "fluid overload": "fluid_overload",
    "alcohol use": "history_of_alcohol_consumption",
    "alcoholism": "history_of_alcohol_consumption",
    "liver failure": "acute_liver_failure",

    # Lymphatic / Immune
    "lymphadenopathy": "swelled_lymph_nodes",
    "swollen lymph nodes": "swelled_lymph_nodes",
    "patches in throat": "patches_in_throat",

    # Report-derived terms (lab values → symptoms)
    "anemia": "fatigue",
    "iron deficiency": "fatigue",
    "low hemoglobin": "fatigue",
    "hypothyroidism": "weight_gain",
    "elevated tsh": "enlarged_thyroid",
    "high cholesterol": "obesity",
    "dyslipidemia": "obesity",
    "proteinuria": "dark_urine",
    "glycosuria": "irregular_sugar_level",
    "bacteriuria": "burning_micturition",
    "high wbc": "high_fever",
    "leukocytosis": "high_fever",
}


class MLDiseasePredictor:
    """
    Wrapper around the trained RandomForest model.
    Converts symptom lists to binary feature vectors and predicts diseases.

    Uses a medical synonym dictionary to bridge normalized medical terms
    (from the LLM normalizer) to Kaggle dataset feature names.
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.label_encoder = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        """Load saved model, features, and encoder."""
        try:
            if not MODEL_PATH.exists():
                print("[MLPredictor] Model not found. Run: python -m ml.train_model")
                return

            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            with open(FEATURES_PATH, "rb") as f:
                self.feature_names = pickle.load(f)
            with open(ENCODER_PATH, "rb") as f:
                self.label_encoder = pickle.load(f)

            self._loaded = True
            print(f"[MLPredictor] Model loaded: {len(self.feature_names)} features, "
                  f"{len(self.label_encoder.classes_)} diseases")

        except Exception as e:
            print(f"[MLPredictor] Error loading model: {e}")
            self._loaded = False

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def _resolve_symptom(self, symptom: str) -> str | None:
        """
        Resolve a symptom string to a Kaggle feature name.

        Priority:
          1. Exact match in feature names
          2. Medical synonym dictionary
          3. Underscore-join match (e.g., "chest pain" → "chest_pain")
          4. Substring match
          5. Fuzzy match (SequenceMatcher ≥ 0.70)
        """
        s = symptom.lower().strip().replace("_", " ")

        # 1. Exact match
        if s in self.feature_names:
            return s
        s_under = s.replace(" ", "_")
        if s_under in self.feature_names:
            return s_under

        # 2. Medical synonym dictionary
        if s in MEDICAL_SYNONYMS:
            target = MEDICAL_SYNONYMS[s]
            if target in self.feature_names:
                return target
        if s_under in MEDICAL_SYNONYMS:
            target = MEDICAL_SYNONYMS[s_under]
            if target in self.feature_names:
                return target

        # 3. Try common word variations
        for syn_key, syn_val in MEDICAL_SYNONYMS.items():
            if s in syn_key or syn_key in s:
                if syn_val in self.feature_names:
                    return syn_val

        # 4. Substring match against feature names
        for feature in self.feature_names:
            feat_clean = feature.replace("_", " ")
            if s in feat_clean or feat_clean in s:
                return feature

        # 5. Fuzzy match
        best_match = None
        best_score = 0.0

        for feature in self.feature_names:
            feat_clean = feature.replace("_", " ")
            score = SequenceMatcher(None, s, feat_clean).ratio()
            if score > best_score and score >= 0.70:
                best_score = score
                best_match = feature

        return best_match

    def get_feature_vector(self, symptoms: list[str]) -> np.ndarray:
        """
        Convert a list of symptom strings to a binary feature vector.
        Uses synonym dictionary + fuzzy matching to handle both
        raw symptoms and normalized medical terms.
        """
        vector = np.zeros(len(self.feature_names), dtype=int)

        matched = []
        unmatched = []
        for symptom in symptoms:
            match = self._resolve_symptom(symptom)
            if match:
                idx = self.feature_names.index(match)
                vector[idx] = 1
                matched.append(f"{symptom} -> {match}")
            else:
                unmatched.append(symptom)

        print(f"[MLPredictor] Matched: {matched}")
        if unmatched:
            print(f"[MLPredictor] Unmatched: {unmatched}")
        return vector

    def predict(self, symptoms: list[str], top_k: int = 3) -> list[dict]:
        """
        Predict diseases from a list of symptoms.

        Args:
            symptoms: list of symptom strings (can be informal OR medical terms)
            top_k: number of top predictions to return

        Returns:
            List of dicts: [{"name": "malaria", "probability": 0.72, "ml_confidence": "High"}, ...]
        """
        if not self._loaded:
            print("[MLPredictor] Model not loaded — skipping ML prediction")
            return []

        if not symptoms:
            return []

        # Build feature vector
        feature_vector = self.get_feature_vector(symptoms)

        # Check if any features matched
        if feature_vector.sum() == 0:
            print("[MLPredictor] No symptoms matched feature names — skipping")
            return []

        # Get prediction probabilities
        probabilities = self.model.predict_proba(feature_vector.reshape(1, -1))[0]

        # Get top-k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            prob = probabilities[idx]
            if prob < 0.01:  # Skip very low probability predictions
                continue

            disease_name = self.label_encoder.classes_[idx]

            # Determine confidence level
            if prob >= 0.5:
                confidence = "High"
            elif prob >= 0.2:
                confidence = "Medium"
            else:
                confidence = "Low"

            results.append({
                "name": disease_name.replace("_", " ").title(),
                "probability": round(float(prob), 3),
                "ml_confidence": confidence
            })

        print(f"[MLPredictor] Predictions: {results}")
        return results
