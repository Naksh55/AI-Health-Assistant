"""
ml/train_model.py — Train ML Disease Prediction Model
======================================================
Trains a RandomForestClassifier on symptom-disease data.

DATA SOURCES (in priority order):
  1. Kaggle CSV: ml/kaggle_data/dataset.csv (if downloaded)
  2. Synthetic: auto-generated from diseases.json

USAGE:
  python -m ml.train_model                    # uses best available data
  python -m ml.train_model --source kaggle    # force Kaggle CSV
  python -m ml.train_model --source synthetic # force synthetic

OUTPUT:
  ml/disease_model.pkl    — trained RandomForest model
  ml/feature_names.pkl    — ordered list of symptom feature names
  ml/label_encoder.pkl    — disease label encoder
"""

import os
import sys
import json
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = Path(__file__).resolve().parent
DISEASES_JSON = BASE_DIR / "diseases.json"
KAGGLE_CSV = ML_DIR / "kaggle_data" / "dataset.csv"
MODEL_PATH = ML_DIR / "disease_model.pkl"
FEATURES_PATH = ML_DIR / "feature_names.pkl"
ENCODER_PATH = ML_DIR / "label_encoder.pkl"


def load_kaggle_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Load data from Kaggle Disease-Symptom CSV.
    Expected format: Disease, Symptom_1, Symptom_2, ..., Symptom_17
    """
    if not KAGGLE_CSV.exists():
        raise FileNotFoundError(f"Kaggle CSV not found at {KAGGLE_CSV}")

    print(f"[ML] Loading Kaggle data from {KAGGLE_CSV}")
    df = pd.read_csv(KAGGLE_CSV)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Extract all unique symptoms
    symptom_cols = [c for c in df.columns if c.startswith("Symptom")]
    all_symptoms = set()
    for col in symptom_cols:
        df[col] = df[col].apply(lambda x: str(x).strip().lower().replace("_", " ") if pd.notna(x) else "")
        all_symptoms.update(df[col].unique())

    all_symptoms.discard("")
    all_symptoms.discard("nan")
    all_symptoms = sorted(list(all_symptoms))

    # Build binary feature matrix
    records = []
    for _, row in df.iterrows():
        disease = str(row.get("Disease", "")).strip()
        if not disease:
            continue
        patient_symptoms = set()
        for col in symptom_cols:
            s = str(row[col]).strip().lower().replace("_", " ")
            if s and s != "nan":
                patient_symptoms.add(s)

        feature_vec = {symptom: 1 if symptom in patient_symptoms else 0
                       for symptom in all_symptoms}
        feature_vec["disease"] = disease.lower().strip()
        records.append(feature_vec)

    result_df = pd.DataFrame(records)
    print(f"[ML] Kaggle data: {len(result_df)} samples, {len(all_symptoms)} features, "
          f"{result_df['disease'].nunique()} diseases")
    return result_df, all_symptoms


def generate_synthetic_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Generate synthetic training data from diseases.json.
    Creates ~3000 simulated patient records.
    """
    print(f"[ML] Generating synthetic data from {DISEASES_JSON}")

    with open(DISEASES_JSON, "r") as f:
        database = json.load(f)

    # Collect all unique symptoms
    all_symptoms = set()
    for disease_info in database.values():
        for s in disease_info["symptoms"]:
            all_symptoms.add(s.lower().strip())
    all_symptoms = sorted(list(all_symptoms))
    all_symptoms_set = set(all_symptoms)

    records = []
    random.seed(42)

    for disease_name, disease_info in database.items():
        disease_symptoms = [s.lower().strip() for s in disease_info["symptoms"]]
        num_disease_symptoms = len(disease_symptoms)

        # Generate 30-50 synthetic patients per disease
        num_samples = random.randint(30, 50)

        for _ in range(num_samples):
            # Pick a random subset of the disease's symptoms (at least 2)
            num_present = random.randint(
                max(2, num_disease_symptoms // 2),
                num_disease_symptoms
            )
            present_symptoms = set(random.sample(disease_symptoms, min(num_present, len(disease_symptoms))))

            # Occasionally add 1-2 noise symptoms from other diseases (15% chance)
            if random.random() < 0.15:
                noise_symptoms = list(all_symptoms_set - set(disease_symptoms))
                if noise_symptoms:
                    num_noise = random.randint(1, 2)
                    present_symptoms.update(random.sample(noise_symptoms, min(num_noise, len(noise_symptoms))))

            # Build feature vector
            feature_vec = {symptom: 1 if symptom in present_symptoms else 0
                           for symptom in all_symptoms}
            feature_vec["disease"] = disease_name.lower().strip()
            records.append(feature_vec)

    result_df = pd.DataFrame(records)
    # Shuffle
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[ML] Synthetic data: {len(result_df)} samples, {len(all_symptoms)} features, "
          f"{result_df['disease'].nunique()} diseases")
    return result_df, all_symptoms


def train_model(source: str = "auto"):
    """Train the RandomForest model."""

    # ── Load data ─────────────────────────────────────────────────────────
    df = None
    feature_names = None

    if source == "kaggle" or (source == "auto" and KAGGLE_CSV.exists()):
        try:
            df, feature_names = load_kaggle_data()
            print("[ML] Using Kaggle dataset")
        except Exception as e:
            print(f"[ML] Kaggle load failed: {e}")
            if source == "kaggle":
                raise
            print("[ML] Falling back to synthetic data...")

    if df is None:
        df, feature_names = generate_synthetic_data()
        print("[ML] Using synthetic dataset")

    # ── Prepare features and labels ───────────────────────────────────────
    X = df[feature_names].values
    y = df["disease"].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"\n[ML] Training data shape: X={X.shape}, y={y_encoded.shape}")
    print(f"[ML] Number of diseases: {len(label_encoder.classes_)}")

    # ── Split data ────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # ── Train model ───────────────────────────────────────────────────────
    print("\n[ML] Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[ML] Test Accuracy: {accuracy:.4f}")
    print("\n[ML] Classification Report (top classes):")

    # Show report for top 15 most frequent classes
    unique_test = np.unique(y_test)
    target_names = [label_encoder.classes_[i] for i in unique_test]
    report = classification_report(
        y_test, y_pred,
        labels=unique_test[:15],
        target_names=target_names[:15],
        zero_division=0
    )
    print(report)

    # ── Save model ────────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[ML] Model saved to {MODEL_PATH}")

    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_names, f)
    print(f"[ML] Feature names saved to {FEATURES_PATH}")

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"[ML] Label encoder saved to {ENCODER_PATH}")

    print(f"\n[ML] Training complete! Files saved in {ML_DIR}")
    return model, feature_names, label_encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML disease prediction model")
    parser.add_argument(
        "--source", choices=["auto", "kaggle", "synthetic"],
        default="auto",
        help="Data source: auto (try kaggle first), kaggle, or synthetic"
    )
    args = parser.parse_args()
    train_model(source=args.source)
