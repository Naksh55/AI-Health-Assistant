"""
tests/test_ml_predictor.py — Tests for ML Disease Prediction
==============================================================
Tests the MLDiseasePredictor class and training pipeline.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


class TestSyntheticDataGeneration:
    """Test that synthetic training data is generated correctly."""

    def test_load_diseases_json(self):
        """diseases.json should exist and be valid."""
        diseases_path = Path(__file__).resolve().parent.parent / "diseases.json"
        assert diseases_path.exists(), "diseases.json not found"

        with open(diseases_path, "r") as f:
            data = json.load(f)

        assert len(data) > 50, "Should have 50+ diseases"

        # Each disease should have symptoms, advice, severity
        for disease, info in data.items():
            assert "symptoms" in info, f"{disease} missing symptoms"
            assert "advice" in info, f"{disease} missing advice"
            assert "severity" in info, f"{disease} missing severity"
            assert len(info["symptoms"]) >= 2, f"{disease} should have 2+ symptoms"

    def test_generate_synthetic_data(self):
        """Synthetic data generation should produce valid training samples."""
        from ml.train_model import generate_synthetic_data

        df, feature_names = generate_synthetic_data()

        assert len(df) > 1000, "Should generate 1000+ samples"
        assert len(feature_names) > 50, "Should have 50+ unique symptoms"
        assert "disease" in df.columns, "Should have disease column"

        # Check that all values are 0 or 1 (binary features)
        for col in feature_names:
            assert set(df[col].unique()).issubset({0, 1}), f"Column {col} should be binary"


class TestMLPredictor:
    """Test the MLDiseasePredictor class."""

    def test_predictor_initialization(self):
        """Predictor should initialize (may not have model yet)."""
        from ml.predictor import MLDiseasePredictor
        predictor = MLDiseasePredictor()
        # Should not crash even if model doesn't exist yet
        assert isinstance(predictor.is_ready, bool)

    def test_empty_symptoms(self):
        """Empty symptoms should return empty predictions."""
        from ml.predictor import MLDiseasePredictor
        predictor = MLDiseasePredictor()
        results = predictor.predict([])
        assert results == []

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "ml" / "disease_model.pkl").exists(),
        reason="ML model not trained yet"
    )
    def test_predict_with_model(self):
        """If model is trained, predictions should work."""
        from ml.predictor import MLDiseasePredictor
        predictor = MLDiseasePredictor()

        if not predictor.is_ready:
            pytest.skip("Model not available")

        results = predictor.predict(["fever", "headache", "chills"])

        assert len(results) > 0, "Should return predictions"
        assert len(results) <= 3, "Should return at most 3 predictions"

        for r in results:
            assert "name" in r
            assert "probability" in r
            assert "ml_confidence" in r
            assert 0 <= r["probability"] <= 1

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "ml" / "disease_model.pkl").exists(),
        reason="ML model not trained yet"
    )
    def test_fuzzy_matching(self):
        """Fuzzy matching should handle informal symptom names."""
        from ml.predictor import MLDiseasePredictor
        predictor = MLDiseasePredictor()

        if not predictor.is_ready:
            pytest.skip("Model not available")

        # These should match something in the feature names
        import numpy as np
        vector = predictor.get_feature_vector(["fever", "cough"])
        assert np.sum(vector) >= 1, "Should match at least one feature"


class TestTrainingPipeline:
    """Test the full training pipeline."""

    def test_training_completes(self, tmp_path):
        """Training should complete without errors."""
        from ml.train_model import generate_synthetic_data
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        # Generate data
        df, feature_names = generate_synthetic_data()

        # Prepare features
        X = df[feature_names].values
        y = df["disease"].values

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Train a small model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y_encoded)

        # Should be able to predict
        sample = np.zeros(len(feature_names))
        # Set "fever" to 1 if it exists in features
        if "fever" in feature_names:
            sample[feature_names.index("fever")] = 1

        pred = model.predict(sample.reshape(1, -1))
        assert pred is not None
        assert len(pred) == 1
