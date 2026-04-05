"""Test that the synonym dictionary fixes the matching issue."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ml.predictor import MLDiseasePredictor
p = MLDiseasePredictor()

print("=" * 60)
print("TEST 1: Normalized terms (previously FAILED)")
print("Input: ['dysuria', 'lumbago']")
print("-" * 60)
r1 = p.predict(["dysuria", "lumbago"])
for r in r1:
    print("  %s: %.1f%% (%s)" % (r["name"], r["probability"]*100, r["ml_confidence"]))
print("RESULT:", "PASS" if r1 else "FAIL - no predictions!")

print()
print("=" * 60)
print("TEST 2: Mixed raw + normalized (new approach)")
print("Input: ['dysuria', 'lumbago', 'burning urination', 'back pain']")
print("-" * 60)
r2 = p.predict(["dysuria", "lumbago", "burning urination", "back pain"])
for r in r2:
    print("  %s: %.1f%% (%s)" % (r["name"], r["probability"]*100, r["ml_confidence"]))
print("RESULT:", "PASS" if r2 else "FAIL")

print()
print("=" * 60)
print("TEST 3: Report symptoms - anemia + thyroid")
print("Input: ['fatigue', 'cold extremities', 'weight gain', 'lethargy', 'pyrexia']")
print("-" * 60)
r3 = p.predict(["fatigue", "cold extremities", "weight gain", "lethargy", "pyrexia"])
for r in r3:
    print("  %s: %.1f%% (%s)" % (r["name"], r["probability"]*100, r["ml_confidence"]))
print("RESULT:", "PASS" if r3 else "FAIL")
