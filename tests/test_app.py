"""
Test Suite — AI Crop Recommendation System
Run: pytest tests/ -v --cov=. --cov-report=term-missing
"""

import pytest
import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from ml.dataset import generate_dataset, CROP_PROFILES
from ml.model import CropRecommender, FEATURES


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def ensure_data():
    """Generate dataset and train model once for the whole test session."""
    if not os.path.exists("data/crop_data.csv"):
        generate_dataset()
    rec = CropRecommender()
    if not os.path.exists("data/kmeans_model.pkl"):
        rec.train()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── Dataset Tests ─────────────────────────────────────────────────────────────

class TestDataset:
    def test_dataset_file_exists(self):
        assert os.path.exists("data/crop_data.csv")

    def test_dataset_shape(self):
        import pandas as pd
        df = pd.read_csv("data/crop_data.csv")
        assert len(df) == 1800, "Expected 1800 samples (200 per crop)"
        assert list(df.columns) == FEATURES + ["crop"]

    def test_all_crops_present(self):
        import pandas as pd
        df = pd.read_csv("data/crop_data.csv")
        assert set(df["crop"].unique()) == set(CROP_PROFILES.keys())

    def test_samples_per_crop(self):
        import pandas as pd
        df = pd.read_csv("data/crop_data.csv")
        for crop in CROP_PROFILES:
            count = len(df[df["crop"] == crop])
            assert count == 200, f"{crop} should have 200 samples, got {count}"

    def test_no_negative_values(self):
        import pandas as pd
        df = pd.read_csv("data/crop_data.csv")
        for col in FEATURES:
            assert (df[col] >= 0).all(), f"Negative values found in {col}"


# ── ML Model Tests ────────────────────────────────────────────────────────────

class TestMLModel:
    def test_model_file_exists(self):
        assert os.path.exists("data/kmeans_model.pkl")

    def test_model_loads(self):
        rec = CropRecommender().load()
        assert rec.kmeans is not None
        assert rec.scaler is not None

    def test_prediction_returns_valid_crop(self):
        rec = CropRecommender().load()
        result = rec.predict(25, 65, 100, 6.5, 50, 80, 40, 40)
        assert result["recommended_crop"] in CROP_PROFILES.keys()

    def test_suitability_score_range(self):
        rec = CropRecommender().load()
        result = rec.predict(25, 65, 100, 6.5, 50, 80, 40, 40)
        assert 0 <= result["suitability_score"] <= 100

    def test_all_scores_present(self):
        rec = CropRecommender().load()
        result = rec.predict(25, 65, 100, 6.5, 50, 80, 40, 40)
        assert set(result["all_scores"].keys()) == set(CROP_PROFILES.keys())

    def test_result_has_required_keys(self):
        rec = CropRecommender().load()
        result = rec.predict(25, 65, 100, 6.5, 50, 80, 40, 40)
        required = ["recommended_crop", "cluster_id", "suitability_score",
                    "all_scores", "alternatives", "fertilizer",
                    "irrigation", "seasonal_tip", "yield_estimate",
                    "reasons", "varieties", "input"]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_reasons_have_correct_prefix(self):
        rec = CropRecommender().load()
        result = rec.predict(25, 65, 100, 6.5, 50, 80, 40, 40)
        for r in result["reasons"]:
            assert r.startswith(("OK ", "WARN ", "BAD ")), f"Bad reason prefix: {r[:10]}"

    def test_rice_prediction(self):
        """High humidity + high rainfall should predict rice."""
        rec = CropRecommender().load()
        result = rec.predict(28, 82, 200, 6.0, 75, 90, 45, 45)
        assert result["recommended_crop"] == "rice"

    def test_wheat_prediction(self):
        """Low temp + low rainfall should predict wheat."""
        rec = CropRecommender().load()
        result = rec.predict(22, 55, 60, 6.5, 40, 80, 40, 40)
        assert result["recommended_crop"] == "wheat"

    def test_elbow_data(self):
        rec = CropRecommender().load()
        data = rec.elbow_data()
        assert "k" in data and "inertia" in data
        assert len(data["k"]) == len(data["inertia"])
        assert data["k"] == list(range(2, 13))

    def test_cluster_summary(self):
        rec = CropRecommender().load()
        summary = rec.cluster_summary()
        assert len(summary) == 9
        for item in summary:
            assert "cluster" in item and "crop" in item and "profile" in item

    def test_variety_details_all(self):
        rec = CropRecommender().load()
        data = rec.variety_details()
        assert "vegetables" in data
        assert "pulses" in data
        assert len(data["vegetables"]) >= 4
        assert len(data["pulses"]) >= 4

    def test_variety_details_single(self):
        rec = CropRecommender().load()
        pulses = rec.variety_details("pulses")
        assert isinstance(pulses, list)
        assert len(pulses) >= 4
        for v in pulses:
            assert "name" in v and "yield" in v and "disease" in v


# ── API Endpoint Tests ────────────────────────────────────────────────────────

class TestAPIEndpoints:
    def test_index_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_health_endpoint(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["status"] == "healthy"
        assert d["model_loaded"] is True
        assert d["prediction_ok"] is True

    def test_recommend_valid_input(self, client):
        r = client.post("/api/recommend",
                        json={"temp":28,"humidity":82,"rainfall":200,
                              "ph":6.0,"moisture":75,"N":90,"P":45,"K":45})
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["status"] == "success"
        assert d["data"]["recommended_crop"] == "rice"

    def test_recommend_missing_field(self, client):
        r = client.post("/api/recommend", json={"temp": 25})
        assert r.status_code == 400

    def test_recommend_returns_varieties(self, client):
        r = client.post("/api/recommend",
                        json={"temp":25,"humidity":70,"rainfall":120,
                              "ph":6.4,"moisture":60,"N":110,"P":55,"K":55})
        d = json.loads(r.data)
        assert "varieties" in d["data"]

    def test_elbow_endpoint(self, client):
        r = client.get("/api/elbow")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["status"] == "success"
        assert len(d["data"]["k"]) == 11

    def test_clusters_endpoint(self, client):
        r = client.get("/api/clusters")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert len(d["data"]) == 9

    def test_dataset_stats_endpoint(self, client):
        r = client.get("/api/dataset-stats")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["data"]["total_samples"] == 1800

    def test_varieties_all(self, client):
        r = client.get("/api/varieties")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert "vegetables" in d["data"]
        assert "pulses" in d["data"]

    def test_varieties_filter(self, client):
        r = client.get("/api/varieties?crop=pulses")
        assert r.status_code == 200
        d = json.loads(r.data)
        names = [v["name"] for v in d["data"]]
        assert "Chickpea (Chana)" in names
