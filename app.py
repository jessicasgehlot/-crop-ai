"""
AI Crop Recommendation System — Flask Backend
Run: python app.py
"""

from flask import Flask, render_template, request, jsonify
from ml.model import get_recommender
from ml.dataset import generate_dataset
import os, json

app = Flask(__name__)

# ── Ensure dataset & model exist on startup ──────────────────────────────────
if not os.path.exists("data/crop_data.csv"):
    generate_dataset()

recommender = get_recommender()

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """Main prediction endpoint"""
    try:
        d = request.get_json()
        result = recommender.predict(
            temp=float(d["temp"]),
            humidity=float(d["humidity"]),
            rainfall=float(d["rainfall"]),
            ph=float(d["ph"]),
            moisture=float(d["moisture"]),
            N=float(d["N"]),
            P=float(d["P"]),
            K=float(d["K"]),
        )
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/elbow")
def elbow():
    """Elbow method data for chart"""
    try:
        data = recommender.elbow_data()
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/clusters")
def clusters():
    """Cluster summary for dashboard"""
    try:
        data = recommender.cluster_summary()
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/dataset-stats")
def dataset_stats():
    """Basic dataset statistics"""
    try:
        import pandas as pd
        df = pd.read_csv("data/crop_data.csv")
        stats = {
            "total_samples": len(df),
            "crops": df["crop"].value_counts().to_dict(),
            "feature_means": {col: round(df[col].mean(), 2)
                              for col in ["temp","humidity","rainfall","ph","moisture","N","P","K"]},
        }
        return jsonify({"status": "success", "data": stats})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/retrain", methods=["POST"])
def retrain():
    """Retrain model (admin use)"""
    try:
        generate_dataset()
        recommender.train()
        return jsonify({"status": "success", "message": "Model retrained successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/varieties")
def varieties():
    """Return variety details — all crops or ?crop=wheat"""
    try:
        crop = request.args.get("crop", None)
        data = recommender.variety_details(crop)
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/health")
def health():
    """Health check endpoint — used by monitoring script & Docker HEALTHCHECK"""
    try:
        import pandas as pd
        model_ok  = recommender.kmeans is not None
        data_ok   = os.path.exists("data/crop_data.csv")
        sample    = recommender.predict(25, 65, 100, 6.5, 50, 80, 40, 40)
        pred_ok   = "recommended_crop" in sample
        return jsonify({
            "status":      "healthy" if (model_ok and data_ok and pred_ok) else "degraded",
            "model_loaded": model_ok,
            "dataset_ok":   data_ok,
            "prediction_ok": pred_ok,
            "test_crop":    sample.get("recommended_crop"),
            "version":      "1.0.0",
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", debug=debug, port=port)
