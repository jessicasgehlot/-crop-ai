# 🌾 AI Crop Recommendation System

> An intelligent ML-powered web application that recommends the most suitable crops based on soil and weather conditions using **K-Means Clustering**.

---

## 📌 Project Overview

This system helps farmers make data-driven decisions by analyzing:
- **Weather**: Temperature, Humidity, Rainfall
- **Soil**: pH, Moisture, Nitrogen (N), Phosphorus (P), Potassium (K)

The AI clusters similar agricultural conditions and maps them to the best-suited crop from 9 categories: Wheat, Rice, Cotton, Sugarcane, Maize, Barley, Pulses, Vegetables, Fruits.

---

## 🏗️ System Architecture

```
User Input (Browser)
       ↓
Flask Backend (app.py)
       ↓
Data Preprocessing (StandardScaler)
       ↓
K-Means Clustering (scikit-learn)
       ↓
Cluster → Crop Mapping
       ↓
Recommendation + Advice (JSON)
       ↓
Dashboard (Chart.js + Bootstrap)
```

---

## 📁 Folder Structure

```
crop-recommendation/
├── app.py                  # Flask backend + API routes
├── requirements.txt        # Python dependencies
├── README.md
├── ml/
│   ├── __init__.py
│   ├── dataset.py          # Synthetic dataset generator
│   └── model.py            # K-Means model + recommendation engine
├── data/
│   ├── crop_data.csv       # Generated dataset (1800 samples)
│   └── kmeans_model.pkl    # Trained model (auto-generated)
├── templates/
│   └── index.html          # Main UI
├── static/
│   ├── css/style.css       # Custom styles
│   └── js/app.js           # Frontend logic + charts
└── docs/
    └── documentation.md    # IEEE-style documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone / navigate to project folder
cd "crop pre"

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py

# 5. Open browser
# http://localhost:5000
```

The dataset and ML model are **auto-generated** on first run — no manual setup needed.

---

## 🤖 Machine Learning Details

| Component | Details |
|-----------|---------|
| Algorithm | K-Means Clustering |
| Features | 8 (temp, humidity, rainfall, pH, moisture, N, P, K) |
| Clusters (K) | 9 (one per crop type) |
| Scaler | StandardScaler (zero mean, unit variance) |
| Training Samples | 1800 (200 per crop) |
| Optimal K Selection | Elbow Method (WCSS vs K plot) |

### How K-Means Works Here
1. Dataset of 1800 samples is scaled using StandardScaler
2. KMeans(n_clusters=9) groups similar conditions into 9 clusters
3. Each cluster is labeled with its dominant crop
4. New input is scaled → assigned to nearest cluster → crop recommended
5. Suitability score = distance-based proximity to cluster center

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main web interface |
| POST | `/api/recommend` | Get crop recommendation |
| GET | `/api/elbow` | Elbow method data |
| GET | `/api/clusters` | Cluster profiles |
| GET | `/api/dataset-stats` | Dataset statistics |
| POST | `/api/retrain` | Retrain model |

### Sample API Request
```json
POST /api/recommend
{
  "temp": 25, "humidity": 65, "rainfall": 100,
  "ph": 6.5, "moisture": 50, "N": 80, "P": 40, "K": 40
}
```

---

## 🎯 Features

- ✅ K-Means ML crop recommendation
- ✅ Suitability score with visual ring
- ✅ Fertilizer, irrigation & seasonal advice
- ✅ AI explanation (why this crop?)
- ✅ Alternative crop suggestions
- ✅ Elbow method chart
- ✅ Cluster visualization table
- ✅ Dataset distribution chart
- ✅ Radar chart (your soil vs ideal)
- ✅ Disease awareness section
- ✅ AI chatbot (AgriBot)
- ✅ Voice assistant
- ✅ Quick-fill presets
- ✅ Mobile responsive design

---

## 📊 Dataset

- **Size**: 1800 samples (200 per crop)
- **Features**: temp, humidity, rainfall, ph, moisture, N, P, K
- **Labels**: 9 crop types
- **Generation**: Gaussian distribution around realistic crop profiles
- **Source**: Synthetic (based on agricultural research data)

---

## 🔮 Future Scope

1. **Real weather API** integration (OpenWeatherMap)
2. **Satellite imagery** analysis for field health
3. **Crop disease detection** using CNN image classification
4. **Yield prediction** using regression models
5. **Market price integration** for profit optimization
6. **Government scheme** recommendation engine
7. **Multilingual support** (Hindi, regional languages)
8. **Mobile app** (React Native / Flutter)
9. **IoT sensor** integration for real-time soil data
10. **Federated learning** for privacy-preserving model updates

---

## 🎓 Viva Questions & Answers

**Q1: Why K-Means for crop recommendation?**
> K-Means groups similar soil/weather conditions into clusters. Each cluster naturally corresponds to a crop type, making it ideal for unsupervised pattern discovery in agricultural data.

**Q2: How is the optimal K (9) chosen?**
> Using the Elbow Method — we plot WCSS (Within-Cluster Sum of Squares) vs K. The "elbow" point where inertia stops decreasing sharply indicates the optimal K. For 9 crop types, K=9 is both mathematically optimal and domain-justified.

**Q3: What is StandardScaler and why use it?**
> StandardScaler normalizes features to zero mean and unit variance. Without it, features with large ranges (rainfall: 0–500) would dominate over small-range features (pH: 3–10), biasing the clustering.

**Q4: How is suitability score calculated?**
> It's distance-based: `score = (1 - dist/max_dist) × 100`. Closer to the cluster center = higher suitability. This gives an intuitive percentage of how well conditions match the recommended crop.

**Q5: What are the limitations of K-Means?**
> K-Means assumes spherical clusters, is sensitive to outliers, and requires K to be specified. For production, ensemble methods (Random Forest) could improve accuracy.

**Q6: How does the chatbot work?**
> Rule-based keyword matching against a farming knowledge base. Future versions could use NLP/LLM for more natural conversations.

**Q7: What is the social impact of this project?**
> Helps smallholder farmers make data-driven decisions, potentially increasing yield by 20–30%, reducing water waste, and optimizing fertilizer use — contributing to food security and sustainable agriculture.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, Bootstrap 5, Chart.js |
| Backend | Python 3, Flask |
| ML | Scikit-learn, Pandas, NumPy |
| Visualization | Chart.js (browser), Matplotlib (server) |
| Storage | CSV + Pickle |

---

## 📄 License

MIT License — Free for educational and research use.

---

*Built with ❤️ for farmers — AI-powered smart agriculture*
