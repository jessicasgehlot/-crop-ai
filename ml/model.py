"""
AI Crop Recommendation Engine
Uses K-Means Clustering to group soil/weather conditions
and recommend the most suitable crops
"""

import numpy as np
import pandas as pd
import pickle, os, warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
warnings.filterwarnings("ignore")

FEATURES = ["temp", "humidity", "rainfall", "ph", "moisture", "N", "P", "K"]
MODEL_PATH = "data/kmeans_model.pkl"
DATA_PATH  = "data/crop_data.csv"
OPTIMAL_K  = 9  # one cluster per crop type

# ── Fertilizer advice per crop ──────────────────────────────────────────────
FERTILIZER_ADVICE = {
    "wheat":      "Apply Urea (46-0-0) at sowing; top-dress with DAP at tillering stage.",
    "rice":       "Use NPK 20-20-0 as basal dose; apply Potash before panicle initiation.",
    "cotton":     "Apply DAP + MOP at planting; foliar spray of Boron during flowering.",
    "sugarcane":  "Use FYM 25 t/ha + NPK 250:85:120 kg/ha split over 3 doses.",
    "maize":      "Apply Urea in 3 splits; use Zinc Sulphate if soil Zn is deficient.",
    "barley":     "Moderate N (60 kg/ha); avoid excess N to prevent lodging.",
    "pulses":     "Minimal N needed (Rhizobium inoculation); apply P & K as basal.",
    "vegetables": "High organic matter + balanced NPK; foliar micronutrients weekly.",
    "fruits":     "Apply FYM + NPK 100:50:100 kg/ha; foliar Ca & Mg spray.",
}

IRRIGATION_ADVICE = {
    "wheat":      "Irrigate at crown root initiation, tillering, jointing & grain filling (5–6 irrigations).",
    "rice":       "Maintain 2–5 cm standing water; drain 10 days before harvest.",
    "cotton":     "Drip irrigation preferred; critical stages: squaring & boll development.",
    "sugarcane":  "Furrow/drip irrigation every 7–10 days; avoid waterlogging.",
    "maize":      "Irrigate at knee-high, tasseling & grain filling stages.",
    "barley":     "2–3 irrigations sufficient; first at crown root initiation.",
    "pulses":     "Minimal irrigation; 1–2 life-saving irrigations at flowering & pod fill.",
    "vegetables": "Drip/sprinkler every 3–5 days; mulching reduces water loss.",
    "fruits":     "Drip irrigation; increase frequency during fruit development.",
}

SEASONAL_TIPS = {
    "wheat":      "Rabi crop — sow Oct–Nov; harvest Mar–Apr. Avoid late sowing.",
    "rice":       "Kharif crop — transplant Jun–Jul; harvest Oct–Nov.",
    "cotton":     "Kharif crop — sow Apr–May; pick Oct–Dec. Monitor bollworm.",
    "sugarcane":  "Plant Feb–Mar (spring) or Oct–Nov (autumn); 12-month crop.",
    "maize":      "Kharif (Jun–Jul) or Rabi (Oct–Nov); short duration 90–110 days.",
    "barley":     "Rabi crop — sow Oct–Nov; early maturing, harvest Feb–Mar.",
    "pulses":     "Kharif or Rabi depending on variety; fix atmospheric nitrogen.",
    "vegetables": "Year-round with season-specific varieties; use poly-tunnels in winter.",
    "fruits":     "Perennial; plant at start of monsoon; mulch in summer.",
}

ALTERNATIVES = {
    "wheat":["barley","maize"],   "rice":["sugarcane","vegetables"],
    "cotton":["maize","pulses"],  "sugarcane":["rice","fruits"],
    "maize":["wheat","pulses"],   "barley":["wheat","pulses"],
    "pulses":["maize","vegetables"],"vegetables":["fruits","pulses"],
    "fruits":["vegetables","sugarcane"],
}

YIELD_ESTIMATE = {
    "wheat":"3–5 t/ha","rice":"4–6 t/ha","cotton":"2–3 t/ha (lint)",
    "sugarcane":"60–80 t/ha","maize":"4–7 t/ha","barley":"2–4 t/ha",
    "pulses":"1–2 t/ha","vegetables":"15–30 t/ha","fruits":"10–25 t/ha",
}

# ── Specific variety details per crop group ──────────────────────────────────
VARIETY_DETAILS = {
    "vegetables": [
        {
            "name": "Tomato",
            "best_season": "Rabi (Oct–Jan) / Summer (Feb–May)",
            "soil": "Well-drained loamy, pH 6.0–7.0",
            "temp_range": "18–27°C",
            "water_need": "Moderate — drip irrigation, 400–600 mm/season",
            "fertilizer": "NPK 120:60:60 kg/ha; foliar spray of Ca & Mg",
            "duration": "60–90 days",
            "yield": "25–35 t/ha",
            "disease": "Early blight, Late blight — use Mancozeb + Metalaxyl",
            "companion": "Basil, Carrot, Marigold",
            "market": "High demand year-round; peak price in summer",
        },
        {
            "name": "Onion",
            "best_season": "Rabi (Oct–Nov sowing)",
            "soil": "Sandy loam, pH 6.0–7.5",
            "temp_range": "13–24°C",
            "water_need": "Low-moderate — 350–500 mm/season; stop 10 days before harvest",
            "fertilizer": "NPK 100:50:50 kg/ha; Sulphur 20 kg/ha",
            "duration": "120–150 days",
            "yield": "20–30 t/ha",
            "disease": "Purple blotch, Thrips — use Iprodione + Imidacloprid",
            "companion": "Carrot, Beetroot, Chamomile",
            "market": "Export-quality; high price in off-season (Apr–Jun)",
        },
        {
            "name": "Potato",
            "best_season": "Rabi (Oct–Dec)",
            "soil": "Loose sandy loam, pH 5.5–6.5",
            "temp_range": "15–20°C",
            "water_need": "Moderate — 500–700 mm; critical at tuber initiation",
            "fertilizer": "NPK 150:80:100 kg/ha; earthing-up at 30 days",
            "duration": "75–120 days",
            "yield": "20–40 t/ha",
            "disease": "Late blight (Phytophthora) — use Ridomil Gold; Aphids — Imidacloprid",
            "companion": "Beans, Corn, Horseradish",
            "market": "Stable demand; cold storage extends market window",
        },
        {
            "name": "Brinjal (Eggplant)",
            "best_season": "Kharif (Jun–Jul) & Rabi (Oct–Nov)",
            "soil": "Well-drained loam, pH 5.5–6.8",
            "temp_range": "22–32°C",
            "water_need": "Moderate — drip/furrow, 500–600 mm/season",
            "fertilizer": "NPK 100:50:50 kg/ha; Boron foliar spray at flowering",
            "duration": "90–120 days",
            "yield": "20–30 t/ha",
            "disease": "Shoot & fruit borer — use Spinosad; Wilt — Trichoderma",
            "companion": "Beans, Tarragon, Marigold",
            "market": "Year-round demand; good local & export market",
        },
        {
            "name": "Cauliflower",
            "best_season": "Rabi (Sep–Nov)",
            "soil": "Rich loam, pH 6.0–7.0",
            "temp_range": "15–20°C",
            "water_need": "Moderate — 400–500 mm; consistent moisture needed",
            "fertilizer": "NPK 120:60:60 kg/ha; Boron 1 kg/ha to prevent hollow stem",
            "duration": "60–90 days",
            "yield": "15–25 t/ha",
            "disease": "Black rot, Downy mildew — use Copper oxychloride",
            "companion": "Celery, Dill, Mint",
            "market": "High winter demand; good price Nov–Feb",
        },
        {
            "name": "Spinach",
            "best_season": "Rabi (Oct–Feb)",
            "soil": "Loamy, pH 6.5–7.5",
            "temp_range": "10–20°C",
            "water_need": "Low — 250–350 mm; sprinkler irrigation ideal",
            "fertilizer": "N 80 kg/ha in 2 splits; avoid excess P",
            "duration": "30–45 days (multiple cuts)",
            "yield": "10–15 t/ha",
            "disease": "Downy mildew — use Metalaxyl; Leaf miner — Neem oil",
            "companion": "Strawberry, Peas, Radish",
            "market": "Fast-growing; 4–5 cuts per season; urban market demand",
        },
    ],
    "pulses": [
        {
            "name": "Chickpea (Chana)",
            "best_season": "Rabi (Oct–Nov)",
            "soil": "Sandy loam to clay loam, pH 6.0–8.0",
            "temp_range": "15–25°C",
            "water_need": "Low — 300–400 mm; 1–2 irrigations only",
            "fertilizer": "N 20 kg/ha + P 40 kg/ha; Rhizobium seed treatment",
            "duration": "90–120 days",
            "yield": "1.5–2.5 t/ha",
            "disease": "Wilt (Fusarium) — use resistant varieties; Pod borer — Bt spray",
            "companion": "Wheat, Mustard, Linseed",
            "market": "Highest traded pulse; MSP support; export to Middle East",
        },
        {
            "name": "Lentil (Masoor)",
            "best_season": "Rabi (Oct–Nov)",
            "soil": "Loamy, pH 6.0–8.0",
            "temp_range": "15–25°C",
            "water_need": "Very low — 250–350 mm; rain-fed mostly",
            "fertilizer": "N 20 kg/ha + P 40 kg/ha; Rhizobium inoculation",
            "duration": "100–120 days",
            "yield": "0.8–1.5 t/ha",
            "disease": "Rust, Stemphylium blight — use Mancozeb",
            "companion": "Wheat, Barley",
            "market": "Stable demand; dal processing industry; good export value",
        },
        {
            "name": "Green Gram (Moong)",
            "best_season": "Kharif (Jun–Jul) & Zaid (Mar–Apr)",
            "soil": "Sandy loam, pH 6.2–7.2",
            "temp_range": "25–35°C",
            "water_need": "Low — 300–400 mm; 2–3 irrigations",
            "fertilizer": "N 20 kg/ha + P 40 kg/ha + K 20 kg/ha",
            "duration": "60–75 days",
            "yield": "0.8–1.2 t/ha",
            "disease": "Yellow mosaic virus — use whitefly-resistant varieties",
            "companion": "Maize, Sorghum, Cotton",
            "market": "Short duration; fits well in crop rotation; good local demand",
        },
        {
            "name": "Black Gram (Urad)",
            "best_season": "Kharif (Jun–Jul)",
            "soil": "Loamy to clay loam, pH 6.0–7.5",
            "temp_range": "25–35°C",
            "water_need": "Low — 300–450 mm; avoid waterlogging",
            "fertilizer": "N 20 kg/ha + P 40 kg/ha; Rhizobium seed treatment",
            "duration": "70–90 days",
            "yield": "0.6–1.0 t/ha",
            "disease": "Leaf crinkle virus, Cercospora leaf spot — use Carbendazim",
            "companion": "Sorghum, Maize, Groundnut",
            "market": "High demand for dal; used in idli/dosa industry",
        },
        {
            "name": "Pigeon Pea (Arhar/Tur)",
            "best_season": "Kharif (Jun–Jul)",
            "soil": "Well-drained loam, pH 6.0–7.5",
            "temp_range": "26–30°C",
            "water_need": "Low — 600–650 mm; drought-tolerant",
            "fertilizer": "N 20 kg/ha + P 50 kg/ha + K 20 kg/ha",
            "duration": "150–180 days (long duration)",
            "yield": "1.0–2.0 t/ha",
            "disease": "Wilt, Sterility mosaic — use resistant varieties + Carbendazim",
            "companion": "Sorghum, Groundnut, Cotton",
            "market": "Most consumed pulse in India; MSP guaranteed; high protein content",
        },
        {
            "name": "Peas (Matar)",
            "best_season": "Rabi (Oct–Nov)",
            "soil": "Well-drained loam, pH 6.0–7.5",
            "temp_range": "10–18°C",
            "water_need": "Moderate — 350–500 mm; critical at flowering",
            "fertilizer": "N 20 kg/ha + P 60 kg/ha + K 40 kg/ha",
            "duration": "60–90 days",
            "yield": "8–10 t/ha (green pods)",
            "disease": "Powdery mildew — use Sulphur dust; Root rot — Trichoderma",
            "companion": "Carrot, Radish, Spinach",
            "market": "Fresh & frozen market; processing industry; good winter price",
        },
    ],
    "fruits": [
        {
            "name": "Mango",
            "best_season": "Perennial; flowering Jan–Feb; harvest May–Jul",
            "soil": "Deep well-drained loam, pH 5.5–7.5",
            "temp_range": "24–30°C",
            "water_need": "Low after establishment — drip 600–800 mm/year",
            "fertilizer": "NPK 1000:500:1000 g/tree/year; FYM 50 kg/tree",
            "duration": "3–5 years to first fruit; 40+ year productive life",
            "yield": "10–20 t/ha",
            "disease": "Anthracnose, Powdery mildew — use Carbendazim + Sulphur",
            "companion": "Banana (intercrop), Papaya (young orchards)",
            "market": "King of fruits; export to Europe & Middle East; Alphonso premium variety",
        },
        {
            "name": "Banana",
            "best_season": "Plant Jun–Jul or Feb–Mar",
            "soil": "Rich loamy, pH 6.0–7.5",
            "temp_range": "26–30°C",
            "water_need": "High — drip 1200–2200 mm/year; moisture-sensitive",
            "fertilizer": "NPK 200:60:300 g/plant; FYM 10 kg/plant",
            "duration": "11–15 months to harvest",
            "yield": "40–60 t/ha",
            "disease": "Panama wilt (Fusarium) — use resistant varieties; Sigatoka — Propiconazole",
            "companion": "Legumes as cover crop",
            "market": "Year-round demand; Cavendish & Robusta most traded varieties",
        },
        {
            "name": "Papaya",
            "best_season": "Plant Jun–Jul or Feb–Mar",
            "soil": "Sandy loam, pH 6.0–7.0; avoid waterlogging",
            "temp_range": "22–30°C",
            "water_need": "Moderate — drip 800–1000 mm/year",
            "fertilizer": "NPK 250:250:500 g/plant; FYM 20 kg/plant",
            "duration": "9–12 months to first harvest",
            "yield": "40–50 t/ha",
            "disease": "Papaya ringspot virus — use aphid control; Phytophthora — Metalaxyl",
            "companion": "Maize (windbreak), Legumes",
            "market": "Papain extraction industry; fresh market; Red Lady variety preferred",
        },
        {
            "name": "Guava",
            "best_season": "Plant Jun–Jul; fruits twice a year",
            "soil": "Wide range, pH 5.0–7.5; drought-tolerant",
            "temp_range": "23–28°C",
            "water_need": "Low — 1000 mm/year; tolerates dry spells",
            "fertilizer": "NPK 600:300:600 g/tree/year",
            "duration": "2–3 years to first fruit",
            "yield": "15–25 t/ha",
            "disease": "Wilt (Fusarium), Fruit fly — use Malathion bait spray",
            "companion": "Legumes as intercrop",
            "market": "Vitamin C rich; juice & processing industry; Allahabad Safeda premium",
        },
    ],
    "wheat": [
        {
            "name": "HD-2967 (Wheat)",
            "best_season": "Rabi — sow Nov 1–15",
            "soil": "Loamy to clay loam, pH 6.5–7.5",
            "temp_range": "20–25°C at sowing; 12–15°C at grain fill",
            "water_need": "5–6 irrigations; critical at CRI, tillering, jointing",
            "fertilizer": "NPK 120:60:40 kg/ha",
            "duration": "120–130 days",
            "yield": "4.5–5.5 t/ha",
            "disease": "Yellow rust — use Propiconazole; Loose smut — seed treatment",
            "companion": "Mustard, Lentil",
            "market": "MSP guaranteed; FCI procurement; export quality",
        },
        {
            "name": "GW-322 (Wheat)",
            "best_season": "Rabi — sow Oct 25–Nov 10",
            "soil": "Sandy loam to loam, pH 6.0–7.5",
            "temp_range": "18–24°C",
            "water_need": "4–5 irrigations; drought-tolerant variety",
            "fertilizer": "NPK 100:50:40 kg/ha",
            "duration": "110–120 days",
            "yield": "3.5–4.5 t/ha",
            "disease": "Stem rust resistant; Karnal bunt — seed treatment with Vitavax",
            "companion": "Chickpea, Mustard",
            "market": "Good for arid regions; Gujarat & Rajasthan preferred variety",
        },
    ],
    "rice": [
        {
            "name": "Basmati 1121",
            "best_season": "Kharif — transplant Jun 15–Jul 15",
            "soil": "Clay loam, pH 5.5–6.5",
            "temp_range": "25–35°C",
            "water_need": "High — 1200–1500 mm; 2–5 cm standing water",
            "fertilizer": "NPK 120:60:60 kg/ha; Zinc 25 kg/ha",
            "duration": "140–145 days",
            "yield": "4–5 t/ha",
            "disease": "Blast — Tricyclazole; BPH — Imidacloprid",
            "companion": "Azolla (biofertilizer)",
            "market": "Premium export variety; highest price; Middle East & Europe demand",
        },
        {
            "name": "IR-64 (Non-Basmati)",
            "best_season": "Kharif — transplant Jun–Jul",
            "soil": "Clay to clay loam, pH 5.5–7.0",
            "temp_range": "25–32°C",
            "water_need": "High — 1000–1200 mm",
            "fertilizer": "NPK 100:50:50 kg/ha",
            "duration": "110–120 days",
            "yield": "5–6 t/ha",
            "disease": "Sheath blight — Hexaconazole; Gall midge — resistant variety",
            "companion": "Azolla, Sesbania",
            "market": "Domestic staple; PDS procurement; stable MSP",
        },
    ],
    "maize": [
        {
            "name": "DKC-9144 (Hybrid Maize)",
            "best_season": "Kharif (Jun–Jul)",
            "soil": "Well-drained loam, pH 6.0–7.5",
            "temp_range": "24–30°C",
            "water_need": "Moderate — 500–700 mm; critical at tasseling",
            "fertilizer": "NPK 150:75:75 kg/ha; Zinc 25 kg/ha",
            "duration": "95–105 days",
            "yield": "7–9 t/ha",
            "disease": "Turcicum blight — Mancozeb; FAW — Emamectin benzoate",
            "companion": "Beans, Squash (Three Sisters)",
            "market": "Poultry feed industry; starch industry; ethanol production",
        },
    ],
    "cotton": [
        {
            "name": "Bt Cotton (Bollgard II)",
            "best_season": "Kharif — sow Apr–May",
            "soil": "Black cotton soil (Vertisol), pH 6.5–8.0",
            "temp_range": "25–35°C",
            "water_need": "Moderate — drip 600–800 mm; critical at boll development",
            "fertilizer": "NPK 150:60:60 kg/ha; Boron 1 kg/ha",
            "duration": "160–180 days",
            "yield": "2.5–3.5 t/ha (seed cotton)",
            "disease": "Pink bollworm resistant; Leaf curl virus — whitefly control",
            "companion": "Cowpea, Sorghum (border crop)",
            "market": "Textile industry; MSP support; export to Bangladesh & China",
        },
    ],
    "sugarcane": [
        {
            "name": "Co-0238 (Sugarcane)",
            "best_season": "Spring (Feb–Mar) or Autumn (Oct–Nov)",
            "soil": "Deep loamy, pH 6.5–7.5",
            "temp_range": "27–38°C germination; 20–25°C ripening",
            "water_need": "High — furrow/drip 1500–2500 mm/year",
            "fertilizer": "NPK 250:85:120 kg/ha in 3 splits",
            "duration": "12 months",
            "yield": "80–100 t/ha",
            "disease": "Red rot — use disease-free setts; Smut — hot water treatment",
            "companion": "Garlic, Onion (intercrop in early stage)",
            "market": "Sugar mills; ethanol; jaggery; FRP guaranteed by government",
        },
    ],
    "barley": [
        {
            "name": "RD-2794 (Barley)",
            "best_season": "Rabi — sow Oct 25–Nov 10",
            "soil": "Sandy loam to loam, pH 6.5–8.0",
            "temp_range": "15–20°C",
            "water_need": "Very low — 2–3 irrigations; 250–350 mm",
            "fertilizer": "NPK 80:40:30 kg/ha",
            "duration": "100–110 days",
            "yield": "3–4 t/ha",
            "disease": "Stripe rust — Propiconazole; Loose smut — seed treatment",
            "companion": "Mustard, Chickpea",
            "market": "Malt barley for brewery; animal feed; flour industry",
        },
    ],
}


class CropRecommender:
    def __init__(self):
        self.scaler  = StandardScaler()
        self.kmeans  = None
        self.df      = None
        self.cluster_crop_map = {}   # cluster_id → dominant crop
        self.cluster_profiles = {}   # cluster_id → feature means

    # ── Training ─────────────────────────────────────────────────────────────
    def train(self, data_path=DATA_PATH):
        self.df = pd.read_csv(data_path)
        X = self.df[FEATURES].values
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=20)
        self.df["cluster"] = self.kmeans.fit_predict(X_scaled)

        # Map each cluster to its dominant crop
        for cid in range(OPTIMAL_K):
            mask = self.df["cluster"] == cid
            dominant = self.df.loc[mask, "crop"].value_counts().idxmax()
            self.cluster_crop_map[cid] = dominant
            self.cluster_profiles[cid] = self.df.loc[mask, FEATURES].mean().to_dict()

        # Persist
        os.makedirs("data", exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"scaler": self.scaler, "kmeans": self.kmeans,
                         "map": self.cluster_crop_map,
                         "profiles": self.cluster_profiles}, f)
        print("Model trained and saved.")
        return self

    def load(self):
        if not os.path.exists(MODEL_PATH):
            from ml.dataset import generate_dataset
            generate_dataset()
            self.train()
            return self
        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)
        self.scaler, self.kmeans = obj["scaler"], obj["kmeans"]
        self.cluster_crop_map   = obj["map"]
        self.cluster_profiles   = obj["profiles"]
        return self

    # ── Prediction ───────────────────────────────────────────────────────────
    def predict(self, temp, humidity, rainfall, ph, moisture, N, P, K):
        x = np.array([[temp, humidity, rainfall, ph, moisture, N, P, K]])
        x_scaled = self.scaler.transform(x)
        cluster  = int(self.kmeans.predict(x_scaled)[0])
        crop     = self.cluster_crop_map[cluster]

        # Suitability: distance-based score (closer = higher %)
        center   = self.kmeans.cluster_centers_[cluster]
        dist     = np.linalg.norm(x_scaled - center)
        max_dist = 5.0  # empirical cap
        suitability = max(0, round((1 - dist / max_dist) * 100, 1))

        # Per-crop suitability across all clusters
        all_scores = {}
        for cid, c_crop in self.cluster_crop_map.items():
            d = np.linalg.norm(x_scaled - self.kmeans.cluster_centers_[cid])
            all_scores[c_crop] = max(0, round((1 - d / max_dist) * 100, 1))

        # Explanation
        profile = self.cluster_profiles[cluster]
        reasons = self._explain(temp, humidity, rainfall, ph, moisture, N, P, K, profile, crop)

        return {
            "recommended_crop": crop,
            "cluster_id": cluster,
            "suitability_score": min(suitability, 99.9),
            "all_scores": dict(sorted(all_scores.items(), key=lambda x: -x[1])),
            "alternatives": ALTERNATIVES.get(crop, []),
            "fertilizer": FERTILIZER_ADVICE.get(crop, ""),
            "irrigation": IRRIGATION_ADVICE.get(crop, ""),
            "seasonal_tip": SEASONAL_TIPS.get(crop, ""),
            "yield_estimate": YIELD_ESTIMATE.get(crop, ""),
            "reasons": reasons,
            "varieties": VARIETY_DETAILS.get(crop, []),
            "input": {"temp":temp,"humidity":humidity,"rainfall":rainfall,
                      "ph":ph,"moisture":moisture,"N":N,"P":P,"K":K},
        }

    def _explain(self, temp, humidity, rainfall, ph, moisture, N, P, K, profile, crop):
        reasons = []
        checks = [
            ("Temperature", temp, profile["temp"], "°C"),
            ("Humidity",    humidity, profile["humidity"], "%"),
            ("Rainfall",    rainfall, profile["rainfall"], " mm"),
            ("Soil pH",     ph, profile["ph"], ""),
            ("Soil Moisture", moisture, profile["moisture"], "%"),
            ("Nitrogen (N)", N, profile["N"], " kg/ha"),
            ("Phosphorus (P)", P, profile["P"], " kg/ha"),
            ("Potassium (K)", K, profile["K"], " kg/ha"),
        ]
        for label, val, ideal, unit in checks:
            diff = abs(val - ideal)
            pct  = diff / (ideal + 1e-9) * 100
            if pct < 15:
                reasons.append(f"OK {label} ({val:.1f}{unit}) closely matches ideal for {crop} ({ideal:.1f}{unit}).")
            elif pct < 35:
                direction = "above" if val > ideal else "below"
                reasons.append(f"WARN {label} is slightly {direction} ideal ({ideal:.1f}{unit}); manageable.")
            else:
                direction = "high" if val > ideal else "low"
                reasons.append(f"BAD {label} is {direction} ({val:.1f}{unit}); ideal is {ideal:.1f}{unit}.")
        return reasons

    # ── Variety details endpoint ────────────────────────────────────────────────
    def variety_details(self, crop=None):
        if crop:
            return VARIETY_DETAILS.get(crop, [])
        return VARIETY_DETAILS

    # ── Elbow data for chart ──────────────────────────────────────────────────
    def elbow_data(self, data_path=DATA_PATH):
        df = pd.read_csv(data_path)
        X  = self.scaler.transform(df[FEATURES].values)
        inertias = []
        ks = list(range(2, 13))
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(round(km.inertia_, 2))
        return {"k": ks, "inertia": inertias}

    # ── Cluster summary for dashboard ────────────────────────────────────────
    def cluster_summary(self):
        return [
            {"cluster": cid, "crop": crop,
             "profile": {k: round(v, 2) for k, v in self.cluster_profiles[cid].items()}}
            for cid, crop in self.cluster_crop_map.items()
        ]


# Singleton
_recommender = None

def get_recommender():
    global _recommender
    if _recommender is None:
        _recommender = CropRecommender().load()
    return _recommender
