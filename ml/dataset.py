"""
Synthetic Dataset Generator for AI Crop Recommendation System
Generates realistic agricultural data for 9 crop types
"""

import numpy as np
import pandas as pd
import os

# Crop profiles: [temp, humidity, rainfall, pH, moisture, N, P, K]
# Each tuple: (mean, std) per feature
CROP_PROFILES = {
    "wheat":      {"temp":(22,3),  "humidity":(55,8),  "rainfall":(60,15),  "ph":(6.5,0.4), "moisture":(40,8),  "N":(80,15), "P":(40,10), "K":(40,10)},
    "rice":       {"temp":(28,3),  "humidity":(82,6),  "rainfall":(200,30), "ph":(6.0,0.4), "moisture":(75,8),  "N":(90,15), "P":(45,10), "K":(45,10)},
    "cotton":     {"temp":(30,3),  "humidity":(60,8),  "rainfall":(80,20),  "ph":(7.0,0.4), "moisture":(45,8),  "N":(120,15),"P":(50,10), "K":(50,10)},
    "sugarcane":  {"temp":(32,3),  "humidity":(75,6),  "rainfall":(180,25), "ph":(6.5,0.4), "moisture":(70,8),  "N":(100,15),"P":(50,10), "K":(50,10)},
    "maize":      {"temp":(26,3),  "humidity":(65,8),  "rainfall":(100,20), "ph":(6.2,0.4), "moisture":(55,8),  "N":(85,15), "P":(45,10), "K":(45,10)},
    "barley":     {"temp":(18,3),  "humidity":(50,8),  "rainfall":(50,12),  "ph":(6.8,0.4), "moisture":(35,8),  "N":(70,15), "P":(35,10), "K":(35,10)},
    "pulses":     {"temp":(24,3),  "humidity":(58,8),  "rainfall":(70,15),  "ph":(6.3,0.4), "moisture":(42,8),  "N":(20,10), "P":(60,10), "K":(80,10)},
    "vegetables": {"temp":(25,3),  "humidity":(70,8),  "rainfall":(120,20), "ph":(6.4,0.4), "moisture":(60,8),  "N":(110,15),"P":(55,10), "K":(55,10)},
    "fruits":     {"temp":(27,3),  "humidity":(72,8),  "rainfall":(150,25), "ph":(6.6,0.4), "moisture":(65,8),  "N":(95,15), "P":(50,10), "K":(50,10)},
}

SAMPLES_PER_CROP = 200

def generate_dataset(save_path="data/crop_data.csv"):
    np.random.seed(42)
    rows = []
    for crop, profile in CROP_PROFILES.items():
        for _ in range(SAMPLES_PER_CROP):
            row = {feat: max(0, np.random.normal(v[0], v[1])) for feat, v in profile.items()}
            row["crop"] = crop
            rows.append(row)
    df = pd.DataFrame(rows)[["temp","humidity","rainfall","ph","moisture","N","P","K","crop"]]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Dataset saved: {save_path} | Shape: {df.shape}")
    return df

if __name__ == "__main__":
    generate_dataset()
