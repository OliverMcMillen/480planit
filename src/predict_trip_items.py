import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

PREPROC_PATH = os.path.join(MODELS_DIR, "preprocessors_activities.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "trained_trip_item_mlp.pth")

class TripItemMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)

with open(PREPROC_PATH, "rb") as f:
    saved = pickle.load(f)

print("PREDICT: reading preprocessors from:",
      os.path.abspath(PREPROC_PATH))
print("PREDICT: saved keys:", saved.keys())

encoder = saved["encoder"]
scaler = saved["scaler"]
mlb = saved["mlb"]
categorical_cols = saved["categorical_cols"]
numeric_cols = saved["numeric_cols"]
activity_cols = saved["activity_cols"]
activity_map = saved["activity_map"]
all_activities = saved["all_activities"]

def parse_and_canonicalize_activities(value: str) -> set[str]:
    """Parse raw activities string and map to canonical activity types."""
    if not value:
        return set()
    raw = [a.strip() for a in str(value).split(",") if a.strip()]
    canon = set()
    for a in raw:
        canon_name = activity_map.get(a, a)  # fallback to raw if not in map
        canon.add(canon_name)
    return canon

# we also want names for output
catalog_df = pd.read_excel(os.path.join(DATA_DIR, "ItemCatalog_Dec3rd.xlsx"))
catalog_df.columns = catalog_df.columns.str.strip().str.lower()
id_to_name = {int(r["id"]): r["name"] for _, r in catalog_df.iterrows()}

# build model with correct sizes
encoded_size = encoder.transform(
    pd.DataFrame([{c: "" for c in categorical_cols}])
).shape[1]
input_size = len(numeric_cols) + encoded_size + len(activity_cols)

output_size = len(mlb.classes_)
model = TripItemMLP(input_size=input_size, hidden_size=256, output_size=output_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def build_features_from_input(user_trip: dict) -> torch.Tensor:
    """
    user_trip is a dict like:
    {
        "destination": "Denver, CO",
        "season": "winter",
        "weather": "snow",
        "activities": "ski_resort, hiking_area",
        "duration_days": 5,
        ...
    }
    Only the cols present in your training file will be used.
    """
    # 1) Build a DataFrame row for numeric + categorical cols
    df_row = pd.DataFrame([user_trip])

    # ensure numeric columns exist
    for col in numeric_cols:
        if col not in df_row.columns:
            df_row[col] = 0

    # ensure categorical columns exist
    for col in categorical_cols:
        if col not in df_row.columns:
            df_row[col] = ""

    # enforce column order
    df_num = df_row[numeric_cols]
    df_cat = df_row[categorical_cols]

    # 2) Scale numeric
    scaled_nums = scaler.transform(df_num)            # shape (1, len(numeric_cols))

    # 3) Encode categorical (destination, season, weather)
    encoded_cats = encoder.transform(df_cat)          # shape (1, encoded_cat_dim)

    # 4) Canonicalize and multi-hot encode activities
    acts_set = parse_and_canonicalize_activities(user_trip.get("activities", ""))

    activity_vector = np.zeros((1, len(activity_cols)), dtype=np.float32)
    for idx, canon_name in enumerate(all_activities):
        if canon_name in acts_set:
            activity_vector[0, idx] = 1.0

    # 5) Concatenate all parts: [nums | cats | activities]
    x = np.concatenate([scaled_nums, encoded_cats, activity_vector], axis=1).astype(np.float32)

    x_tensor = torch.from_numpy(x)
    return x_tensor  # shape (1, input_size)

def predict_items(user_trip: dict, threshold: float = 0.65, fallback_topk: int = 5):
    x = build_features_from_input(user_trip)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).numpy()[0]  # shape (num_items,)

    # binary mask
    mask = probs > threshold
    picked_idxs = np.where(mask)[0]

    # if nothing above threshold, pick top-k
    if len(picked_idxs) == 0:
        topk = np.argsort(-probs)[:fallback_topk]
        picked_idxs = topk

    # map from mlb column index to item id
    item_ids = [int(mlb.classes_[i]) for i in picked_idxs]
    item_names = [id_to_name.get(i, str(i)) for i in item_ids]

    # also return probabilities for transparency
    scored = [
        {
            "item_id": int(mlb.classes_[i]),
            "item_name": id_to_name.get(int(mlb.classes_[i]), str(int(mlb.classes_[i]))),
            "prob": float(probs[i]),
        }
        for i in picked_idxs
    ]

    # sort descending by prob
    scored.sort(key=lambda d: d["prob"], reverse=True)
    return scored

if __name__ == "__main__":
    example_trip1 = {
        "destination": "Grand Canyon, AZ",
        "season": "summer",
        "weather": "hot",
        "activities": "campground, hiking_area, national_park, historical_landmark, monument",   # must be a single value like in training
        "duration_days": 6,
        "avg_temp_high": 85,
        "avg_temp_low": 75,
        "rain_chance_percent": 10,
        "humidity_percent": 16,
    }

    example_trip2 = {
        "destination": "Honolulu, HI",
        "season": "fall",
        "activities": "hiking_area,park,beach",
        "duration_days": 8,
        "avg_temp_high": 76,
        "avg_temp_low": 76,
        "avg_precipitation_chance": 73,
        "humidity_percent": 51,
    }

    example_trip3 = {
        "destination": "Ocean City, MD",
        "season": "summer",
        "activities": "public_bathroom,beach,museum,,restaurant,american_restaurant,mediterranean_restaurant,nail_salon,tourist_attraction,seafood_restaurant,water_park,pub,bar_and_grill,stable,hospital,movie_theater,video_arcade,shopping_mall,gym,amusement_park,coffee_shop,spa,health",
        "duration_days": 7,
        "avg_temp_high": 75,
        "avg_temp_low": 75,
        "avg_precipitation_chance": 13,
        "humidity_percent": 83,
    }

    example_trip4 = {
        "destination": "Breckenridge, CO",
        "season": "winter",
        "activities": "ski_resort",
        "duration_days": 4,
        "avg_temp_high": 28,
        "avg_temp_low": 14,
        "avg_precipitation_chance": 50,
        "humidity_percent": 83,
    }

    results = predict_items(example_trip2, threshold=0.25, fallback_topk=5)
    print("Predicted items to pack:")
    for r in results:
        print(f"- {r['item_name']} ({r['prob']:.3f})")