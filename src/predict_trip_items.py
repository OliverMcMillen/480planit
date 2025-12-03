import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os

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

MODELS_DIR = "models"
PREPROC_PATH = os.path.join(MODELS_DIR, "preprocessors.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "trained_trip_item_mlp.pth")

with open(PREPROC_PATH, "rb") as f:
    saved = pickle.load(f)

encoder = saved["encoder"]
scaler = saved["scaler"]
mlb = saved["mlb"]
categorical_cols = saved["categorical_cols"]
numeric_cols = saved["numeric_cols"]

# we also want names for output
catalog_df = pd.read_excel("data/ItemCatalog_Dec3rd.xlsx")
catalog_df.columns = catalog_df.columns.str.strip().str.lower()
id_to_name = {int(r["id"]): r["name"] for _, r in catalog_df.iterrows()}

# build model with correct sizes
input_size = len(numeric_cols) + encoder.transform(
    pd.DataFrame([{c: "" for c in categorical_cols}])
).shape[1]  # trick to get encoded size

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
        "activities": "skiing",
        "duration_days": 5,
        "avg_temp_high": 30,
        "avg_temp_low": 10,
        "rain_chance_percent": 10,
        "humidity_percent": 40,
        "uv_index": 3,
        "altitude_m": 1600
    }
    Only the cols present in your training file will be used.
    """

    # make a 1-row DataFrame so encoder/scaler work
    df_row = pd.DataFrame([user_trip])

    # ensure all expected numeric columns exist
    for col in numeric_cols:
        if col not in df_row.columns:
            df_row[col] = 0

    # ensure all expected categorical columns exist
    for col in categorical_cols:
        if col not in df_row.columns:
            df_row[col] = ""

    # order columns
    df_num = df_row[numeric_cols]
    df_cat = df_row[categorical_cols]

    # scale nums
    scaled_nums = scaler.transform(df_num)

    # encode cats
    encoded_cats = encoder.transform(df_cat)

    # combine
    x = np.concatenate([scaled_nums, encoded_cats], axis=1).astype(np.float32)
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
    example_trip = {
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

    results = predict_items(example_trip, threshold=0.65, fallback_topk=5)
    print("Predicted items to pack:")
    for r in results:
        print(f"- {r['item_name']} ({r['prob']:.3f})")