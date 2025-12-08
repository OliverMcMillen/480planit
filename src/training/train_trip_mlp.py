import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Load Excel files
trip_df = pd.read_excel(os.path.join(DATA_DIR, "trip_scenarios_1000_Dec3rd.xlsx"))
catalog_df = pd.read_excel(os.path.join(DATA_DIR, "ItemCatalog_Dec3rd.xlsx"))

activity_map = {
    "hiking_area": "outdoor_hiking",
    "national_park": "outdoor_hiking",
    "campground": "outdoor_hiking",
    "state_park": "outdoor_hiking",
    "park": "outdoor_hiking",

    "beach": "beach_water",
    "water_park": "beach_water",

    "marina": "fish_charter",

    "ski_resort": "skiing",

    "museum": "sightseeing",
    "art_gallery": "sightseeing",
    "historical_landmark": "sightseeing",
    "observation_deck": "sightseeing",

}

def parse_and_canonicalize_activities(s: str):
    if pd.isna(s):
        return set()
    raw = [a.strip() for a in str(s).split(',') if a.strip()]
    canon = set()
    for a in raw:
        if a in activity_map:
            canon.add(activity_map[a])
        else:
            canon.add(a)  # fallback: keep raw
    return canon

trip_df["activities_parsed"] = trip_df["activities"].apply(parse_and_canonicalize_activities)

all_activities = sorted({a for acts in trip_df["activities_parsed"] for a in acts})
activity_cols = [f"act__{a}" for a in all_activities]

print("Using activity columns:", activity_cols)


print(activity_cols)

for a in all_activities:
    col = f"act__{a}"
    trip_df[col] = trip_df["activities_parsed"].apply(lambda acts: 1.0 if a in acts else 0.0)

# Standardize column names early so later code can rely on them
trip_df.columns = trip_df.columns.str.strip().str.lower()
catalog_df.columns = catalog_df.columns.str.strip().str.lower()

# The trip dataset stores item names (text) separated by '|' — map those names to catalog ids.
def normalize_name(s: str) -> str:
    s = str(s).lower()
    # replace slashes with space, remove parentheses and punctuation except alphanumeric and spaces
    s = s.replace('/', ' ')
    s = re.sub(r"[^a-z0-9 ]+", ' ', s)
    s = re.sub(r"\s+", ' ', s).strip()
    return s

# build mapping from normalized catalog name -> list of ids (handle duplicate names)
catalog_df['name_norm'] = catalog_df['name'].apply(normalize_name)
name_to_ids = catalog_df.groupby('name_norm')['id'].apply(list).to_dict()

def map_item_names_to_ids(cell):
    # split by '|' (dataset uses |) and map each token to one or more ids
    parts = [p.strip() for p in str(cell).split('|') if p.strip()]
    ids = []
    for p in parts:
        norm = normalize_name(p)
        if norm in name_to_ids:
            ids.extend([int(i) for i in name_to_ids[norm]])
        else:
            # not found in catalog; ignore or log — here we ignore
            pass
    # deduplicate
    return sorted(set(ids))

trip_df["items"] = trip_df["items"].apply(map_item_names_to_ids)

# Define feature columns
categorical_cols = ["destination", "season", "weather"]

# Only use columns that exist
categorical_cols = [c for c in categorical_cols if c in trip_df.columns]

# intended numeric columns
intended_numeric_cols = [
    "duration_days", "avg_temp_high", "avg_temp_low",
    "avg_precipitation_chance", "humidity_percent"
]

# Filter to existing numeric columns
numeric_cols = [col for col in intended_numeric_cols if col in trip_df.columns]
print("Using numeric columns:", numeric_cols)

# Encode categorical and numeric data
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

# Encode only if categorical_cols exist
if categorical_cols:
    encoded_cats = encoder.fit_transform(trip_df[categorical_cols])
    encoded_cat_cols = encoder.get_feature_names_out(categorical_cols)
else:
    encoded_cats = np.zeros((len(trip_df), 0))
    encoded_cat_cols = []

# Scale numeric features
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(trip_df[numeric_cols])

# new:
activity_matrix = trip_df[activity_cols].values.astype(np.float32)

X = np.concatenate([scaled_nums, encoded_cats, activity_matrix], axis=1)


# Multi-label encode items
# Ensure catalog ids are ints and provide them as ordered class list to the binarizer
catalog_ids = catalog_df['id'].astype(int).tolist()
mlb = MultiLabelBinarizer(classes=catalog_ids)
Y = mlb.fit_transform(trip_df["items"])


# Keep row indices so we can map test rows back to the original dataframe for CSV output
indices = np.arange(X.shape[0])
X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
    X, Y, indices, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)
# keep a numpy copy of test targets and indices for CSV/export
Y_test_np_for_export = Y_test.numpy()
idx_test_for_export = idx_test

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=8, shuffle=True)


# Define the MLP model
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

# Initialize model
model = TripItemMLP(
    input_size=X_train.shape[1],
    hidden_size=256,
    output_size=Y_train.shape[1]
)


# Training setup
criterion = nn.BCEWithLogitsLoss()   # multi-label loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100


# Train loop
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}] | Loss: {total_loss/len(train_loader):.4f}")

#Evaluate model
print("\nStarting evaluation...")
model.eval()
with torch.no_grad():

    preds = torch.sigmoid(model(X_test))
    # preds_binary = (preds > 0.65).float()
    for thresh in [0.3, 0.4, 0.5]:
        preds_binary = (preds > thresh).float()
    accuracy = (preds_binary == Y_test).float().mean()
    print(f"\n Test Accuracy: {accuracy:.4f}")


# --- Prepare CSV output of predictions for inspection/evaluation ---
preds_np = preds.numpy()
preds_binary_np = preds_binary.numpy()

# Inverse-transform true labels back to item ids (lists)
true_items_list = mlb.inverse_transform(Y_test_np_for_export)

# Get class ids (as stored in the mlb) corresponding to columns
class_ids = list(mlb.classes_)

# build id -> name mapping from catalog for human-readable output
id_to_name = {int(r['id']): r['name'] for _, r in catalog_df.iterrows()}

# Build a simple human-readable summary for each test sample
rows = []
top_10 = 10
for i, orig_idx in enumerate(idx_test_for_export):
    probs = preds_np[i]
    bin_preds = preds_binary_np[i]

    # predicted item ids (binary)
    predicted_items = [int(class_ids[j]) for j in np.where(bin_preds == 1)[0]]

    # If no items meet the threshold, use top 5 predictions
    if len(predicted_items) == 0:
        top5_idx = np.argsort(-probs)[:5]
        predicted_items = [int(class_ids[j]) for j in top5_idx]

    # map ids to human-readable names
    true_names = [id_to_name.get(int(i), str(i)) for i in true_items_list[i]]
    predicted_names = [id_to_name.get(int(i), str(i)) for i in predicted_items]

    # top-k predictions with probabilities
    top10_idx = np.argsort(-probs)[:top_10]
    top10_pairs = [f"{int(class_ids[j])}:{probs[j]:.3f}" for j in top10_idx]

    # also build topk name:prob pairs
    top10_name_pairs = [f"{id_to_name.get(int(class_ids[j]), class_ids[j])}:{probs[j]:.3f}" for j in top10_idx]

    rows.append({
        "orig_row": int(orig_idx),
        "true_items": ";".join(map(str, true_items_list[i])) if len(true_items_list[i])>0 else "",
        "true_item_names": ";".join(true_names) if true_names else "",
        "predicted_items": ";".join(map(str, predicted_items)) if predicted_items else "",
        "predicted_item_names": ";".join(predicted_names) if predicted_names else "",
        "top10_predictions": ";".join(top10_pairs),
        "top10_predictions_names": ";".join(top10_name_pairs),
    })

results_df = pd.DataFrame(rows)

# Optionally join some original trip info for easier inspection (if available)
try:
    # add a few original columns if they exist
    add_cols = [c for c in ["destination", "season", "weather", "activities"] if c in trip_df.columns]
    if add_cols:
        # align by original dataframe index
        meta = trip_df.loc[results_df["orig_row"], add_cols].reset_index(drop=True)
        results_df = pd.concat([results_df, meta], axis=1)
except Exception:
    pass

os.makedirs(MODELS_DIR, exist_ok=True)
csv_path = os.path.join(MODELS_DIR, "predictions1000_STRICT.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nSaved prediction results to: {csv_path}")

MODEL_PATH = os.path.join(MODELS_DIR, "trained_trip_item_mlp.pth")
PREPROC_PATH = os.path.join(MODELS_DIR, "preprocessors_activities.pkl")

torch.save(model.state_dict(), MODEL_PATH)

print("TRAIN: writing preprocessors to:", os.path.abspath(PREPROC_PATH))
print("TRAIN: categorical_cols:", categorical_cols)
print("TRAIN: saving keys:",
      ["encoder", "scaler", "mlb", "categorical_cols",
       "numeric_cols", "activity_cols", "activity_map", "all_activities"])

with open(PREPROC_PATH, "wb") as f:
    pickle.dump({
        "encoder": encoder,
        "scaler": scaler,
        "activity_cols": activity_cols,
        "activity_map": activity_map,
        "all_activities": all_activities,
        "mlb": mlb,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,

    }, f)

print("\nModel and preprocessors saved successfully!")
