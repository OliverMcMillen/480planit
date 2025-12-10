# 480planit – Trip Packing Assistant (MLP)

This project uses a **Multi-Layer Perceptron (MLP)** model to suggest items travelers should bring on a trip.  
It takes trip details (destination, season, duration, weather, activities, etc.) and predicts the most likely items needed, grouped by category.

---

## ⚙️ Setup and running the model:
1. Clone the repository:\
git clone https://github.com/OliverMcMillen/480planit.git
2. Configure virtual environment:\
python -m venv .venv\
Activate virtual environment:\
source .venv/bin/activate
3. Delete files in /models directory. These will be regenerated. 
4. Install dependencies:
pip install -r src/requirements.txt 
5. Train the model: Run train_trip_mlp.py
   6. The model will be trained on the data in the /data directory.
6. Run the predict_items_script.py to use the model and test predictions. Adjust the example_trip used at the bottom of the script.

