from flask import Flask, request, jsonify
from predict_trip_items import predict_items
import traceback

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return [
        {"status": "ok"}
    ]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400
        results = predict_items(data, threshold=0.25, fallback_topk=5)
        return jsonify({
            "status": "success",
            "predicted_items": results
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)