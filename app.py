from flask import Flask, request, jsonify
import joblib

print("🚀 Starting Smart Traffic Congestion Predictor...")

app = Flask(__name__)

# Load model and encoder
model = joblib.load("model/traffic_model.pkl")
encoder = joblib.load("model/junction_encoder.pkl")

@app.route("/")
def home():
    return "🚦 Smart Traffic Congestion Predictor API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    hour = data["hour"]
    day = data["day"]
    junction = data["junction"]

    junction_encoded = encoder.transform([junction])[0]
    prediction = model.predict([[hour, day, junction_encoded]])

    # Convert to congestion level
    if prediction[0] < 50:
        level = "Low"
    elif prediction[0] < 120:
        level = "Medium"
    else:
        level = "High"

    return jsonify({
        "predicted_vehicle_count": int(prediction[0]),
        "congestion_level": level
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
