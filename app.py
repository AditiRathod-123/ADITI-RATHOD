from pathlib import Path
import joblib
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fake_news_pipeline.joblib"

app = Flask(__name__)
pipeline = None

def load_model():
    global pipeline
    if MODEL_PATH.exists():
        pipeline = joblib.load(MODEL_PATH)

load_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global pipeline
    if pipeline is None:
        return jsonify({
            "error": "The model has not been trained yet. Run: python train_model.py"
        }), 503

    data = request.get_json(silent=True) or request.form
    news_text = (data.get("news_text") or "").strip()

    if len(news_text) < 25:
        return jsonify({"error": "Please enter at least 25 characters of news text."}), 400

    prediction = pipeline.predict([news_text])[0]
    probabilities = pipeline.predict_proba([news_text])[0]
    classes = list(pipeline.classes_)
    confidence = float(probabilities[classes.index(prediction)]) * 100

    # This is the model's score, not a verified probability that a claim is true.
    return jsonify({
        "prediction": str(prediction).upper(),
        "confidence": round(confidence, 1),
        "message": (
            "The model found language patterns more similar to its REAL training examples."
            if prediction == "REAL" else
            "The model found language patterns more similar to its FAKE training examples."
        ),
        "disclaimer": "This is an educational ML prediction, not a fact-check. Verify claims with reliable sources."
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": pipeline is not None})

if __name__ == "__main__":
    app.run(debug=True)
