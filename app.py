from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from pymongo import MongoClient
from dotenv import load_dotenv
import os, re, unicodedata, contractions

# Load .env variables (for local testing)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Load the Hugging Face model and tokenizer
MODEL_NAME = "JeswinMS4/scam-alert-mobile-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Initialize the pipeline
scam_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Connect to MongoDB
mongo_uri = os.environ.get("MONGO_URI")
client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
db = client["scamx"]
messages_collection = db["messages"]

# Text cleaning function
def clean_text(text):
    text = unicodedata.normalize("NFKD", text)      # Normalize unicode characters
    text = contractions.fix(text)                   # Expand contractions
    text = re.sub(r"[^\w\s]", "", text)             # Remove punctuation
    text = text.lower().strip()                     # Lowercase and trim
    return text

# API endpoint to predict message
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' field"}), 400

        raw_text = data["message"]
        cleaned_text = clean_text(raw_text)

        # Model prediction
        prediction = scam_classifier(cleaned_text)[0]
        label = prediction['label']
        confidence = round(prediction['score'] * 100, 2)

        # Store in MongoDB
        messages_collection.insert_one({
            "message": raw_text,
            "cleaned": cleaned_text,
            "label": label,
            "confidence": confidence
        })

        return jsonify({
            "input": raw_text,
            "label": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    print("🔧 Server running on http://127.0.0.1:5000")
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)

