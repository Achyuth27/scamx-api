from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import re, unicodedata, contractions

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer from Hugging Face
MODEL_NAME = "JeswinMS4/scam-alert-mobile-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Initialize the pipeline for text classification
scam_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Text cleaning function
def clean_text(text):
    text = unicodedata.normalize("NFKD", text)      # Normalize unicode characters
    text = contractions.fix(text)                   # Expand contractions
    text = re.sub(r"[^\w\s]", "", text)             # Remove punctuation
    text = text.lower().strip()                     # Convert to lowercase and trim
    return text

# Define the route for classifying messages
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' field"}), 400

        raw_text = data["message"]
        cleaned_text = clean_text(raw_text)

        # Get model prediction
        prediction = scam_classifier(cleaned_text)[0]
        label = prediction['label']
        confidence = round(prediction['score'] * 100, 2)

        # Return response
        return jsonify({
            "input": raw_text,
            "label": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    print("🔧 Server running on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
