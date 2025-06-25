from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# === Configurable Paths ===
MODEL_PATH = "rf_acc_68.pkl"
NORMALIZER_PATH = "normalizer.pkl"

# === Load Model & Normalizer ===
def load_pickle_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    with open(path, "rb") as file:
        return pickle.load(file)

try:
    model = load_pickle_file(MODEL_PATH)
    normalizer = load_pickle_file(NORMALIZER_PATH)
except Exception as e:
    print(f"❌ Error loading model or normalizer: {e}")
    raise SystemExit("Exiting due to model load failure.")

# === Routes ===
@app.route('/')
def index():
    return render_template('forms/index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and transform features
        features = [float(request.form[key]) for key in request.form]
        features = np.array([features])
        features = normalizer.transform(features)

        # Predict
        prediction = model.predict(features)[0]
        result = "✅ Liver Cirrhosis Detected" if prediction == 1 else "✅ No Cirrhosis Detected"
        return render_template('forms/index.html', prediction=result)
    except Exception as e:
        error_msg = f"❌ Error during prediction: {str(e)}"
        print(error_msg)
        return render_template('forms/index.html', prediction=error_msg)

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
