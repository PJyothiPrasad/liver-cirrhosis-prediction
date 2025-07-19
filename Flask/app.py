from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and imputer with specified filenames
model = joblib.load('Flask/rf_acc_68.pkl')
imputer = joblib.load('Flask/normalizer.pkl')

# List of features expected in the form
feature_names = [
    'Duration_of_alcohol_consumptionyears', 'Total_Bilirubin_mgdl',
    'RBC_million_cellsmicroliter', 'USG_Abdomen_diffuse_liver_or_not',
    'MCHC_gramsdeciliter', 'Direct_mgdl', 'ALPhosphatase_UL',
    'Platelet_Count_lakhsmm', 'Lymphocytes_', 'AG_Ratio', 'SGOTAST_UL',
    'PCV_', 'Total_Count', 'Albumin_gdl', 'Indirect_mgdl'
]

@app.route('/')
def form():
    return render_template('forms/index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    input_data = [float(request.form[feature]) for feature in feature_names]

    # Impute and predict
    input_array = np.array(input_data).reshape(1, -1)
    input_array = imputer.transform(input_array)
    prediction = model.predict(input_array)[0]

    result = "Patient likely has liver cirrhosis." if prediction == 1 else "Patient likely does NOT have liver cirrhosis."
    return render_template('forms/index.html', prediction_text=result, feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)
