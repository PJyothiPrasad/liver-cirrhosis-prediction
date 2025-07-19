import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
data = pd.read_csv('Data/HealthCareData.csv')

# List of important features
important_features = [
    'Duration_of_alcohol_consumptionyears', 'Total_Bilirubin_mgdl',
    'RBC_million_cellsmicroliter', 'USG_Abdomen_diffuse_liver_or_not',
    'MCHC_gramsdeciliter', 'Direct_mgdl', 'ALPhosphatase_UL',
    'Platelet_Count_lakhsmm', 'Lymphocytes_', 'AG_Ratio', 'SGOTAST_UL',
    'PCV_', 'Total_Count', 'Albumin_gdl', 'Indirect_mgdl'
]

# Separate features and target
X = data[important_features]
y = data['Predicted_ValueOut_ComePatient_suffering_from_liver_cirrosis_or_not']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42, stratify=y)

# Train Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Save the trained model and imputer with specified filenames
joblib.dump(model, 'Flask/rf_acc_68.pkl')
joblib.dump(imputer, 'Flask/normalizer.pkl')

print("✅ Model saved as 'rf_acc_68.pkl' and imputer as 'normalizer.pkl'.")
