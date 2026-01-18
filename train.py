import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("traffic.csv")

print("Dataset columns:", data.columns)

# Convert DateTime column
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['hour'] = data['DateTime'].dt.hour
data['day'] = data['DateTime'].dt.day

# Encode Junction
encoder = LabelEncoder()
data['Junction'] = encoder.fit_transform(data['Junction'])

# Features and target
X = data[['hour', 'day', 'Junction']]
y = data['Vehicles']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

# Save model
joblib.dump(model, "model/traffic_model.pkl")
joblib.dump(encoder, "model/junction_encoder.pkl")

print("✅ Model and encoder saved successfully!")
