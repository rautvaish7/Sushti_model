from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import pandas as pd

# Sample data
data = {
    "Appliances": [
        ["Air Conditioner", "Refrigerator"],
        ["Washing Machine", "Refrigerator"],
        ["Air Conditioner", "Washing Machine"]
    ],
    "Electricity_Consumption_kWh": [500, 350, 450]
}
df = pd.DataFrame(data)

# Encode appliances
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Appliances"])

# Train model
model = NearestNeighbors(n_neighbors=3, algorithm='auto')
model.fit(X)

# Save model and encoder
joblib.dump(model, "model.pkl")
joblib.dump(mlb, "appliance_encoder.pkl")
