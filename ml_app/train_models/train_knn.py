import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

def train_knn():
    # Load the dataset
    df = pd.read_csv('car_fuel_type_classification.csv')
    
    # Prepare features and target
    X = df[['Engine_Size', 'Mileage']]  # Numeric features
    y = df['Fuel_Type']  # Target variable
    
    # Encode the target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    
    # Save the model, scaler, and encoder
    if not os.path.exists('ml_app/models'):
        os.makedirs('ml_app/models')
    joblib.dump(model, 'ml_app/models/knn_model.pkl')
    joblib.dump(scaler, 'ml_app/models/knn_scaler.pkl')
    joblib.dump(target_encoder, 'ml_app/models/knn_label_encoder.pkl')
    
    return accuracy

if __name__ == "__main__":
    train_knn() 