import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

def train_logistic():
    # Load the dataset
    df = pd.read_csv('diabetes_prediction.csv')
    
    # Create synthetic negative cases by slightly modifying the existing data
    negative_cases = df.copy()
    negative_cases['Glucose_Level'] = negative_cases['Glucose_Level'] * 0.7  # Lower glucose levels
    negative_cases['BMI'] = negative_cases['BMI'] * 0.9  # Lower BMI
    negative_cases['Diabetes'] = 0  # Set as non-diabetic
    
    # Combine positive and negative cases
    df_combined = pd.concat([df, negative_cases], ignore_index=True)
    
    # Prepare features and target
    X = df_combined[['Age', 'BMI', 'Glucose_Level', 'Blood_Pressure']]
    y = df_combined['Diabetes']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    
    # Save the model and scaler
    if not os.path.exists('ml_app/models'):
        os.makedirs('ml_app/models')
    joblib.dump(model, 'ml_app/models/logistic_model.pkl')
    joblib.dump(scaler, 'ml_app/models/logistic_scaler.pkl')
    
    return accuracy

if __name__ == "__main__":
    train_logistic() 