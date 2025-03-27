import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import os

def train_polynomial():
    # Load the dataset
    df = pd.read_csv('car_performance_prediction.csv')
    
    # Prepare features and target
    X = df[['Speed', 'Engine_Performance']]  # Use Speed and Engine_Performance as features
    y = df['Fuel_Efficiency']  # Fuel efficiency as target
    
    # Create polynomial features (degree=2)
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy (R² score)
    accuracy = r2_score(y_test, y_pred)
    print(f"Model Accuracy (R² Score): {accuracy}")
    
    # Save the model and polynomial features
    if not os.path.exists('ml_app/models'):
        os.makedirs('ml_app/models')
    joblib.dump(model, 'ml_app/models/polynomial_model.pkl')
    joblib.dump(poly_features, 'ml_app/models/polynomial_features.pkl')
    
    return accuracy

if __name__ == "__main__":
    train_polynomial() 