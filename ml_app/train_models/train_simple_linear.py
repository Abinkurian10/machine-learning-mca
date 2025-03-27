import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import os

def train_simple_linear():
    # Load the dataset
    df = pd.read_csv('electricity_consumption.csv')
    
    # Prepare features and target
    X = df[['Time_of_Day']]  # Use Time_of_Day as feature
    y = df['Power_Usage']    # Power_Usage is the target variable
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy (R² score)
    accuracy = r2_score(y_test, y_pred)
    print(f"Model Accuracy (R² Score): {accuracy}")
    
    # Save the model
    if not os.path.exists('ml_app/models'):
        os.makedirs('ml_app/models')
    joblib.dump(model, 'ml_app/models/simple_linear_model.pkl')
    
    return accuracy

if __name__ == "__main__":
    train_simple_linear() 