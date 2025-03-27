from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from .train_models.train_simple_linear import train_simple_linear
from .train_models.train_multiple_linear import train_multiple_linear
from .train_models.train_polynomial import train_polynomial
from .train_models.train_logistic import train_logistic
from .train_models.train_knn import train_knn

def home(request):
    return render(request, 'ml_app/home.html')

@csrf_exempt  # Temporarily disable CSRF for testing
def simple_linear(request):
    if request.method == 'POST':
        try:
            # Load the model
            model = joblib.load('ml_app/models/simple_linear_model.pkl')
            
            # Get time of day input from the form
            time_of_day = float(request.POST['feature_0'])
            
            # Make prediction with time of day
            prediction = model.predict([[time_of_day]])[0]
            
            return render(request, 'ml_app/simple_linear.html', {'prediction': prediction})
        except Exception as e:
            print(f"Error in simple_linear view: {str(e)}")
            return render(request, 'ml_app/simple_linear.html', {'error': str(e)})
    
    return render(request, 'ml_app/simple_linear.html')

@csrf_exempt  # Temporarily disable CSRF for testing
def multiple_linear(request):
    if request.method == 'POST':
        try:
            # Load the model
            model = joblib.load('ml_app/models/multiple_linear_model.pkl')
            
            # Get input values from the form
            input_values = [float(request.POST[f'feature_{i}']) for i in range(len(request.POST)-1)]
            
            # Make prediction
            prediction = model.predict([input_values])[0]
            
            return render(request, 'ml_app/multiple_linear.html', {'prediction': prediction})
        except Exception as e:
            print(f"Error in multiple_linear view: {str(e)}")
            return render(request, 'ml_app/multiple_linear.html', {'error': str(e)})
    
    return render(request, 'ml_app/multiple_linear.html')

@csrf_exempt  # Temporarily disable CSRF for testing
def polynomial(request):
    if request.method == 'POST':
        try:
            # Load the model and polynomial features
            model = joblib.load('ml_app/models/polynomial_model.pkl')
            poly_features = joblib.load('ml_app/models/polynomial_features.pkl')
            
            # Get input values from the form (only Speed and Engine_Performance)
            speed = float(request.POST['feature_0'])
            engine_performance = float(request.POST['feature_1'])
            
            # Transform input to polynomial features
            input_poly = poly_features.transform([[speed, engine_performance]])
            
            # Make prediction
            prediction = model.predict(input_poly)[0]
            
            return render(request, 'ml_app/polynomial.html', {'prediction': prediction})
        except Exception as e:
            print(f"Error in polynomial view: {str(e)}")
            return render(request, 'ml_app/polynomial.html', {'error': str(e)})
    
    return render(request, 'ml_app/polynomial.html')

@csrf_exempt  # Temporarily disable CSRF for testing
def logistic(request):
    if request.method == 'POST':
        try:
            # Load the model and scaler
            model = joblib.load('ml_app/models/logistic_model.pkl')
            scaler = joblib.load('ml_app/models/logistic_scaler.pkl')
            
            # Get input values from the form
            age = float(request.POST['feature_0'])
            bmi = float(request.POST['feature_1'])
            glucose = float(request.POST['feature_2'])
            blood_pressure = float(request.POST['feature_3'])
            
            # Scale input values
            input_scaled = scaler.transform([[age, bmi, glucose, blood_pressure]])
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            return render(request, 'ml_app/logistic.html', {
                'prediction': prediction,
                'probability': probability
            })
        except Exception as e:
            print(f"Error in logistic view: {str(e)}")
            return render(request, 'ml_app/logistic.html', {'error': str(e)})
    
    return render(request, 'ml_app/logistic.html')

@csrf_exempt  # Temporarily disable CSRF for testing
def knn(request):
    if request.method == 'POST':
        try:
            # Load the model, scaler, and label encoder
            model = joblib.load('ml_app/models/knn_model.pkl')
            scaler = joblib.load('ml_app/models/knn_scaler.pkl')
            label_encoder = joblib.load('ml_app/models/knn_label_encoder.pkl')
            
            # Get input values from the form (Engine_Size and Mileage)
            engine_size = float(request.POST['feature_0'])
            mileage = float(request.POST['feature_1'])
            
            # Scale input values
            input_scaled = scaler.transform([[engine_size, mileage]])
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Convert prediction back to original label
            prediction_label = label_encoder.inverse_transform([prediction])[0]
            
            return render(request, 'ml_app/knn.html', {
                'prediction': prediction_label,
                'probability': probability
            })
        except Exception as e:
            print(f"Error in knn view: {str(e)}")
            return render(request, 'ml_app/knn.html', {'error': str(e)})
    
    return render(request, 'ml_app/knn.html')
