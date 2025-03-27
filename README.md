# Machine Learning Models Web Application

This Django web application implements five different machine learning models:
1. Simple Linear Regression - Electricity Consumption Prediction
2. Multiple Linear Regression - Health Risk Prediction
3. Polynomial Regression - Fuel Efficiency Prediction
4. Logistic Regression - Diabetes Prediction
5. KNN Classification - Car Fuel Type Prediction

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Install required packages:
```bash
pip install django pandas numpy scikit-learn joblib
```

4. Train all models:
```bash
python train_all_models.py
```

5. Run the Django development server:
```bash
python manage.py runserver
```

6. Open your web browser and navigate to:
```
http://127.0.0.1:8000
```

## Features

- User-friendly interface for all five machine learning models
- Real-time predictions
- Model accuracy tracking
- No authentication required
- Responsive design using Bootstrap

## Model Details

### Simple Linear Regression
- Predicts electricity consumption based on:
  - Temperature
  - Humidity
  - Wind Speed

### Multiple Linear Regression
- Predicts health risk based on:
  - Age
  - BMI
  - Blood Pressure
  - Cholesterol

### Polynomial Regression
- Predicts fuel efficiency based on:
  - Engine Size
  - Horsepower
  - Weight
  - Acceleration

### Logistic Regression
- Predicts diabetes based on:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age

### KNN Classification
- Predicts car fuel type based on:
  - Engine Size
  - Horsepower
  - Weight
  - Acceleration
  - Fuel Efficiency

## Note

Make sure all the required CSV files are present in the root directory before training the models. 