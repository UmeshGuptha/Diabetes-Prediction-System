from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the diabetes dataset
diabetes_dataset = pd.read_excel('data/diabetes.xlsx')
# Prepare data and model as globals for simplicity (not recommended for production)
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
classifier = SVC(kernel='linear')
classifier.fit(X_train, Y_train)

@app.route('/')
def index():
    return render_template("prg12.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from form
    input_data = [
        request.form['Pregnancies'],
        request.form['Glucose'],
        request.form['BloodPressure'],
        request.form['SkinThickness'],
        request.form['Insulin'],
        request.form['BMI'],
        request.form['DiabetesPedigreeFunction'],
        request.form['Age']
    ]

    # Convert input data to numpy array and reshape
    input_data_np = np.asarray(input_data, dtype=np.float64).reshape(1, -1)

    # Standardize input data
    input_data_std = scaler.transform(input_data_np)

    # Make prediction
    prediction = classifier.predict(input_data_std)

    # Prepare response
    result = 'The Person is not Diabetic.' if prediction[0] == 0 else 'The Person is Diabetic. Please Consult a Diabetologist.'

    # Render the result in prg12.html
    return render_template("prg12.html", prediction_text=result)

@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    # Make predictions on the test set
    Y_pred = classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    return jsonify({'accuracy': accuracy})

if __name__ == '__main__':
    app.run(debug=True)
