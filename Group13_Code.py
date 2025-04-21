import zipfile
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


def extract_dataset(zip_file, extract_to="data"):
    """
    Extracts the dataset from a zip file.

    Parameters:
        zip_file (str): Path to the zip file.
        extract_to (str): Directory to extract files to.

    Returns:
        str: Path to the extracted dataset file.
    """
    print(f"Extracting {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(extract_to)
    print(f"Extracted to {extract_to}.")
    
    # Find the first CSV file in the extracted directory
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            if file.endswith(".csv"):
                return os.path.join(root, file)
    raise FileNotFoundError("No CSV file found in the zip archive.")


def load_data(zip_file, extract_to="data"):
    """
    Loads the dataset from a zip file.

    Parameters:
        zip_file (str): Path to the zip file.
        extract_to (str): Directory to extract files to.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """
    dataset_path = extract_dataset(zip_file, extract_to)
    print(f"Loading dataset from {dataset_path}...")
    data = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
    return data


def preprocess_data(data):
    """
    Preprocesses the dataset for training.

    Parameters:
        data (pd.DataFrame): The raw dataset.

    Returns:
        pd.DataFrame, pd.Series: Features (X) and target (y).
    """
    # Handle missing values (example: replace 0s with NaN for certain columns)
    columns_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    data[columns_to_replace] = data[columns_to_replace].replace(0, np.nan)
    
    # Fill missing values with the median
    data[columns_to_replace] = data[columns_to_replace].fillna(data[columns_to_replace].median())
    
    # Separate features and target
    X = data.drop("target", axis=1)  # Features
    y = data["target"]  # Target variable
    return X, y


def train_model(X, y):
    """
    Trains a Random Forest model.

    Parameters:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.

    Returns:
        model: Trained model.
        dict: Evaluation metrics.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Return the model and evaluation metrics
    metrics = {
        "accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return model, metrics


def predict_risk(model, patient_data):
    """
    Predicts diabetes risk for a new patient.

    Parameters:
        model: Trained model.
        patient_data (dict): Patient features.

    Returns:
        str: Risk assessment.
    """
    # Convert patient data to DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # Predict the risk
    prediction = model.predict(patient_df)
    risk = "High Risk" if prediction[0] == 1 else "Low Risk"
    return risk


if __name__ == "__main__":
    # Path to the zip file
    zip_file = "Group13_Data.zip"

    # Load and preprocess the dataset
    data = load_data(zip_file)
    X, y = preprocess_data(data)

    # Train the model
    model, metrics = train_model(X, y)

    # Example: Predict diabetes risk for a new patient
    patient_data = {
        "Glucose": 120,
        "BloodPressure": 80,
        "SkinThickness": 35,
        "Insulin": 90,
        "BMI": 30.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 45,
    }

    risk = predict_risk(model, patient_data)
    print(f"\nDiabetes Risk Assessment for Patient: {risk}")
