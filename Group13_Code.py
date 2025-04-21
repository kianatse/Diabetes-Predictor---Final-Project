import pandas as pd
import zipfile
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

# Function to extract a .zip file and return the path to the extracted .csv file
def extract_zip(zip_path, extract_to="extracted_data"):
    """
    Extracts a .zip file and returns the path to the first .csv file found.
    
    Parameters:
        zip_path (str): Path to the .zip file.
        extract_to (str): Directory to extract the files to.
    
    Returns:
        str: Path to the extracted .csv file.
    """
    print(f"Extracting {zip_path}...")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"The file {zip_path} is not a valid .zip file.")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
        print(f"Extracted files to {extract_to}.")
    
    # Find the first .csv file in the extracted directory
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                print(f"Found CSV file: {csv_path}")
                return csv_path
    
    raise FileNotFoundError("No CSV file found in the extracted .zip archive.")

# Function to load the dataset from a .csv file
def load_data(file_path):
    """
    Loads the dataset from a .csv file.
    
    Parameters:
        file_path (str): Path to the .csv file.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    print(f"Loading dataset from {file_path}...")
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    print(f"Columns in dataset: {list(data.columns)}")
    return data

# Function to preprocess the dataset
def preprocess_data(data):
    """
    Preprocesses the dataset for model training.
    
    Parameters:
        data (pd.DataFrame): The dataset.
    
    Returns:
        tuple: Features (X) and target (y) for training.
    """
    # Define features and target variable
    features = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education'
    ]
    target = 'Diabetes_012'

    # Check for missing columns
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"Warning: The following features are missing and will be ignored: {missing_features}")
        features = [f for f in features if f in data.columns]

    if target not in data.columns:
        raise KeyError(f"The target column '{target}' is missing in the dataset.")

    # Select features and target
    X = data[features]
    y = data[target]

    print(f"Preprocessed dataset with {len(features)} features.")
    return X, y

# Function to train and evaluate the model
def train_model(X, y, model_path="diabetes_model.pkl"):
    """
    Trains and evaluates a Random Forest Classifier and saves the model to a file.
    
    Parameters:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        model_path (str): Path to save the trained model.
    
    Returns:
        RandomForestClassifier: The trained model.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training the Random Forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save the model to a file
    joblib.dump(model, model_path)
    print(f"Trained model saved to {model_path}.")

    return model

# Function to load a saved model
def load_model(model_path):
    """
    Loads a trained model from a file.
    
    Parameters:
        model_path (str): Path to the saved model file.
    
    Returns:
        RandomForestClassifier: The loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file '{model_path}' does not exist.")
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    return model

# Function to run a demo for user interaction
def run_demo(model, features):
    """
    Runs a demo that allows users to input feature values and get predictions.
    
    Parameters:
        model (RandomForestClassifier): The trained model.
        features (list): List of feature names.
    """
    print("\n--- Diabetes Prediction Demo ---")
    print("Enter the following feature values:")
    user_input = {}
    for feature in features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                user_input[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    # Convert user input to a DataFrame
    input_df = pd.DataFrame([user_input])

    # Make prediction
    prediction = model.predict(input_df)[0]
    print(f"\nPrediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}")

if __name__ == "__main__":
    # Suppress warnings for better readability
    warnings.filterwarnings("ignore")

    # Define the path to the .zip file and the model file
    zip_file_path = "Group13_Data.zip"
    model_file_path = "diabetes_model.pkl"
    
    try:
        # Extract and load the dataset
        csv_file_path = extract_zip(zip_file_path)
        data = load_data(csv_file_path)

        # Preprocess the data
        X, y = preprocess_data(data)

        # Train the model if it doesn't already exist
        if not os.path.exists(model_file_path):
            trained_model = train_model(X, y, model_file_path)
        else:
            trained_model = load_model(model_file_path)

        # Run the demo
        feature_list = X.columns.tolist()
        run_demo(trained_model, feature_list)

    except Exception as e:
        print(f"An error occurred: {e}")
