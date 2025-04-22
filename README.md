# Diabetes Prediction Model

This project predicts whether a person is likely to have diabetes based on health-related attributes using a Random Forest Classifier.

---

## **Dataset**
The dataset used for training the model is bundled as a `.zip` file (`Group13_Data.zip`) in the repository. It includes health-related attributes and a binary target variable (`Diabetes_binary`) indicating the presence of diabetes.

**Dataset Source**: [Placeholder for Dataset URL or Description]

---

## **Model**
The trained model is saved as a compressed file: `diabetes_model.pkl.gz`. This file will be generated when the script is run, or you can download a pre-trained version if provided.

---

## **How to Run the Code**

### **1. Prerequisites**
Ensure you have the following installed:
- Python (3.7 or later)
- `pip` (Python package manager)

---

### **2. Clone the Repository**
Follow these steps to clone the repository to your local environment:

1. Open your terminal or command prompt.
2. Run the following command:
   ```bash
   git clone https://github.com/jordynrichardson/Diabetes-Predictor---Final-Project.git
   ```
3. Navigate into the cloned repository:
   ```bash
   cd Diabetes-Predictor---Final-Project
   ```

---

### **3. Ensure the Dataset is Present in the Repository Root**
1. Check if the dataset (`Group13_Data.zip`) is in the repository root:
   ```bash
   ls
   ```
   - If you see `Group13_Data.zip` listed, the dataset is already in the root directory.

2. If the dataset is missing:
   - Download the dataset from the source ([Placeholder for Dataset URL or Description]).
   - Move the file to the repository root:
     ```bash
     mv /path/to/Group13_Data.zip /path/to/Diabetes-Predictor---Final-Project/
     ```

3. Verify the dataset is now in the root directory:
   ```bash
   ls
   ```

---

### **4. Install Dependencies**
Install the required Python packages:
```bash
pip install -r requirements.txt
```

---

### **5. Run the Script**
Run the Python script to train the model, evaluate it, and start the prediction demo:
```bash
python Group13_Code.py
```

---

### **6. Follow the Interactive Demo**
The script includes a demo where you can input health-related values to get predictions:
- Enter the feature values when prompted.
- The script will output whether the prediction is `Diabetes` or `No Diabetes`.

---

### **7. Check the Generated Model File**
After running the script, the trained model will be saved as `diabetes_model.pkl.gz` in the repository directory. You can use this file for inference in future runs without retraining.

---

## **Model Details**
- **Model Type**: Random Forest Classifier
- **Number of Trees**: 50
- **Features**:
  - HighBP
  - HighChol
  - CholCheck
  - BMI
  - Smoker
  - Stroke
  - HeartDiseaseorAttack
  - PhysActivity
  - Fruits
  - Veggies
  - HvyAlcoholConsump
  - AnyHealthcare
  - NoDocbcCost
  - GenHlth
  - MentHlth
  - PhysHlth
  - DiffWalk
  - Sex
  - Age
  - Education
- **Target Variable**: `Diabetes_binary`

---

## **How the Code Works**
1. **Dataset Extraction**:
   - A `.zip` file is extracted to retrieve the dataset.
2. **Data Preprocessing**:
   - Features and the target variable are selected, and optional feature selection is applied.
3. **Model Training**:
   - A Random Forest Classifier is trained on the dataset.
4. **Model Saving**:
   - The trained model is saved as a compressed `.pkl.gz` file.
5. **Interactive Demo**:
   - Users can input feature values to predict if diabetes is likely.

