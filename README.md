# Diabetes Prediction Model

Our project’s goal is to create a simple biomedical model of diabetes risk assessment. Diabetes is a widespread, chronic disease that affects about 12% of the US adult population. It may lead to serious consequences if not treated or detected early. The model will predict the likelihood of a patient having or developing the condition, based on the datasets found on heart disease, diabetes health indicators, glucose levels, etc. With this information, patients can initiate lifestyle changes to improve their conditions. With this model, we aim to be more proactive and preventative concerning the spread of diabetes. This project predicts whether a person is likely to have diabetes based on health-related attributes using a Random Forest Classifier.

---

## **Dataset**
The dataset used for training the model is bundled as a `.zip` file (`Group13_Data.zip`) in the repository. It includes health-related attributes and a binary target variable (`Diabetes_binary`) indicating the presence of diabetes. 

**Dataset Source**: (https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

---

## **How to Run the Code**

### **1. Prerequisites**
Ensure you have the following installed:
- Python (3.7 or later)
- `pip` (Python package manager)

### **2. Clone the Repository**
Clone the repository to your local environment or open it in a GitHub Codespace:
```bash
git clone https://github.com/jordynrichardson/diabetes-prediction.git
cd diabetes-prediction
```

### **3. Install Dependencies**
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### **4. Add the Dataset**
Ensure the dataset file (`Group13_Data.zip`) is present in the repository root. If it’s not already there, download it from the source and place it in the repository directory.

### **5. Run the Script**
Run the Python script to train the model, evaluate it, and start the prediction demo:
```bash
python diabetes_model.py
```

### **6. Follow the Interactive Demo**
The script includes a demo where you can input health-related values to get predictions:
- Enter the feature values when prompted.
- The script will output whether the prediction is `Diabetes` or `No Diabetes`.

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
4. **Interactive Demo**:
   - Users can input feature values to predict if diabetes is likely.
