# Lung Cancer Prediction

## Project Overview
This project aims to predict the likelihood of lung cancer in patients based on various health metrics and lifestyle factors. The model is trained using machine learning techniques and deployed through a web application.

## Dataset
Download the dataset from Kaggle: [Lung Cancer Dataset](https://www.kaggle.com/datasets/iamtanmayshukla/lung-cancer-data?resource=download)

## Project Structure

├── processor.py         # Data preprocessing and model training script
├── main.py             # Web application for prediction using Streamlit
├── lung_cancer_model.pkl  # Trained ML model
├── scaler.pkl          # Scaler for feature standardization
├── lung cancer data.csv # Dataset (to be downloaded)
└── README.md           # Project documentation


## Installation
1. Clone the repository:
   sh
   git clone https://github.com/your-repo/lung-cancer-prediction.git
   
2. Install required dependencies:
   sh
   pip install -r requirements.txt
   
3. Download the dataset and place it in the project folder.

## Data Preprocessing
The processor.py script handles:
- Loading the dataset (lung cancer data.csv).
- Encoding categorical variables (GENDER, SMOKING, COUGHING, FATIGUE, SHORTNESS_BREATH, LUNG_CANCER).
- Splitting data into training and testing sets.
- Standardizing numerical features.
- Training a *Random Forest Classifier*.
- Evaluating the model using accuracy and classification metrics.
- Saving the trained model (lung_cancer_model.pkl) and scaler (scaler.pkl).

Run preprocessing and model training:
sh
python processor.py


## Model Training and Evaluation
The script evaluates the model using:
- *Accuracy*
- *Precision, Recall, F1-Score*
- *Confusion Matrix*

## Web Application
A *Streamlit*-based web app (main.py) allows users to:
- Upload a CSV file.
- View dataset overview (shape, data types, missing values).
- Filter data by gender and age.
- Select a machine learning model (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes).
- View model evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Visualize correlation heatmap and feature distributions.

Run the web app:
sh
streamlit run main.py


## Machine Learning Models
The project supports the following classifiers:
- *Logistic Regression*
- *Decision Tree*
- *Random Forest*
- *Support Vector Machine (SVM)*
- *K-Nearest Neighbors (KNN)*
- *Naive Bayes*

## Results and Insights
1. *Smoking* is strongly correlated with lung cancer.
2. *Age* is a significant predictor; older individuals have a higher risk.
3. *Random Forest* performs best due to its ability to capture complex relationships.

## Future Improvements
- Implement additional ML models like *XGBoost*.
- Deploy the model as an API using *Flask/FastAPI*.
- Optimize model hyperparameters using *GridSearchCV*.
