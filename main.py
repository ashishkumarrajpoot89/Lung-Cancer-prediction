import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB



#st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Lung Cancer Prediction App")


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    lung_data = pd.read_csv(uploaded_file)

    st.write("### Dataset Overview")
    st.dataframe(lung_data.head())

    # Display data information
    st.write("### Data Information")
    st.write("Shape of Dataset:", lung_data.shape)
    st.write("Data Types:")
    st.write(lung_data.dtypes)
    st.write("Null Values:")
    st.write(lung_data.isnull().sum())

    # Mapping categorical values
    lung_data.GENDER = lung_data.GENDER.map({"M": 1, "F": 2})
    lung_data.LUNG_CANCER = lung_data.LUNG_CANCER.map({"YES": 1, "NO": 2})

    # Data Filtering
    st.write("### Filter Dataset")
    gender_filter = st.selectbox("Select Gender", options=["All", "Male", "Female"])
    if gender_filter == "Male":
        lung_data = lung_data[lung_data.GENDER == 1]
    elif gender_filter == "Female":
        lung_data = lung_data[lung_data.GENDER == 2]

    age_filter = st.slider("Select Age Range", int(lung_data.AGE.min()), int(lung_data.AGE.max()), (int(lung_data.AGE.min()), int(lung_data.AGE.max())))
    lung_data = lung_data[(lung_data.AGE >= age_filter[0]) & (lung_data.AGE <= age_filter[1])]

    st.write("Filtered Dataset:")
    st.dataframe(lung_data)

    # Data Splitting
    x = lung_data.iloc[:, :-1]
    y = lung_data.iloc[:, -1]
    test_size = st.slider("Select Test Size", 0.1, 0.5, 0.33)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)

    st.write("### Choose Classifier for Prediction")
    classifier_name = st.selectbox("Select Classifier", ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM", "Naive Bayes"])

    if classifier_name == "Logistic Regression":
        model = LogisticRegression()
    elif classifier_name == "KNN":
        n_neighbors = st.slider("Number of Neighbors (K)", 1, 15, 3)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif classifier_name == "Decision Tree":
        max_depth = st.slider("Maximum Depth", 1, 20, 5)
        model = DecisionTreeClassifier(random_state=0, criterion="entropy", max_depth=max_depth)
    elif classifier_name == "Random Forest":
        n_estimators = st.slider("Number of Trees", 50, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif classifier_name == "SVM":
        c_value = st.slider("C Value", 0.1, 10.0, 1.0)
        model = OneVsRestClassifier(BaggingClassifier(SVC(C=c_value, kernel='rbf', random_state=9, probability=True), n_jobs=-1))
    elif classifier_name == "Naive Bayes":
        model = GaussianNB()

    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction, average='binary')
    recall = recall_score(y_test, prediction, average='binary')
    f1 = f1_score(y_test, prediction, average='binary')

    st.write("### Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, prediction)
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["No Cancer", "Cancer"], yticklabels=["No Cancer", "Cancer"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    st.pyplot(fig)

    # Correlation Heatmap
    if st.checkbox("Show Correlation Heatmap"):
        cn = lung_data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cn, cmap="Blues", annot=True, square=True)
        st.pyplot(fig)

    # Histogram Plot
    if st.checkbox("Show Feature Histograms"):
        num_list = list(lung_data.columns)
        fig, axes = plt.subplots(len(num_list), 1, figsize=(10, len(num_list) * 3))
        for i, feature in enumerate(num_list):
            axes[i].hist(lung_data[feature], color='blue', alpha=0.5)
            axes[i].set_title(feature)
            axes[i].set_xticks([])
        st.pyplot(fig)
else:
    st.write("Please upload a CSV file to continue.")


