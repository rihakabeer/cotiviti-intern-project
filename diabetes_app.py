# Import Statements
import streamlit as st #Used to create the web interface for the app.
import pandas as pd #handles data manipulation and loading.
from PIL import Image
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score #Used for model evaluation
from sklearn.ensemble import RandomForestClassifier #manages complex data
from sklearn.model_selection import train_test_split, GridSearchCV #Handles machine learning model creation, hyperparameter tuning, and data splitting.
import matplotlib.pyplot as plt #Used for visualizations
import seaborn as sns #Used for visualizations
import numpy as np
import joblib  # For saving and loading models

# Load Data
df = pd.read_csv(r'C:\Users\Mohds\OneDrive\Desktop\diabetes_prediction-master\diabetes_prediction-master\diabetes.csv') #The df variable loads the diabetes dataset from a CSV file into a Pandas DataFrame.

# Headings
st.title('Diabetes Prediction App') #Display the title and a description of the app.
st.markdown("""
This app predicts the likelihood of diabetes based on a number of health factors. You can adjust the values on the sidebar and get a prediction instantly!
""")
st.subheader('Dataset Overview') #Display headings and an overview of the dataset
st.write(df.describe()) #which shows the summary statistics of the data.

# Feature Distribution Histograms
st.subheader('Feature Distribution')
fig, ax = plt.subplots(2, 4, figsize=(15, 8))
features = df.columns[:-1]
for i, feature in enumerate(features):
    row = i // 4
    col = i % 4
    ax[row, col].hist(df[feature], bins=20, edgecolor='k', color='skyblue')
    ax[row, col].set_title(feature)
plt.tight_layout()
st.pyplot(fig)

# Correlation Heatmap
st.subheader('Feature Correlation Heatmap') #This section creates histograms for each feature in the dataset.
correlation_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Prepare Data for Model
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Function to Get User Input
def user_report(): #This function creates sliders on the sidebar using st.sidebar.slider()
    st.sidebar.header('Input Your Health Parameters:') #allowing users to input their health parameters.
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    return pd.DataFrame(user_data, index=[0])


# Collect user input
user_data = user_report()

# Train Model with Hyperparameter Tuning
rf = RandomForestClassifier(random_state=42) #A machine learning algorithm that builds an ensemble of decision trees.
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None]
}
grid_search = GridSearchCV(rf, param_grid, cv=3) #It finds the best combination of hyperparameters by performing cross-validation on the training data.
grid_search.fit(x_train, y_train)

# Get Best Model
best_rf = grid_search.best_estimator_

# Feature Importance
st.subheader('Feature Importance') #The Random Forest model provides an importance score for each feature.
importances = best_rf.feature_importances_ #This section visualizes which features are most influential in the prediction (e.g., Glucose, BMI).
sorted_indices = np.argsort(importances)[::-1]
fig, ax = plt.subplots()
ax.barh(np.array(features)[sorted_indices], importances[sorted_indices], color='teal')
ax.set_title('Feature Importances')
st.pyplot(fig)

# Accuracy
st.subheader('Model Accuracy:')
accuracy = accuracy_score(y_test, best_rf.predict(x_test)) #he app calculates the accuracy of the model on the test set and displays it using accuracy_score().
st.write(f'{accuracy * 100:.2f}%') # Accuracy is the percentage of correct predictions on the test data.

# ROC Curve
st.subheader('ROC Curve')
y_prob = best_rf.predict_proba(x_test)[:, 1] #A graphical representation of the model's performance by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) at different thresholds.
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob) #Calculates the Area Under the Curve (AUC), which gives a measure of model performance; the higher the AUC, the better the model is at distinguishing between classes.
fig, ax = plt.subplots()
ax.plot(fpr, tpr, marker='.', color='darkorange', label=f'AUC = {auc_score:.2f}')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
st.pyplot(fig)

# Prediction
st.subheader('Prediction Based on Your Inputs')
user_result = best_rf.predict(user_data) #The best Random Forest model predicts whether the user is diabetic or healthy based on their input from the sidebar.
user_prob = best_rf.predict_proba(user_data)[0]
labels = ['Healthy', 'Diabetic'] #The prediction and corresponding probability are displayed as a message in the app.
result_text = f'You are likely to be **{labels[user_result[0]]}** with a probability of {user_prob[user_result[0]] * 100:.2f}%.'
st.write(result_text)

# Model Saving Option
if st.sidebar.button('Save Model'): #If the user clicks the "Save Model" button, the best trained model is saved to a file using Joblib
    joblib.dump(best_rf, 'diabetes_model.pkl') #This allows you to reuse the model later without retraining it.
    st.sidebar.write("Model saved successfully!")
