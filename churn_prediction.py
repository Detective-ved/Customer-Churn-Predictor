# AI-Based Customer Churn Prediction System

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
data = pd.read_csv('customer_churn.csv')  # Make sure the CSV file is in the working directory

# Step 3: Explore Dataset
print(data.head())
print(data.info())
print(data.describe())
print(data['Churn'].value_counts())  # Check class distribution

# Step 4: Data Preprocessing
# Handle missing values
data = data.dropna()

# Encode categorical variables
le = LabelEncoder()
for column in data.select_dtypes(include='object').columns:
    if column != 'Churn':  # Skip target column for now
        data[column] = le.fit_transform(data[column])

# Encode target variable
data['Churn'] = data['Churn'].map({'No':0, 'Yes':1})

# Feature scaling
scaler = StandardScaler()
features = data.drop('Churn', axis=1)
features_scaled = scaler.fit_transform(features)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, data['Churn'], test_size=0.2, random_state=42)

# Step 6: Train Models
# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Step 7: Evaluate Models
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()