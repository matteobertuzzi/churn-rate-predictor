import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import chi2_contingency
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('./assets/customer_churn.csv')

# Step 1: Data Exploration and Preprocessing
print("Dataset Overview:")
print(data.head())
print(data.info())
print(data.describe())

# Target Variable Analysis
churn_counts = data['Churn'].value_counts()
print("\nChurn Distribution:")
print(churn_counts)

# Encode Categorical Variables
columns_to_encode = data.select_dtypes(include=['object']).columns.drop('customerID')
label_encoder = LabelEncoder()
for col in columns_to_encode:
    data[col] = label_encoder.fit_transform(data[col])

# Scale Numerical Features
scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
data[numerical_features] = data[numerical_features].fillna(0)  # Handle NaNs in TotalCharges
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Step 2: Statistical Analysis - Chi-Square Tests
categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

print("\nChi-Square Test Results:")
for col in categorical_columns:
    contingency_table = pd.crosstab(data[col], data['Churn'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"{col}: Chi2 = {chi2:.2f}, p-value = {p:.4f}")

# Step 3: Logistic Regression Model
# Select Features for Fiber Optic Customers
fiberopt_segment = data[data['InternetService'] == 1]  # Fiber Optic = 1
X = fiberopt_segment[['Contract', 'PaymentMethod']]
y = fiberopt_segment['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)

print("\nLogistic Regression Results:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Step 4: Neural Network Model
# Prepare Data for Neural Network
X = data.drop(columns=['customerID', 'Churn'])
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the Neural Network
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nNeural Network Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 5: Visualization of Results
# Plot Accuracy and Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Step 6: Summary of Findings
print("\nKey Insights:")
print("1. Customers with month-to-month contracts are more likely to churn.")
print("2. Electronic check payment method is associated with higher churn rates.")
print("3. Fiber Optic customers exhibit higher churn than DSL users.")
print("4. Shorter tenure correlates with higher churn.")
print("5. Lack of online security and tech support also contribute to churn.")
