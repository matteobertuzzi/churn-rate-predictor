# Telco Customer Churn Prediction

This project analyzes the Telco Customer Churn dataset to identify the key factors contributing to customer churn and builds predictive models, including a neural network, to predict customer churn.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
3. [Project Workflow](#project-workflow)
4. [Technologies Used](#technologies-used)
5. [Key Findings](#key-findings)
6. [How to Run the Project](#how-to-run-the-project)
7. [Results](#results)
8. [License](#license)

---

## Introduction

Customer churn is a critical issue for any business, including the telecom industry. Retaining existing customers is more cost-effective than acquiring new ones, therefore the need to minimize churn rates. This project explores customer churn data to:

- Identify patterns and factors affecting churn.
- Build predictive models to classify whether a customer will churn.

---

## Dataset Overview

The dataset used in this project is the Telco Customer Churn dataset, which contains information about:

- Customer demographics.
- Services subscribed.
- Contract and payment information.

### Column Descriptions:

- **customerID**: Unique identifier for each customer.
- **gender**: Gender of the customer (Male/Female).
- **SeniorCitizen**: Whether the customer is a senior citizen (1/0).
- **Partner**: Whether the customer has a partner (Yes/No).
- **Dependents**: Whether the customer has dependents (Yes/No).
- **tenure**: Duration of the customer relationship (in months).
- **PhoneService**: Whether the customer has phone service (Yes/No).
- **MultipleLines**, **InternetService**, **OnlineSecurity**, **OnlineBackup**, **DeviceProtection**, **TechSupport**, **StreamingTV**, **StreamingMovies**: Subscription to various services.
- **Contract**: Customer's contract type (Month-to-month/One year/Two year).
- **PaperlessBilling**: Whether the customer uses paperless billing (Yes/No).
- **PaymentMethod**: Customer's payment method.
- **MonthlyCharges**: Monthly amount charged.
- **TotalCharges**: Total amount charged.
- **Churn**: Whether the customer churned (Yes/No).

---

## Project Workflow

1. **Data Preprocessing**:
   - Handle missing values and outliers.
   - Encode categorical variables.
   - Standardize numerical features.

2. **Exploratory Data Analysis (EDA)**:
   - Analyze churn rates across different demographics, services, and payment methods.
   - Perform chi-square tests to identify significant relationships.

3. **Modeling**:
   - Build and evaluate logistic regression and neural network models.
   - Optimize the neural network architecture.

4. **Visualization**:
   - Use Seaborn and Matplotlib to create visual insights.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, SciPy, Scikit-learn, TensorFlow/Keras
- **Neural Network**: Built with TensorFlow's API
- **Statistical Tests**: Chi-Square tests

---

## Key Findings

1. **Contract Type**: Customers with month-to-month contracts have the highest churn rates compared to those with one-year or two-year contracts.
2. **Payment Method**: Customers paying via electronic checks churn more frequently.
3. **Internet Service**: Customers with fiber optic internet are more likely to churn than those with DSL or no internet service.
4. **Tenure**: Longer-tenured customers are less likely to churn.
5. **Demographics**: Senior citizens and customers without dependents have higher churn rates.

---

## Results

- **Logistic Regression**:
  - Achieved an accuracy of ~78% on predicting churn.

- **Neural Network**:
  - Achieved a test accuracy of ~82%.
  - Demonstrated better handling of complex relationships in the data compared to logistic regression.

---

## Visualizations

Key visualizations include:

- Churn rates across demographics (gender, senior citizen status).
- Churn rates for various services (InternetService, TechSupport, StreamingTV).
- Churn based on payment methods and contract types.

