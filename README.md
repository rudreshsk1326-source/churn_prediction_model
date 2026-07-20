# Customer Churn Prediction using ANN

A deep learning project that predicts whether a bank customer will churn using an Artificial Neural Network (ANN), with an interactive Streamlit web app.

## Project Structure

```
annclassification/
├── app.py                        # Streamlit web application
├── experiments.ipynb             # Data preprocessing & model training
├── hyperparametertuningann.ipynb # Hyperparameter tuning experiments
├── prediction.ipynb              # Prediction experiments
├── salaryregression.ipynb        # Salary regression experiments
├── Churn_Modelling.csv           # Dataset (10,000 records)
├── model.h5                      # Trained ANN model
├── scaler.pkl                    # Fitted StandardScaler
├── label_encoder_gender.pkl      # Fitted LabelEncoder for Gender
├── onehot_encoder_geo.pkl        # Fitted OneHotEncoder for Geography
└── requirements.txt              # Python dependencies
```

## Model Architecture

- Input: 12 features
- Hidden Layer 1: 64 neurons, ReLU
- Hidden Layer 2: 32 neurons, ReLU
- Output: 1 neuron, Sigmoid (binary classification)
- Optimizer: Adam (lr=0.01)
- Loss: Binary Crossentropy

## Features Used

| Feature | Description |
|---|---|
| CreditScore | Customer credit score |
| Geography | France / Germany / Spain (one-hot encoded) |
| Gender | Male / Female (label encoded) |
| Age | Customer age (18–92) |
| Tenure | Years with the bank (0–10) |
| Balance | Account balance |
| NumOfProducts | Number of bank products (1–4) |
| HasCrCard | Has credit card (0/1) |
| IsActiveMember | Is active member (0/1) |
| EstimatedSalary | Estimated annual salary |

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Navigate to project folder
cd annclassification

# Run the app
streamlit run app.py
```

## Usage

1. Fill in the customer details in the sidebar inputs.
2. Click **Predict**.
3. The app displays the churn probability and a verdict.

> Probability > 0.5 → Customer is likely to churn.
