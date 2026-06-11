<<<<<<< HEAD
# Customer Churn Prediction using ANN

A deep learning project that predicts whether a bank customer will churn (leave) using an Artificial Neural Network (ANN), with an interactive Streamlit web app for real-time predictions.

## Project Overview

This project uses the [Churn Modelling dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) to train a binary classification ANN model. It also includes a salary regression model trained on the same dataset.

## Project Structure

```
annclassification/
├── app.py                        # Streamlit web application
├── experiment.py                 # Data preprocessing script
├── experiments.ipynb             # Model training notebook
├── prediction.ipynb              # Inference/prediction notebook
├── hyperparametertuningann.ipynb # Hyperparameter tuning with Keras Tuner
├── salaryregression.ipynb        # ANN regression for salary prediction
├── Churn_Modelling.csv           # Dataset
├── model.h5                      # Trained classification model
├── regression_model.h5           # Trained regression model
├── scaler.pkl                    # Fitted StandardScaler
├── label_encoder_gender.pkl      # Fitted LabelEncoder for Gender
├── onehot_encoder_geo.pkl        # Fitted OneHotEncoder for Geography
└── requirements.txt
```

## Features Used

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Has credit card (0/1) |
| IsActiveMember | Active member status (0/1) |
| EstimatedSalary | Estimated annual salary |

**Target:** `Exited` — 1 if the customer churned, 0 otherwise.

## Setup

### Prerequisites
- Python 3.8+

### Installation

```bash
git clone https://github.com/your-username/annclassification.git
cd annclassification
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser, fill in the customer details, and click **Predict** to get the churn probability.

### Run Prediction Notebook

Open `prediction.ipynb` in Jupyter and run all cells to see a sample prediction walkthrough.

### Retrain the Model

1. Preprocess data:
   ```bash
   python experiment.py
   ```
2. Open and run `experiments.ipynb` to train and save the model.

## Model Details

- **Architecture:** ANN with fully connected layers, ReLU activations, and a sigmoid output
- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Preprocessing:** StandardScaler for numerical features, LabelEncoder for Gender, OneHotEncoder for Geography

## Tech Stack

- TensorFlow / Keras
- Scikit-learn 
- Pandas, NumPy
- Streamlit
- Matplotlib
=======
---
title: Churn Prediction Model
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: 'Customer churn prediction web app using an ANN trained '
license: mit
---

# Welcome to Streamlit!

Edit `/src/streamlit_app.py` to customize this app to your heart's desire. :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).
>>>>>>> ec06781699bcc893c46ac47ac88f893b97462bc3
