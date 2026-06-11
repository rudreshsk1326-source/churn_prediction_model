import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the dataset
data = pd.read_csv("Churn_Modelling.csv")

# Drop unnecessary columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Separate features and target
X = data.drop('Exited', axis=1)
y = data['Exited']

# Encode Gender
le_gender = LabelEncoder()
X['Gender'] = le_gender.fit_transform(X['Gender'])

# One-hot encode Geography
X = pd.get_dummies(X, columns=['Geography'], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the preprocessors
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder_gender.pkl', 'wb') as f:
    pickle.dump(le_gender, f)

print("Data preprocessing completed successfully!")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
