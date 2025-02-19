import torch
import torch.nn as nn
import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# --- Data Loading and Preprocessing (REQUIRED: Identical to training!) ---
print("Current working directory:", os.getcwd())
file_pattern = "CC_2019_Sepsis/training_setA/*.psv"
files = glob.glob(file_pattern)
if not files:
    absolute_file_pattern = "C:/Users/bunju/OneDrive/Desktop/Project/CC-2019-Sepsis/CC-2019-Sepsis/training_setA/*.psv"
    files = glob.glob(absolute_file_pattern)

dfs = [pd.read_csv(file, sep='|') for file in files]
combined_df = pd.concat(dfs, ignore_index=True)

combined_df.drop(columns=["Bilirubin_direct", "Alkalinephos", "AST", "EtCO2", "TroponinI"], inplace=True)

for col in combined_df.columns:
    if pd.api.types.is_numeric_dtype(combined_df[col]):
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())

scaler = MinMaxScaler()
combined_df[combined_df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(combined_df.select_dtypes(include=['float64', 'int64']))

X = combined_df.drop('SepsisLabel', axis=1)
y = combined_df['SepsisLabel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=66, shuffle=True)

# ---OVERSAMPLING MUST BE INCLUDED SINCE IT SHAPES THE DATA---
count_1s = y_train[y_train == 1].shape[0]
count_0s = y_train[y_train == 0].shape[0]
target_count = count_0s
over = RandomOverSampler(sampling_strategy={1: target_count}, random_state=42)
X_train, y_train = over.fit_resample(X_train, y_train)
train_df = pd.concat([pd.DataFrame(X_train), pd.Series(y_train)], axis=1)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values


# --- Feature Selection ---
selected_features = ['HR', 'Temp', 'SBP', 'MAP', 'Resp', 'BaseExcess', 'pH', 'PaCO2', 'Potassium', 'Hct', 'Hgb', 'Platelets', 'Age', 'Gender', 'Unit1', 'HospAdmTime', 'ICULOS']

X_train = X_train[:, [list(X.columns).index(feature) for feature in selected_features]]
X_test = X_test[selected_features]

X_test = X_test.to_numpy().astype(np.float32)
y_test = y_test.to_numpy().astype(np.float32)

# --- PyTorch Model Definition (MUST MATCH TRAINING SCRIPT) ---
class PytorchModelBinary(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(PytorchModelBinary, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_layers[-1], 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- Model Loading and Evaluation ---
input_dim = len(selected_features)
hidden_layers = [256, 128, 64]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PytorchModelBinary(input_dim, hidden_layers)
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
model.eval()

X_test_tensor = torch.tensor(X_test).to(device)
y_test_tensor = torch.tensor(y_test).unsqueeze(1).to(device)

with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs > 0.5).float().cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Generate and print FULL Classification Report
report = classification_report(actual, predicted, zero_division=0, labels=[0.0, 1.0])
print("Classification Report:\n", report)