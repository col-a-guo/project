import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- Data Loading and Preprocessing (same as your original code) ---
print("Current working directory:", os.getcwd())
# Define the path pattern to locate .psv files in the folder
file_pattern = "CC_2019_Sepsis/training_setC/*.psv"
files = glob.glob(file_pattern)
print("Files found:", files)
if not files:
    absolute_file_pattern = "C:/Users/r2d2go/Desktop/project/CC-2019-Sepsis/training_setC/*.psv"
    files = glob.glob(absolute_file_pattern)
    print("Files found (absolute path):", files)
# Option 1: Read all files into a list of DataFrames
dfs = [pd.read_csv(file, sep='|') for file in files]

# Combine all DataFrames into a single one (if desired)
combined_df = pd.concat(dfs, ignore_index=True)

###TODO: MAKE THIS NOT EAT A TINY LITTLE BIT OF OUR DATA
combined_df = combined_df[:10000]

# Calculate number of NaNs per column
nan_counts = combined_df.isna().sum()

# Calculate percentage of NaNs per column
nan_percentage = (nan_counts / len(combined_df)) * 100

# Create a DataFrame to display both counts and percentages
nan_df = pd.DataFrame({
    'num_nan': nan_counts,
    'percent_nan': nan_percentage
}).sort_values('percent_nan', ascending=True)

print(nan_df)
total_rows = combined_df.shape[0]
print("Total rows:", total_rows)

#Bilirubin_direct    789033
#Alkalinephos        778683
#AST                 778395

combined_df.drop(columns=["Bilirubin_direct", "Alkalinephos", "AST", "EtCO2", "TroponinI"], inplace=True)

# Verify the columns were dropped by printing the DataFrame's shape and column names
print("New shape:", combined_df.shape)
print("Remaining columns:", combined_df.columns)

for col in combined_df.columns:
    if pd.api.types.is_numeric_dtype(combined_df[col]):
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())

print(combined_df)

#used minmax scaler check with Collin
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
combined_df[combined_df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(combined_df.select_dtypes(include=['float64', 'int64']))

print(combined_df)

from sklearn.model_selection import train_test_split
X = combined_df.drop('SepsisLabel', axis=1)
y = combined_df['SepsisLabel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=66)

print("Training Features:\n", X_train)
print("Testing Features:\n", X_test)
print("Training Labels:\n", y_train)
print("Testing Labels:\n", y_test)


X_train = X_train.to_numpy().astype(np.float32)  # Ensure X_train is a NumPy array of float32
X_test = X_test.to_numpy().astype(np.float32)
y_train = y_train.to_numpy().astype(np.int64)
y_test = y_test.to_numpy().astype(np.int64)

# --- PyTorch Model Definition ---

class PytorchModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PytorchModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 25)
        self.layer3 = nn.Linear(25, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.softmax(self.layer3(x))
        return x

# --- Model Instantiation and Training ---

input_dim = X_train.shape[1]  # Dynamically determine input dimension
output_dim =  y_train.shape
model = PytorchModel(input_dim, output_dim)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Applies softmax internally
optimizer = optim.Adam(model.parameters())

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)  

# Training loop
epochs = 100
batch_size = 32  # Add a batch size

for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        # Get batch
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print (f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# --- Evaluation (Example) ---
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation during evaluation
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)  # Get predicted class labels

    # Calculate accuracy
    correct = (predicted == torch.argmax(y_test_tensor, dim=1)).sum().item()
    total = y_test_tensor.size(0)
    accuracy = correct / total
    print(f'Accuracy on the test set: {accuracy:.4f}')