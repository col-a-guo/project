import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score  # Import f1_score for calculating F1 outside of classification report
from collections import Counter

# --- Data Loading and Preprocessing (same as your previous code) ---
print("Current working directory:", os.getcwd())
# Define the path pattern to locate .psv files in the folder
file_pattern = "CC_2019_Sepsis/training_setA/*.psv"
files = glob.glob(file_pattern)
print("Files found:", files)
if not files:
    absolute_file_pattern = "C:/Users/bunju/OneDrive/Desktop/Project/CC-2019-Sepsis/CC-2019-Sepsis/training_setA/*.psv"
    files = glob.glob(absolute_file_pattern)
    print("Files found (absolute path):", files)
# Option 1: Read all files into a list of DataFrames
dfs = [pd.read_csv(file, sep='|') for file in files]

# Combine all DataFrames into a single one (if desired)
combined_df = pd.concat(dfs, ignore_index=True)


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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=66, shuffle=True)  # Explicitly set shuffle to True



# --- Oversampling/Undersampling to Geometric Mean ---

# Calculate the number of samples for each class
count_1s = y_train[y_train == 1].shape[0]
count_0s = y_train[y_train == 0].shape[0]

# Target number of samples (equal to the majority class count)
target_count = count_0s

# Initialize RandomOverSampler
over = RandomOverSampler(sampling_strategy={1: target_count}, random_state=42)

# Apply oversampling
X_train, y_train = over.fit_resample(X_train, y_train)

print(f"Resampled class distribution: {Counter(y_train)}")

# **SHUFFLE X_train and y_train *AFTER* RESAMPLING WHILE PRESERVING PAIRINGS**
# Concatenate X_train and y_train horizontally
train_df = pd.concat([pd.DataFrame(X_train), pd.Series(y_train)], axis=1)

# Shuffle the combined DataFrame
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split back into X_train and y_train
X_train = train_df.iloc[:, :-1].values  # All columns except the last one
y_train = train_df.iloc[:, -1].values   # The last column


# --- Add Print After Resampling ---
print("\nAfter RandomOverSampler and Shuffle:")
print("First 20 y_train labels (after resample):", y_train[:20])
print("Last 20 y_train labels (after resample):", y_train[-20:])


# --- Feature Selection ---
selected_features = ['HR', 'Temp', 'SBP', 'MAP', 'Resp', 'BaseExcess', 'pH', 'PaCO2', 'Potassium', 'Hct', 'Hgb', 'Platelets', 'Age', 'Gender', 'Unit1', 'HospAdmTime', 'ICULOS']

X_train = X_train[:, [list(X.columns).index(feature) for feature in selected_features]] #Added back the indices
X_test = X_test[selected_features]

print("Selected Training Features shape:\n", X_train.shape)
print("Selected Testing Features shape:\n", X_test.shape)


# --- NumPy Array Conversion (Fixed types, NO one-hot encoding) ---

X_train = X_train.astype(np.float32)
X_test = X_test.to_numpy().astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.to_numpy().astype(np.float32)


# --- PyTorch Model Definition ---

class PytorchModelBinary(nn.Module):  # Renamed to indicate binary classification
    def __init__(self, input_dim, hidden_layers):
        super(PytorchModelBinary, self).__init__()
        self.layers = nn.ModuleList()  # Use ModuleList to store layers

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# --- Model Configuration ---
hidden_layers = [256, 128, 64]
learning_rate = 0.000005  #Fixed learning rate
print(f"Training with neurons: {hidden_layers}, learning rate: {learning_rate}")

# Model Instantiation
input_dim = X_train.shape[1]
model = PytorchModelBinary(input_dim, hidden_layers)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Fixed learning rate

# Define Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Loss function  (Cross-Entropy is still needed for training)
criterion = nn.BCELoss()

# Data to tensors and move to device
X_train_tensor = torch.tensor(X_train).to(device)
y_train_tensor = torch.tensor(y_train).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test).to(device)
y_test_tensor = torch.tensor(y_test).unsqueeze(1).to(device)


# --- Training loop with Early Stopping ---
epochs = 1000  # Set maximum number of epochs
batch_size = 32
patience = 100  # Number of epochs to wait for improvement
best_macro_f1 = 0.0
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation and Early Stopping
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation during evaluation
        test_outputs = model(X_test_tensor)
        predicted_probs = test_outputs.cpu().numpy() #Move to CPU for calculating F1
        predicted = (predicted_probs > 0.5).astype(int) #Convert probs to binary predictions
        actual = y_test_tensor.cpu().numpy().astype(int)

    # Calculate Macro F1 Score
    macro_f1 = f1_score(actual, predicted, average='macro') #Use the actual binary values for F1 score
    if epoch%10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Macro-average F1: {macro_f1:.4f}, current best F1: {best_macro_f1:.4f}')

    # Early stopping check
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth') # Save the best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            print(f"Best Macro-average F1 score: {best_macro_f1:.4f}")
            break

# --- Load Best Model and Evaluate ---
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs > 0.5).float().cpu().numpy()  # Move to CPU if using GPU
    actual = y_test_tensor.cpu().numpy()

report = classification_report(actual, predicted, output_dict=True, zero_division=0, labels=[0.0, 1.0])  # added zero_division, set labels


print(f"Learning Rate: {learning_rate}") #Added to report
print(f"Best Macro F1: {report['macro avg']['f1-score']:.4f}")
print(f"F1-score (0.0): {report['0.0']['f1-score']:.4f}")
print(f"F1-score (1.0): {report['1.0']['f1-score']:.4f}")
print(f"Accuracy: {report['accuracy']:.4f}")