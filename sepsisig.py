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
from imblearn.over_sampling import SMOTE  # Import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# --- Data Loading and Preprocessing (same as your previous code) ---
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
X_train_original, X_test, y_train_original, y_test = train_test_split(X, y, test_size=0.3, random_state=66)


# --- Oversampling with SMOTE (ONLY on the training set) ---
smote = SMOTE(random_state=42)  # Instantiate SMOTE
X_train_original, y_train_original = smote.fit_resample(X_train_original, y_train_original)

# Make a copy of the dataframes since feature selection happens INPLACE
X_train = X_train_original.copy()
y_train = y_train_original.copy()

X_test = X_test.copy() #No oversampling, but copy

print("Oversampled Training Features shape:\n", X_train.shape)
print("Oversampled Training Labels shape:\n", y_train.shape)


# --- Feature Selection Loop ---

original_columns = X_train.columns.tolist()
all_results = []

while len(X_train.columns) > 10:
    print(f"\n--- Feature Selection Iteration: {len(X_train.columns)} features ---")

    # 1. Logistic Regression for Feature Importance
    logistic = LogisticRegression(solver='liblinear', random_state=42)
    logistic.fit(X_train, y_train)

    # 2. Determine Threshold for Feature Removal
    num_features = X_train.shape[1]
    num_to_remove = max(1, int(np.floor(0.15 * num_features))) # Ensure at least one feature is removed
    print(f"Removing {num_to_remove} features")

    # Get feature importances and their corresponding column names
    feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.abs(logistic.coef_[0])}) #Absolute values of coefficients
    feature_importances = feature_importances.sort_values('importance', ascending=True)

    # Identify the least important features to remove
    features_to_remove = feature_importances['feature'].head(num_to_remove).tolist()

    print("Features to remove:", features_to_remove)

    # 3. Remove Least Important Features
    X_train.drop(columns=features_to_remove, inplace=True)
    X_test.drop(columns=features_to_remove, inplace=True)

    print("Remaining features:", X_train.columns.tolist())

    # --- NumPy Array Conversion (Fixed types, NO one-hot encoding) ---

    X_train_np = X_train.to_numpy().astype(np.float32)
    X_test_np = X_test.to_numpy().astype(np.float32)
    y_train_np = y_train.to_numpy().astype(np.float32)
    y_test_np = y_test.to_numpy().astype(np.float32)

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


    # --- Simplified Configuration ---
    hidden_layers = [64, 32]
    optimizer_name = 'Adam'
    print(f"Training with neurons: {hidden_layers}, optimizer: {optimizer_name}")

    # Model Instantiation
    input_dim = X_train_np.shape[1] #X_train is now a dataframe, but X_train_np is a numpy array
    model = PytorchModelBinary(input_dim, hidden_layers)

    # Optimizer
    optimizer = optim.Adam(model.parameters())


    # Loss function
    criterion = nn.BCELoss()

    # Data to tensors
    X_train_tensor = torch.tensor(X_train_np)
    y_train_tensor = torch.tensor(y_train_np).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_np)
    y_test_tensor = torch.tensor(y_test_np).unsqueeze(1)


    # Training loop
    epochs = 100  # Reduced epochs for faster testing
    batch_size = 32
    patience = 10
    best_f1 = 0.0  # Initialize with a low value for F1 score (since higher is better)
    counter = 0

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train_np), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation and F1-score calculation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            predicted = (test_outputs > 0.5).float().cpu().numpy()
            actual = y_test_tensor.cpu().numpy()

        report = classification_report(actual, predicted, output_dict=True, zero_division=0, labels=[0.0, 1.0])  # Get macro-average F1 score, set labels
        macro_f1 = report['macro avg']['f1-score']  # Get macro-average F1 score


        if macro_f1 > best_f1: #Check if current macro_f1 is better than the previous best
            best_f1 = macro_f1
            counter = 0 # Reset counter since macro_f1 improved
        else:
            counter += 1 #Increment counter if macro_f1 did not improve
            if counter >= patience:
                print("Early stopping triggered.")
                break
        model.train() #Back to training mode after validation

        #if (epoch+1) % 10 == 0:
        #    print (f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Validation Loss: {test_loss.item():.4f}')
        #if (epoch+1) % 10 == 0:
        #    print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Macro-average F1: {macro_f1:.4f}')

    # Evaluation and Classification Report
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predicted = (outputs > 0.5).float().cpu().numpy()  # Move to CPU if using GPU
        actual = y_test_tensor.cpu().numpy()

    report = classification_report(actual, predicted, output_dict=True, zero_division=0, labels=[0.0, 1.0])  # added zero_division, set labels
    all_results.append({
        'f1-score_0.0': report['0.0']['f1-score'],
        'f1-score_1.0': report['1.0']['f1-score'],
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'num_features': X_train.shape[1],
        'features': X_train.columns.tolist()
    })


# --- Results Table ---
results_df = pd.DataFrame(all_results)
print("\nPerformance Comparison:")
print(results_df)