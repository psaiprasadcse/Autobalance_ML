import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC, SMOTE, RandomOverSampler

# Function to load dataset
def load_dataset(file_path):
    """Loads the dataset and replaces '?' with NaN for proper handling."""
    df = pd.read_csv(file_path)
    df.replace('?', np.nan, inplace=True)  # Handle missing values
    return df

# Function to identify numerical and categorical columns
def identify_column_types(df):
    """Separates numerical and categorical columns correctly."""
    numerical_cols = []
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])  # Convert to numeric if possible
            numerical_cols.append(col)
        except ValueError:
            pass  # If conversion fails, it's categorical

    categorical_cols = list(set(df.columns) - set(numerical_cols))
    return numerical_cols, categorical_cols

# Function to handle missing values
def handle_missing_values(df, numerical_cols, categorical_cols):
    """Fills missing values: median for numerical, mode for categorical."""
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# Function to check for class imbalance
def check_class_imbalance(df, target_column):
    """Checks class distribution and determines if the dataset is imbalanced."""
    class_counts = df[target_column].value_counts(normalize=True) * 100
    num_classes = len(class_counts)
    imbalance_threshold = 60 if num_classes == 2 else 20  # 60% for binary, 20% for multi-class
    is_imbalanced = any(pct > imbalance_threshold for pct in class_counts)
    return is_imbalanced, class_counts.to_dict(), num_classes

# Function to encode categorical features
def encode_categorical_features(df, categorical_cols):
    """Encodes only categorical features using Label Encoding."""
    label_encodings = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encodings[col] = dict(zip(le.classes_, le.transform(le.classes_)))  # Store mappings
    return df, label_encodings

# Function to apply correct resampling method with "Remarks" column
def apply_resampling(df, target_column, numerical_cols, categorical_cols):
    """Applies SMOTE-NC for mixed categorical/numerical, SMOTE for numerical only, ROS for fully categorical.
       Also adds a 'Remarks' column to indicate if a record is 'New' (synthetic) or 'Old' (original)."""
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical features
    X, label_encodings = encode_categorical_features(X, categorical_cols)

    # Identify categorical feature indices
    categorical_feature_indices = [X.columns.get_loc(col) for col in categorical_cols if col in X.columns]

    # Save original dataset size
    original_size = len(X)

    # Select appropriate resampling technique
    if not numerical_cols:  # Fully categorical dataset
        sampler = RandomOverSampler(random_state=42)
    elif categorical_cols:  # Mixed dataset (Categorical + Numerical)
        sampler = SMOTENC(categorical_features=categorical_feature_indices, random_state=42)
    else:  # Fully numerical dataset
        sampler = SMOTE(random_state=42)

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    # Convert back to DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled

    # Add "Remarks" column: "Old" for original instances, "New" for synthetic ones
    df_resampled["Remarks"] = ["Old"] * original_size + ["New"] * (len(df_resampled) - original_size)

    return df_resampled, label_encodings

# Main execution function
def main():
    file_path = input("Enter the dataset file path (CSV format): ").strip()
    df = load_dataset(file_path)

    # Identify target column
    print("\nColumns in dataset:", list(df.columns))
    target_column = input("Enter the target column name: ").strip()

    # Identify numerical and categorical columns correctly
    numerical_cols, categorical_cols = identify_column_types(df)

    # Ensure the target column is not mistakenly categorized
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # Handle missing values
    df = handle_missing_values(df, numerical_cols, categorical_cols)

    # Check class imbalance
    is_imbalanced, class_distribution, num_classes = check_class_imbalance(df, target_column)

    if is_imbalanced:
        print(f"\nDataset is class-imbalanced. Detected {num_classes} classes. Applying resampling...")

        # Apply resampling
        df_balanced, label_encodings = apply_resampling(df, target_column, numerical_cols, categorical_cols)

        # Save balanced dataset
        balanced_file_path = "balanced_dataset.csv"
        df_balanced.to_csv(balanced_file_path, index=False)
        print(f"\n✅ Balanced dataset saved as: {balanced_file_path}")

        # Save label encoding mappings
        encoding_data = [[col, original, encoded] for col, mapping in label_encodings.items() for original, encoded in mapping.items()]
        label_encoding_df = pd.DataFrame(encoding_data, columns=["Column", "Original Value", "Encoded Value"])
        label_encoding_file = "label_encoding_mappings.csv"
        label_encoding_df.to_csv(label_encoding_file, index=False)
        print(f"\n✅ Label Encoding Mappings saved as: {label_encoding_file}")
        print("\n Remove The Remarks Column Before Proceeding for ML Model Generation")

    else:
        print("\n✅ Dataset is already balanced. No resampling needed.")

if __name__ == "__main__":
    main()
