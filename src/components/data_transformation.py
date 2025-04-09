# src/components/data_transformation.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataTransformation:
    def __init__(self, cleaned_data_path: str = "data/cleaned_walmart.csv"):
        self.cleaned_data_path = cleaned_data_path
        self.scaler = StandardScaler()

    def load_cleaned_data(self) -> pd.DataFrame:
        """
        Load cleaned data from CSV.
        """
        if not os.path.exists(self.cleaned_data_path):
            raise FileNotFoundError(f"❌ Cleaned data not found at {self.cleaned_data_path}")

        df = pd.read_csv(self.cleaned_data_path)
        print(f"✅ Cleaned data loaded! Shape: {df.shape}")
        return df

    def preprocess(self, df: pd.DataFrame):
        """
        Encode categorical data, scale features, and split into train/test.
        """
        print("🧾 Columns in the dataset:", df.columns.tolist())

        # Safe handling of 'IsHoliday'
        if 'IsHoliday' in df.columns:
            df['IsHoliday'] = df['IsHoliday'].map({True: 1, False: 0})
        else:
            print("⚠️ 'IsHoliday' column not found. Skipping encoding...")

      # Handle Date column
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Year'] = df['Date'].dt.year
        df.drop(columns=['Date'], inplace=True)

        # Split features and target
        X = df.drop(columns=['Weekly_Sales'])
        y = df['Weekly_Sales']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        print("🔄 Data split and scaled:")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
        return X_train, X_test, y_train, y_test

    def run(self):
        """
        Run full data transformation pipeline.
        """
        df = self.load_cleaned_data()
        return self.preprocess(df)


# Run as standalone script
if __name__ == "__main__":
    transformer = DataTransformation()
    X_train, X_test, y_train, y_test = transformer.run()
