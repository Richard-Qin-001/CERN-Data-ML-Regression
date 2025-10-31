from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import sqlite3
import config
import matplotlib.pyplot as plt

def load_data_from_db():
    conn = sqlite3.connect(config.DATABASE_PATH)
    try:
        df_loaded = pd.read_sql_query("SELECT * FROM electron_events", conn)
        print(f"Data has been successfully loaded from the database. Total rows: {df_loaded.shape[0]}")
    except Exception as e:
        print(f"Error: Unable to load data from the database. Please check the table name and database file. Error message: {e}")
        exit(0)
    conn.close()
    return df_loaded

def data_split(df):
    features_to_exclude = ['Run', 'Event', 'M']
    X = df.drop(features_to_exclude, axis=1)
    y = df['M']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"\nTraining set X_train shape: {X_train.shape}")
    print(f"\nTesting set X_test shape: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    print("Data segmentation and normalization completed.")
    return X_train_scaled, X_test_scaled, y_train, y_test

def train(X_train_scaled, X_test_scaled, y_train, y_test):
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)

    y_pred_linear = linear_model.predict(X_test_scaled)

    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)

    print("\nLinear Regression Baseline Performance: ")
    print(f"MSE: {mse_linear}")
    print(f"R^2: {r2_linear}")

    return y_pred_linear

def visualization(y_test, y_pred_linear):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_linear, alpha=0.3, s=10, label='Predicted vs. Actual')

    max_val = max(y_test.max(), y_pred_linear.max())
    min_val = min(y_test.min(), y_pred_linear.min())
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='red', linestyle='--', linewidth=2, label='Ideal Prediction (y=x)')

    plt.title("Linear Regression: Predicted vs. Actual Invariant Mass (M)", fontsize=14)
    plt.xlabel("Actual Invariant Mass (GeV)", fontsize=12)
    plt.ylabel("Predicted Invariant Mass (GeV)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    df = load_data_from_db()
    X_train_scaled, X_test_scaled, y_train, y_test = data_split(df)
    y_pred_linear = train(X_train_scaled, X_test_scaled, y_train, y_test)
    visualization(y_test, y_pred_linear)

