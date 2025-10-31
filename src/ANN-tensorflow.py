import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from data_prep import load_data_from_db, data_split
from visualization import visualization_ann

def build_ann_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='relu'),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_ann(X_train_scaled, X_test_scaled, y_train, y_test, input_dim):
    ann_model = build_ann_model(input_dim)
    y_train_array = y_train.values
    print("\n")
    history = ann_model.fit(
        X_train_scaled, y_train_array,
        epochs=50,
        batch_size=32,
        verbose=0,
        validation_split=0.1
    )
    y_pred_ann = ann_model.predict(X_test_scaled).flatten()

    mse_ann = mean_squared_error(y_test, y_pred_ann)
    r2_ann = r2_score(y_test, y_pred_ann)

    print("\nANN Regression Model Performance: ")
    print(f"MSE: {mse_ann}")
    print(f"R^2: {r2_ann}")

    return y_pred_ann

if __name__ == "__main__":
    df = load_data_from_db()
    X_train_scaled, X_test_scaled, y_train, y_test = data_split(df)
    input_dim = X_train_scaled.shape[1]
    y_pred_ann = train_ann(X_train_scaled, X_test_scaled, y_train, y_test, input_dim)
    visualization_ann(y_test, y_pred_ann)