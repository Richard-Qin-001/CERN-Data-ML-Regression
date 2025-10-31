from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from visualization import visualization_linear_regression
from data_prep import load_data_from_db

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

if __name__ == "__main__":
    df = load_data_from_db()
    X_train_scaled, X_test_scaled, y_train, y_test = data_split(df)
    y_pred_linear = train(X_train_scaled, X_test_scaled, y_train, y_test)
    visualization_linear_regression(y_test, y_pred_linear)