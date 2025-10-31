from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from visualization import visualization_linear_regression
from data_prep import load_data_from_db, data_split

def train_lr(X_train_scaled, X_test_scaled, y_train, y_test):
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
    y_pred_linear = train_lr(X_train_scaled, X_test_scaled, y_train, y_test)
    visualization_linear_regression(y_test, y_pred_linear)