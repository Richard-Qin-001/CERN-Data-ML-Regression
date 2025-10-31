import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from data_prep import load_data_from_db, data_split
from visualization import visualization_xgboost

def train_xgboost(X_train_scaled, X_test_scaled, y_train, y_test):
    objective='reg:squarederror'
    n_estimators=500
    max_depth=10
    tree_method='hist'
    learning_rate=0.05
    random_state=42
    xgb_model = xgb.XGBRegressor(
        objective=objective,
        n_estimators=n_estimators,
        max_depth=max_depth,
        tree_method=tree_method,
        n_jobs = -1,
        learning_rate=learning_rate,
        random_state=random_state
    )

    print(f"\nStarting training XGBoost regression model (GPU accelerated, {n_estimators} trees, depth {max_depth})...")
    xgb_model.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    print("\nXGBoost Regression Model Performance :")
    print(f"MSE: {mse_xgb}")
    print(f"R^2: {r2_xgb}")

    return y_pred_xgb, xgb_model


if __name__ == "__main__":
    df = load_data_from_db()
    X_train_scaled, X_test_scaled, y_train, y_test = data_split(df)
    y_pred_xgb, xgb_model = train_xgboost(X_train_scaled, X_test_scaled, y_train, y_test)
    visualization_xgboost(y_test, y_pred_xgb)