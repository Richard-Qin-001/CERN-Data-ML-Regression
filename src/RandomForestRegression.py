from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_prep import load_data_from_db, data_split
from visualization import visualization_random_forest

def train_random_forest(X_train_scaled, X_test_scaled, y_train, y_test):
    n_estimators=100
    max_depth=10
    random_state=42 
    n_jobs=-1
    
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth, 
        random_state=random_state, 
        n_jobs=n_jobs
    )
    print(f"\nStarting to train the Random Forest regression model (n_estimators={n_estimators}, max_depth={max_depth})...")
    rf_model.fit(X_train_scaled, y_train)

    y_pred_rf = rf_model.predict(X_test_scaled)

    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print("\nRandom Forest Regression Performance: ")
    print(f"MSE: {mse_rf}")
    print(f"R^2: {r2_rf}")

    return y_pred_rf, rf_model

if __name__ == "__main__":
    df = load_data_from_db()
    X_train_scaled, X_test_scaled, y_train, y_test, _ = data_split(df)
    y_pred_rf, rf_model = train_random_forest(X_train_scaled, X_test_scaled, y_train, y_test)
    visualization_random_forest(y_test, y_pred_rf)