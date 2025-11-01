from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_prep import load_data_from_db, data_split
from visualization import visualization_dt

def train_decision_tree(X_train_scaled, X_test_scaled, y_train, y_test):
    max_depth=10
    random_state=42
    dt_model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)

    print(f"Start training the Decision Tree Regression model (max_depth={max_depth})...")
    dt_model.fit(X_train_scaled, y_train)

    y_pred_dt = dt_model.predict(X_test_scaled)

    mse_dt = mean_squared_error(y_test, y_pred_dt)
    r2_dt = r2_score(y_test, y_pred_dt)

    print(f"\nDecision Tree Regression Model Performance (max_depth={max_depth})...")
    print(f"MSE: {mse_dt}")
    print(f"R^2: {r2_dt}")

    return y_pred_dt

if __name__ == '__main__':
    df = load_data_from_db()
    X_train_scaled, X_test_scaled, y_train, y_test, _ = data_split(df)
    y_pred_dt = train_decision_tree(X_train_scaled, X_test_scaled, y_train, y_test)
    visualization_dt(y_test, y_pred_dt)