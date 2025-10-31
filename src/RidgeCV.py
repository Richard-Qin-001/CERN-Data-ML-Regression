from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from data_prep import load_data_from_db, data_split
from visualization import visualization_ridge_cv

def train_ridge_cv(X_train_scaled, X_test_scaled, y_train, y_test):
    alphas = np.logspace(-3, 3, 100)
    cv = 5
    ridge_cv_model = RidgeCV(
        alphas=alphas,
        cv=cv,
        scoring='neg_mean_squared_error'
    )

    print("\nStarting to train the RidgeCV ridge regression model, searching for the best alpha...")
    ridge_cv_model.fit(X_train_scaled, y_train)

    best_alpha = ridge_cv_model.alpha_
    y_pred_ridge_cv = ridge_cv_model.predict(X_test_scaled)

    mse_ridge_cv = mean_squared_error(y_test, y_pred_ridge_cv)
    r2_ridge_cv = r2_score(y_test, y_pred_ridge_cv)

    print(f"\nRidgeCV Model Performance: ")
    print(f"Best Alpha Value: {best_alpha}")
    print(f"MSE: {mse_ridge_cv}")
    print(f"R^2: {r2_ridge_cv}")

    return y_pred_ridge_cv

if __name__ == "__main__":
    df = load_data_from_db()
    X_train_scaled, X_test_scaled, y_train, y_test = data_split(df)
    y_pred_ridge_cv = train_ridge_cv(X_train_scaled, X_test_scaled, y_train, y_test)
    visualization_ridge_cv(y_test, y_pred_ridge_cv)