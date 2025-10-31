from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_prep import load_data_from_db, data_split
from visualization import visualization_knn

def train_knn(X_train_scaled, X_test_scaled, y_train, y_test):
    n_neighbors = 5
    knn_model = KNeighborsRegressor(n_neighbors)
    print(f"\nStart training the KNN regression model (K={n_neighbors})...")
    knn_model.fit(X_train_scaled, y_train)
    y_pred_knn = knn_model.predict(X_test_scaled)

    mse_knn = mean_squared_error(y_test, y_pred_knn)
    r2_knn = r2_score(y_test, y_pred_knn)

    print(f"\nKNN Regression Model Performance (K={n_neighbors})")
    print(f"MSE: {mse_knn}")
    print(f"R^2: {r2_knn}")
    return y_pred_knn

if __name__ == '__main__':
    df = load_data_from_db()
    X_train_scaled, X_test_scaled, y_train, y_test = data_split(df)
    y_pred_knn = train_knn(X_train_scaled, X_test_scaled, y_train, y_test)
    visualization_knn(y_test, y_pred_knn)