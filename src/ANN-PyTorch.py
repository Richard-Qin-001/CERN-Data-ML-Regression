import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
import numpy as np
from data_prep import load_data_from_db, data_split
from visualization import visualization_ann, plot_permutation_importance

class ANNRegressor(nn.Module):
    def __init__(self, input_size, *args, **kwargs):
        super(ANNRegressor, self).__init__(*args, **kwargs)
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.output_layer(x)
        return x
    
def train_ann_pytorch(X_train_scaled, X_test_scaled, y_train, y_test):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nThe Model will use device: {device}")

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    INPUT_DIM = X_train_scaled.shape[1]
    model = ANNRegressor(INPUT_DIM).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    batch_size = 32
    print("\n Start training ANN Pytorch Regression Model ...")
    for epoch in range(epochs):
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
             print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor).cpu()
        y_pred_ann_pytorch = y_pred_tensor.numpy().flatten()
    
    mse_ann_pytorch = mean_squared_error(y_test, y_pred_ann_pytorch)
    r2_ann_pytorch = r2_score(y_test, y_pred_ann_pytorch)

    print("\nANN Regression Model Performance: ")
    print(f"MSE: {mse_ann_pytorch}")
    print(f"R^2: {r2_ann_pytorch}")
    return y_pred_ann_pytorch, model

def calculate_permutation_importance(model, X_test_scaled, y_test, X_test_df, device):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_pred_baseline = model(X_test_tensor).cpu().numpy().flatten()
        baseline_mse = mean_squared_error(y_test, y_pred_baseline)
    print(f"\nBaseline MSE: {baseline_mse:.4f}")

    feature_names = X_test_df.columns
    n_features = X_test_scaled.shape[1]
    importances = defaultdict(list)
    n_repeats = 5

    for i in range(n_features):
        feature_name = feature_names[i]
        for repeat in range(n_repeats):
            X_permuted = X_test_scaled.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])

            with torch.no_grad():
                X_permuted_tensor = torch.tensor(X_permuted, dtype=torch.float32).to(device)
                y_pred_permuted = model(X_permuted_tensor).cpu().numpy().flatten()
            
            permuted_mse = mean_squared_error(y_test, y_pred_permuted)

            importance = permuted_mse - baseline_mse
            importances[feature_name].append(importance)
    
    feature_importances = {}
    for name, imp_list in importances.items():
        feature_importances[name] = np.mean(imp_list)
    
    sorted_importance = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)

    print("\nPermutation Feature Importance: ")
    print("Importance score indicates: the increase of MSE relative to the baseline. The higher the score, the more important the feature.")
    for name, importance in sorted_importance:
        print(f"{name:<15}: {importance:.4f}")

    plot_permutation_importance(sorted_importance)

    return sorted_importance


if __name__ == "__main__":
    df = load_data_from_db()
    X_train_scaled, X_test_scaled, y_train, y_test, X_test_df = data_split(df)
    y_pred_ann_pytorch, model_ann_pytorch = train_ann_pytorch(X_train_scaled, X_test_scaled, y_train, y_test)
    visualization_ann(y_test, y_pred_ann_pytorch)
    calculate_permutation_importance(model_ann_pytorch, X_test_scaled, y_test, X_test_df, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))