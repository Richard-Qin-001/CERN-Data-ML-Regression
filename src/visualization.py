# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import config
from data_prep import load_csv

def draw_chart(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['M'], bins=150, kde=True, color='skyblue', edgecolor = 'black')
    plt.title("Distribution of Invariant Mass (M)", fontsize = 16)
    plt.xlabel("Invariant Mass (GeV)", fontsize = 14)
    plt.ylabel("Number of Events", fontsize = 14)
    plt.axvline(91.1876, color = 'b', linestyle = '--', label = 'Z Boson Mass')
    plt.legend()
    plt.grid(axis= 'y', alpha = 0.5)
    plt.xlim(0, 150)
    plt.show()

    cols_to_correlate = df.drop(columns = ['Run', 'Event']).columns
    corr_matrix = df[cols_to_correlate].corr()
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        annot=False,
        fmt='.2f',
        linewidths=.5,
        cbar_kws={'label':'Correlation Coefficient'}
    )
    plt.title('Feature Correlation Matrix (Including M)', fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.show()

    plt.figure(figsize = (14, 7))
    df_signal = df[df['Q_prod'] == -1]
    df_background = df[df['Q_prod'] == 1]
    sns.histplot(
        df_signal['M'], bins=100, kde = False, color = 'green', alpha=0.6, label = r"$Q_1 \cdot Q_2 = -1$(Signal Candidate)"
    )
    sns.histplot(
        df_background['M'], bins=100, kde = False, color = 'red', alpha=0.6, label = r"$Q_1 \cdot Q_2 = +1$(Background Candidate)"
    )
    plt.title("Invariant Mass (M) Distribute by Charge Product ($Q_{prod}$)", fontsize = 18)
    plt.xlabel("Invariant Mass (GeV)",fontsize=14)
    plt.ylabel("Number of Events",fontsize=14)
    plt.axvline(91.1876, color='b', linestyle='--', label='Z Boson Mass')
    plt.xlim(0, 120)
    plt.legend()
    plt.grid(axis= 'y', alpha=0.5)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['pt1'], bins=150, kde=True, color='purple', edgecolor = 'black')
    plt.title("Distribution of Electron 1 Transverse Momentum (pt1)", fontsize=16)
    plt.xlabel("pt1 (GeV)", fontsize=14)
    plt.ylabel('Number of Events', fontsize=14)
    plt.xlim(0, 150)
    # plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['eta1'], bins=150, kde=True, color='green', edgecolor='black')
    plt.title("Distribution of Electron 1 Pseudorapidity (eta1)", fontsize=16)
    plt.xlabel("eta1", fontsize=14)
    plt.ylabel("Number of Events", fontsize=14)
    plt.grid(axis='y', alpha=0.5)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['dR'], bins=150, kde=True, color='green', edgecolor='black')
    plt.title("Distribution of Delta R (dR)", fontsize=16)
    plt.xlabel("Delta R", fontsize=14)
    plt.ylabel("Number of Events", fontsize=14)
    plt.grid(axis='y', alpha=0.5)
    plt.show()

def visualization_linear_regression(y_test, y_pred_linear):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_linear, alpha=0.3, s=10, label='Predicted & Actual')

    max_val = max(y_test.max(), y_pred_linear.max())
    min_val = min(y_test.min(), y_pred_linear.min())
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='red', linestyle='--', linewidth=2, label='Ideal Prediction (y=x)')

    plt.title("Linear Regression: Predicted & Actual Invariant Mass (M)", fontsize=14)
    plt.xlabel("Actual Invariant Mass (GeV)", fontsize=12)
    plt.ylabel("Predicted Invariant Mass (GeV)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def visualization_knn(y_test, y_pred_knn):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_knn, alpha=0.3, s=10, label='Predicted & Actual')

    max_val = max(y_test.max(), y_pred_knn.max())
    min_val = min(y_test.min(), y_pred_knn.min())
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='red', linestyle='--', linewidth=2, label='Ideal Prediction (y=x)')

    plt.title("KNeighbors Regression: Predicted & Actual Invariant Mass (M)", fontsize=14)
    plt.xlabel("Actual Invariant Mass (GeV)", fontsize=12)
    plt.ylabel("Predicted Invariant Mass (GeV)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def visualization_dt(y_test, y_pred_dt):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_dt, alpha=0.3, s=10, label='Predicted & Actual')

    max_val = max(y_test.max(), y_pred_dt.max())
    min_val = min(y_test.min(), y_pred_dt.min())
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='red', linestyle='--', linewidth=2, label='Ideal Prediction (y=x)')

    plt.title("Decision Tree Regression: Predicted & Actual Invariant Mass (M)", fontsize=14)
    plt.xlabel("Actual Invariant Mass (GeV)", fontsize=12)
    plt.ylabel("Predicted Invariant Mass (GeV)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def visualization_ridge_cv(y_test, y_pred_ridge_cv):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_ridge_cv, alpha=0.3, s=10, label='Predicted & Actual')

    max_val = max(y_test.max(), y_pred_ridge_cv.max())
    min_val = min(y_test.min(), y_pred_ridge_cv.min())
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='red', linestyle='--', linewidth=2, label='Ideal Prediction (y=x)')

    plt.title("RidgeCV: Predicted & Actual Invariant Mass (M)", fontsize=14)
    plt.xlabel("Actual Invariant Mass (GeV)", fontsize=12)
    plt.ylabel("Predicted Invariant Mass (GeV)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def visualization_random_forest(y_test, y_pred_random_forest):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_random_forest, alpha=0.3, s=10, label='Predicted & Actual')

    max_val = max(y_test.max(), y_pred_random_forest.max())
    min_val = min(y_test.min(), y_pred_random_forest.min())
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='red', linestyle='--', linewidth=2, label='Ideal Prediction (y=x)')

    plt.title("Random Forest Regression: Predicted & Actual Invariant Mass (M)", fontsize=14)
    plt.xlabel("Actual Invariant Mass (GeV)", fontsize=12)
    plt.ylabel("Predicted Invariant Mass (GeV)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def visualization_xgboost(y_test, y_pred_xgb):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_xgb, alpha=0.3, s=10, label='Predicted & Actual')

    max_val = max(y_test.max(), y_pred_xgb.max())
    min_val = min(y_test.min(), y_pred_xgb.min())
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='red', linestyle='--', linewidth=2, label='Ideal Prediction (y=x)')

    plt.title("XGBoost: Predicted & Actual Invariant Mass (M)", fontsize=14)
    plt.xlabel("Actual Invariant Mass (GeV)", fontsize=12)
    plt.ylabel("Predicted Invariant Mass (GeV)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def visualization_ann(y_test, y_pred_ann):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_ann, alpha=0.3, s=10, label='Predicted & Actual')

    max_val = max(y_test.max(), y_pred_ann.max())
    min_val = min(y_test.min(), y_pred_ann.min())
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='red', linestyle='--', linewidth=2, label='Ideal Prediction (y=x)')

    plt.title("ANN: Predicted & Actual Invariant Mass (M)", fontsize=14)
    plt.xlabel("Actual Invariant Mass (GeV)", fontsize=12)
    plt.ylabel("Predicted Invariant Mass (GeV)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    df = load_csv()
    draw_chart(df)