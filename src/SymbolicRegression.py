from pysr import PySRRegressor
from data_prep import symbolic_regression_data_prep
import numpy as np
import pandas as pd

def run_pysr_regressor(X_train, y_train, feature_names):
    model = PySRRegressor(
        model_selection='best',
        niterations=1000,
        ncycles_per_iteration=700,
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['square'],
        extra_sympy_mappings={np.square : 'square'},
        
        complexity_of_variables=1,
        maxsize=30,
        
        procs=6,
        tempdir='./pysr_temp_cache',
        batching=True
    )
    print('\nStarting PySR symbolic regression (formula evolution)...')
    model.fit(X_train, y_train, variable_names=feature_names)

    best_formula_result = model.get_best()
    
    if isinstance(best_formula_result, pd.Series):
        best_formula_df = best_formula_result.to_frame().T
    else:
        best_formula_df = best_formula_result

    if 'score' not in best_formula_df.columns:
         print("Error: 'score' column not found in PySR output. Check PySR configuration.")
         return None, None

    print('\nThe best formula PySR found: ')

    best_score = best_formula_df.iloc[0]['score'] 
    best_loss = best_formula_df.iloc[0]['loss']
    best_complexity = best_formula_df.iloc[0]['complexity']
    
    r_squared_on_test = model.score(X_train, y_train)

    print(f"Best Formula Score (Occam's Razor): {best_score:.8f}")
    print(f"Best Formula Complexity: {int(best_complexity)}")
    print(f"Best Formula Loss (MSE): {best_loss:.8f}")
    print(f"Estimated R^2 (on Training Set): {r_squared_on_test:.8f}")

    readable_formula = best_formula_df.iloc[0]['sympy_format']

    print('\nFormula (SymPy format): ')
    print(f'{readable_formula}')

    print('\nComparison of physics formulas: ')
    print(f'(E1 + E2)^2 - (P_sum_x^2 + P_sum_y^2 + P_sum_z^2)')

    return readable_formula, model

if __name__ == "__main__":
    X_train_sr, X_test_sr, Y_train_sr, Y_test_sr, core_features = symbolic_regression_data_prep()
    readable_formula, est_gp = run_pysr_regressor(X_train_sr, Y_train_sr, core_features)
    