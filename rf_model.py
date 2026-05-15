import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_preprocess import load_and_preprocess_data


def train_evaluate_model(scada_file, leakages_file):
    # 1. Load Data
    print("Loading and preprocessing data...")
    X, Y = load_and_preprocess_data(scada_file, leakages_file, resample_freq='5min', rolling_window=36)
    
    # 2. Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, shuffle=True)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # 3. Initialize and Train the Model
    print("\nTraining the model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, Y_train)
    
    # 4. Predict and Evaluate
    print("Evaluating model on test data...")
    Y_pred = model.predict(X_test)
    
    print("\n" + "="*30)
    print("--- RANDOM FOREST MODEL EVALUATION ---")
    print("="*30)
    print(f"Mean Squared Error (MSE): {mean_squared_error(Y_test, Y_pred):.4f}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(Y_test, Y_pred):.4f}")
    print(f"R-squared (R2): {r2_score(Y_test, Y_pred):.4f}")

if __name__ == "__main__":
    scada_file = os.path.join('Dataset', '2018_SCADA.xlsx')
    leakages_file = os.path.join('Dataset', '2018_Leakages.csv')
    
    train_evaluate_model(scada_file, leakages_file)