import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from data_preprocess import load_and_preprocess_data


def train_evaluate_model(scada_file, leakages_file):
    # 1. Load Data
    print("Loading and preprocessing data...")
    X, Y = load_and_preprocess_data(scada_file, leakages_file, rolling_window=36)
    
    # Train-Test Split
    X_train, Y_train = X[1200:int(0.04 * len(X))], Y[1200:int(0.04 * len(Y))]
    X_test, Y_test = pd.concat([X[:1200], X[int(0.04 * len(X)):]]), pd.concat([Y[:1200], Y[int(0.04 * len(Y)):]])
    
    contamination_rate = max(Y_train.mean(), 0.001)
    print(f"Training samples: {len(X_train)} (Contamination rate: {contamination_rate:.4f})")
    print(f"Testing samples: {len(X_test)}")
    
    # 3. Initialize and Train the Model
    print("\nTraining the Isolation Forest model on mixed data...")
    model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42, n_jobs=-1)
    model.fit(X_train)
    
    # 4. Predict and Evaluate on ENTIRE dataset
    print("Evaluating model on full historical data...")
    raw_pred = model.predict(X_test)
    Y_pred = (raw_pred == -1).astype(int)
    
    print("\n" + "="*30)
    print("--- ISOLATION FOREST MODEL EVALUATION ---")
    print("="*30)
    print(f"Accuracy (IF): {accuracy_score(Y_test, Y_pred) * 100:.2f}%\n")
    print(f"Precision: {precision_score(Y_test, Y_pred, zero_division=0):.4f}")
    print(f"Recall (Sensitivity to leaks): {recall_score(Y_test, Y_pred, zero_division=0):.4f}")
    print(f"F1-Score: {f1_score(Y_test, Y_pred, zero_division=0):.4f}\n")
    
    print("Classification Report (IF):")
    print(classification_report(Y_test, Y_pred, zero_division=0))
    
    print("Confusion Matrix (IF) [Normal=0, Leak=1]:")
    print(confusion_matrix(Y_test, Y_pred))

if __name__ == "__main__":
    scada_file = os.path.join('Dataset', '2018_SCADA.xlsx')
    leakages_file = os.path.join('Dataset', '2018_Leakages.csv')
    
    train_evaluate_model(scada_file, leakages_file)