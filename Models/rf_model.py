import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from data_preprocess import load_and_preprocess_data


def train_evaluate_model(scada_file, leakages_file):
    # 1. Load Data
    print("Loading and preprocessing data...")
    X, Y = load_and_preprocess_data(scada_file, leakages_file, rolling_window=36)
    
    # 2. Train-Test Split
    X_train, Y_train = X[1200:int(0.3 * len(X))], Y[1200:int(0.3 * len(Y))]
    X_test, Y_test = pd.concat([X[:1200], X[int(0.3 * len(X)):]]), pd.concat([Y[:1200], Y[int(0.3 * len(Y)):]])
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # 3. Initialize and Train the Model
    print("\nTraining the model...")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, Y_train)
    
    # 4. Predict and Evaluate
    print("Evaluating model on test data...")
    Y_pred = model.predict(X_test)
    
    print("\n" + "="*30)
    print("--- RANDOM FOREST MODEL EVALUATION ---")
    print("="*30)
    print(f"Accuracy (RF): {accuracy_score(Y_test, Y_pred) * 100:.2f}%\n")
    print(f"Precision: {precision_score(Y_test, Y_pred, zero_division=0):.4f}")
    print(f"Recall (Sensitivity to leaks): {recall_score(Y_test, Y_pred, zero_division=0):.4f}")
    print(f"F1-Score: {f1_score(Y_test, Y_pred, zero_division=0):.4f}\n")
    
    print("Classification Report (RF):")
    print(classification_report(Y_test, Y_pred, zero_division=0))
    
    print("Confusion Matrix (RF) [Normal=0, Leak=1]:")
    print(confusion_matrix(Y_test, Y_pred))

if __name__ == "__main__":
    scada_file = os.path.join('Dataset', '2018_SCADA.xlsx')
    leakages_file = os.path.join('Dataset', '2018_Leakages.csv')
    
    train_evaluate_model(scada_file, leakages_file)