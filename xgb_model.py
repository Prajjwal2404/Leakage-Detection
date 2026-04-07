import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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
    model = XGBClassifier(n_estimators=100, scale_pos_weight=0.0221281976461, random_state=42, n_jobs=-1)
    model.fit(X_train, Y_train)
    
    # 4. Predict and Evaluate
    print("Evaluating model on test data...")
    Y_pred = model.predict(X_test)
    
    print("\n" + "="*30)
    print("--- XGBOOST MODEL EVALUATION ---")
    print("="*30)
    print(f"Accuracy (XGB): {accuracy_score(Y_test, Y_pred) * 100:.2f}%\n")
    
    print("Classification Report (XGB):")
    print(classification_report(Y_test, Y_pred))
    
    print("Confusion Matrix (XGB):")
    print(confusion_matrix(Y_test, Y_pred))

if __name__ == "__main__":
    scada_file = os.path.join('Dataset', '2018_SCADA.xlsx')
    leakages_file = os.path.join('Dataset', '2018_Leakages.csv')
    
    train_evaluate_model(scada_file, leakages_file)
