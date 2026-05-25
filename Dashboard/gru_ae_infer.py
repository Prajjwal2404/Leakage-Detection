import os
import joblib
import torch
import torch.nn as nn
import numpy as np


class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(GRUAutoencoder, self).__init__()
        self.seq_1 = nn.GRU(input_dim, 80, batch_first=True) # GRU layer 1
        self.dropout_1 = nn.Dropout(0.2)
        self.seq_2 = nn.GRU(80, 80, batch_first=True) # GRU layer 2
        self.dropout_2 = nn.Dropout(0.2)
        self.fc = nn.Linear(80, input_dim) # Output Projection layer

    def forward(self, x):
        x = x.contiguous()
        x, _ = self.seq_1(x)
        x = self.dropout_1(x)
        x, _ = self.seq_2(x)
        x = self.dropout_2(x)
        x = self.fc(x)
        return x

class WaterLeakageDetector:
    """
    Backend class for real-time water leakage detection using a pre-trained GRU Autoencoder model.
    """
    def __init__(self, checkpoint_dir='..\\Checkpoints', scaler=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Dynamic error thresholds
        self.day_threshold = 0.0048
        self.night_threshold = 0.0016
        
        # 2. Load the trained GRU Autoencoder
        model_path = os.path.join(checkpoint_dir, 'gru_ae_best.pth')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Run gru_ae_model.py first.")
             
        self.model = GRUAutoencoder(input_dim=40).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        if scaler:
            # 3. Load the scaler used during training for consistent feature scaling
            scaler_path = os.path.join(checkpoint_dir, 'scaler.gz')
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run data_preprocess.py first.")
            
            self.scaler = joblib.load(scaler_path)

    def predict(self, feature_vector, is_nighttime=0):
        """
        Predicts anomaly (leak) status for a single real-time snapshot / timestep.
        
        Args:
            feature_vector (list or numpy array): An array of exactly 40 numbers.
                Requires the last element (index 39) to be the `Is_Nighttime` flag (1 or 0).
            is_nighttime (int): The nighttime flag (1 or 0).

        Returns:
            is_leak (bool): True if the system detects an anomaly.
            reconstruction_error (float): The actual MSE value of the network.
            threshold_used (float): The specific threshold margin applied (Day vs Night).
        """
        
        features = np.array(feature_vector, dtype=np.float32).reshape(1, -1)  # Reshape to (1, 40)
        
        if hasattr(self, 'scaler'):
            is_nighttime = int(feature_vector[-1])  # Extract the nighttime flag
            features = self.scaler.transform(features)  # Scale features
        
        if features.shape[-1] != 40:
            raise ValueError(f"Model expects exactly 40 features. Received {features.shape[-1]}.")
            
        # Reshape data to (Batch=1, Sequence=1, Features=40) for GRU compatibility
        x_tensor = torch.tensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(x_tensor)
            
        # Calculate reconstruction mean squared error
        error = np.mean(np.square(x_tensor.cpu().numpy() - reconstructed.cpu().numpy()))
        
        # Determine strictness context
        threshold = self.night_threshold if is_nighttime else self.day_threshold
        
        # Flag leak if reconstruction error crosses bounds
        is_leak = bool(error > threshold)
        
        return is_leak, error, threshold

if __name__ == "__main__":
    try:
        print("Initializing Backend Leak Detector...")
        detector = WaterLeakageDetector('..\\Checkpoints', scaler=True)
        
        # Feed mock scaled dashboard data: 39 sensor inputs + 1 Nighttime flag
        mock_dashboard_data = np.random.randn(40).tolist()
        mock_dashboard_data[-1] = 1.0  # Force Nighttime mode
        
        is_leak, mse, thresh = detector.predict(mock_dashboard_data)
        
        print("\n--- INFERENCE RESULT ---")
        print(f"Status:         {'LEAK DETECTED 🚨' if is_leak else 'NORMAL 🟢'}")
        print(f"System Error:   {mse:.5f}")
        print(f"Allowed Limit:  {thresh}")
        
    except Exception as e:
        print(f"Failed to run inference: {e}")
