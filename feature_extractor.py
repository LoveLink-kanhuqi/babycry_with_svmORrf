import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import joblib
import os

class FeatureExtractor:
    def __init__(self):
        self.sample_rate = 22050  # Default sample rate
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract all features from an audio file."""
        # Load audio file with its original duration
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {str(e)}")
            return None
            
        features = []
        
        # 1. Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        features.append(zcr)
        
        # 2. Short-time energy
        ste = np.mean(np.square(y))
        features.append(ste)
        
        # 3. STE acceleration
        ste_acc = np.sum(zcr > 0)
        features.append(ste_acc)
        
        # 4. Short-time zero crossing rate
        stzcr = np.mean(librosa.feature.zero_crossing_rate(y))
        features.append(stzcr)
        
        # 5. Spectral centroid
        spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features.append(spectral_centroids)
        
        # 6. Spectral bandwidth
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features.append(spectral_bandwidth)
        
        # 7. Spectral rolloff
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features.append(spectral_rolloff)
        
        # 8. Spectral flatness
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        features.append(spectral_flatness)
        
        # 9. MFCC (taking mean of first coefficient)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr)[0])
        features.append(mfcc)
        
        return np.array(features)

    def extract_features_batch(self, file_list: List[str]) -> np.ndarray:
        """Extract features from a list of audio files."""
        features_list = []
        for audio_path in file_list:
            features = self.extract_features(audio_path)
            if features is not None:
                features_list.append(features)
        return np.array(features_list)

    def fit_scaler(self, features: np.ndarray):
        """Fit the scaler to the training data."""
        self.scaler.fit(features)
        self.is_fitted = True
        
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using the fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted yet. Call fit_scaler first.")
        return self.scaler.transform(features)
        
    def save_scaler(self, path: str):
        """Save the fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted yet. Cannot save.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        
    def load_scaler(self, path: str):
        """Load a fitted scaler."""
        self.scaler = joblib.load(path)
        self.is_fitted = True

class DataProcessor:
    @staticmethod
    def split_data(data_csv: str, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation and test sets."""
        assert train_ratio + val_ratio + test_ratio == 1.0
        
        # Read CSV file
        df = pd.read_csv(data_csv)
        
        # Convert labels to binary
        df['label'] = (df['is_cry'] == 'yes').astype(int)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Calculate split indices
        train_idx = int(len(df) * train_ratio)
        val_idx = int(len(df) * (train_ratio + val_ratio))
        
        # Split data
        train_df = df[:train_idx]
        val_df = df[train_idx:val_idx]
        test_df = df[val_idx:]
        
        return train_df, val_df, test_df