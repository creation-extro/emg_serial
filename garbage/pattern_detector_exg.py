#!/usr/bin/env python3
"""
EXG Pattern Detector - Real-time Open/Close Detection
Based on discovered patterns from combined EXG data analysis
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import os

class EXGPatternDetector:
    def __init__(self):
        self.patterns = None
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        # Discovered patterns from analysis
        self.open_patterns = {
            'raw_emg1': {'mean': 33104, 'range': (25142, 51100)},
            'raw_emg2': {'mean': 33186, 'range': (16964, 50156)},
            'normalized_emg1': {'mean': 0.464, 'std': 0.187},
            'normalized_emg2': {'mean': 0.473, 'std': 0.194},
            'derivative_emg1': {'mean': 0.7, 'positive_ratio': 0.510},
            'derivative_emg2': {'mean': -0.0, 'positive_ratio': 0.496}
        }
        
        self.close_patterns = {
            'raw_emg1': {'mean': 33020, 'range': (28006, 38057)},
            'raw_emg2': {'mean': 32793, 'range': (24101, 46603)},
            'normalized_emg1': {'mean': 0.460, 'std': 0.187},
            'normalized_emg2': {'mean': 0.448, 'std': 0.193},
            'derivative_emg1': {'mean': -0.1, 'positive_ratio': 0.491},
            'derivative_emg2': {'mean': 0.1, 'positive_ratio': 0.483}
        }
        
        print("🔍 EXG Pattern Detector Initialized")
        print("📊 Ready to detect Open (0) vs Close (1) patterns")
    
    def create_pattern_features(self, raw_emg1, raw_emg2, normalized_emg1, normalized_emg2, derivative_emg1, derivative_emg2):
        """Create pattern-based features from raw EXG data"""
        
        features = []
        
        # Original features
        features.extend([
            raw_emg1, raw_emg2, normalized_emg1, normalized_emg2, derivative_emg1, derivative_emg2
        ])
        
        # Pattern-based features
        
        # 1. Distance from open/close centroids
        open_emg1_dist = abs(raw_emg1 - self.open_patterns['raw_emg1']['mean'])
        close_emg1_dist = abs(raw_emg1 - self.close_patterns['raw_emg1']['mean'])
        open_emg2_dist = abs(raw_emg2 - self.open_patterns['raw_emg2']['mean'])
        close_emg2_dist = abs(raw_emg2 - self.close_patterns['raw_emg2']['mean'])
        
        features.extend([open_emg1_dist, close_emg1_dist, open_emg2_dist, close_emg2_dist])
        
        # 2. Ratio features
        emg_ratio = raw_emg1 / (raw_emg2 + 1e-6)
        norm_ratio = normalized_emg1 / (normalized_emg2 + 1e-6)
        
        features.extend([emg_ratio, norm_ratio])
        
        # 3. Pattern similarity scores
        open_norm1_sim = 1 / (1 + abs(normalized_emg1 - self.open_patterns['normalized_emg1']['mean']))
        close_norm1_sim = 1 / (1 + abs(normalized_emg1 - self.close_patterns['normalized_emg1']['mean']))
        open_norm2_sim = 1 / (1 + abs(normalized_emg2 - self.open_patterns['normalized_emg2']['mean']))
        close_norm2_sim = 1 / (1 + abs(normalized_emg2 - self.close_patterns['normalized_emg2']['mean']))
        
        features.extend([open_norm1_sim, close_norm1_sim, open_norm2_sim, close_norm2_sim])
        
        # 4. Derivative pattern features
        deriv_sum = derivative_emg1 + derivative_emg2
        deriv_diff = abs(derivative_emg1 - derivative_emg2)
        deriv_sign_match = 1 if (derivative_emg1 > 0) == (derivative_emg2 > 0) else 0
        
        features.extend([deriv_sum, deriv_diff, deriv_sign_match])
        
        # 5. Amplitude features
        total_amplitude = raw_emg1 + raw_emg2
        amplitude_balance = abs(raw_emg1 - raw_emg2) / (total_amplitude + 1e-6)
        
        features.extend([total_amplitude, amplitude_balance])
        
        return np.array(features)
    
    def train_pattern_detector(self, data_file='combined_exg_data.csv'):
        """Train the pattern detector using the best performing model"""
        print(f"🚀 Training Pattern Detector...")
        
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            return False
        
        # Load data
        data = pd.read_csv(data_file)
        print(f"📂 Loaded {len(data):,} samples")
        
        # Create pattern features for all samples
        print(f"🔧 Creating pattern features...")
        pattern_features = []
        
        for _, row in data.iterrows():
            features = self.create_pattern_features(
                row['raw_emg1'], row['raw_emg2'],
                row['normalized_emg1'], row['normalized_emg2'],
                row['derivative_emg1'], row['derivative_emg2']
            )
            pattern_features.append(features)
        
        X = np.array(pattern_features)
        y = data['label'].values
        
        print(f"✅ Created features: {X.shape}")
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Neural Network (best performing model)
        print(f"🧠 Training Neural Network...")
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            random_state=42,
            max_iter=500
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Test accuracy
        accuracy = self.model.score(X_test_scaled, y_test)
        print(f"✅ Pattern Detector trained!")
        print(f"🎯 Test Accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        return True
    
    def detect_pattern(self, raw_emg1, raw_emg2, normalized_emg1, normalized_emg2, derivative_emg1, derivative_emg2):
        """Detect open/close pattern from EXG data"""
        
        if not self.is_trained:
            print("❌ Pattern detector not trained yet!")
            return None
        
        # Create pattern features
        features = self.create_pattern_features(
            raw_emg1, raw_emg2, normalized_emg1, normalized_emg2, derivative_emg1, derivative_emg2
        )
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Pattern analysis
        pattern_analysis = self.analyze_pattern_match(
            raw_emg1, raw_emg2, normalized_emg1, normalized_emg2, derivative_emg1, derivative_emg2
        )
        
        result = {
            'prediction': int(prediction),
            'gesture': 'Open' if prediction == 0 else 'Close',
            'confidence': float(probability[prediction]),
            'probabilities': {
                'open': float(probability[0]),
                'close': float(probability[1])
            },
            'pattern_analysis': pattern_analysis,
            'raw_data': {
                'raw_emg1': raw_emg1,
                'raw_emg2': raw_emg2,
                'normalized_emg1': normalized_emg1,
                'normalized_emg2': normalized_emg2,
                'derivative_emg1': derivative_emg1,
                'derivative_emg2': derivative_emg2
            }
        }
        
        return result
    
    def analyze_pattern_match(self, raw_emg1, raw_emg2, normalized_emg1, normalized_emg2, derivative_emg1, derivative_emg2):
        """Analyze how well the data matches known patterns"""
        
        # Calculate distances to pattern centroids
        open_distances = {
            'raw_emg1': abs(raw_emg1 - self.open_patterns['raw_emg1']['mean']),
            'raw_emg2': abs(raw_emg2 - self.open_patterns['raw_emg2']['mean']),
            'norm_emg1': abs(normalized_emg1 - self.open_patterns['normalized_emg1']['mean']),
            'norm_emg2': abs(normalized_emg2 - self.open_patterns['normalized_emg2']['mean'])
        }
        
        close_distances = {
            'raw_emg1': abs(raw_emg1 - self.close_patterns['raw_emg1']['mean']),
            'raw_emg2': abs(raw_emg2 - self.close_patterns['raw_emg2']['mean']),
            'norm_emg1': abs(normalized_emg1 - self.close_patterns['normalized_emg1']['mean']),
            'norm_emg2': abs(normalized_emg2 - self.close_patterns['normalized_emg2']['mean'])
        }
        
        # Calculate total distances
        open_total_distance = sum(open_distances.values())
        close_total_distance = sum(close_distances.values())
        
        # Determine closer pattern
        closer_to = 'Open' if open_total_distance < close_total_distance else 'Close'
        
        # Key discriminative features analysis
        key_features = {
            'raw_emg2_most_discriminative': {
                'value': raw_emg2,
                'open_mean': self.open_patterns['raw_emg2']['mean'],
                'close_mean': self.close_patterns['raw_emg2']['mean'],
                'closer_to': 'Open' if abs(raw_emg2 - self.open_patterns['raw_emg2']['mean']) < abs(raw_emg2 - self.close_patterns['raw_emg2']['mean']) else 'Close'
            },
            'normalized_emg2_discriminative': {
                'value': normalized_emg2,
                'open_mean': self.open_patterns['normalized_emg2']['mean'],
                'close_mean': self.close_patterns['normalized_emg2']['mean'],
                'closer_to': 'Open' if abs(normalized_emg2 - self.open_patterns['normalized_emg2']['mean']) < abs(normalized_emg2 - self.close_patterns['normalized_emg2']['mean']) else 'Close'
            }
        }
        
        return {
            'open_distances': open_distances,
            'close_distances': close_distances,
            'open_total_distance': open_total_distance,
            'close_total_distance': close_total_distance,
            'closer_to_pattern': closer_to,
            'key_features': key_features
        }
    
    def save_detector(self, filename='exg_pattern_detector.pkl'):
        """Save the trained pattern detector"""
        if not self.is_trained:
            print("❌ Cannot save untrained detector!")
            return False
        
        detector_data = {
            'model': self.model,
            'scaler': self.scaler,
            'open_patterns': self.open_patterns,
            'close_patterns': self.close_patterns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(detector_data, filename)
        print(f"💾 Pattern detector saved: {filename}")
        return True
    
    def load_detector(self, filename='exg_pattern_detector.pkl'):
        """Load a trained pattern detector"""
        if not os.path.exists(filename):
            print(f"❌ Detector file not found: {filename}")
            return False
        
        detector_data = joblib.load(filename)
        self.model = detector_data['model']
        self.scaler = detector_data['scaler']
        self.open_patterns = detector_data['open_patterns']
        self.close_patterns = detector_data['close_patterns']
        self.is_trained = detector_data['is_trained']
        
        print(f"✅ Pattern detector loaded: {filename}")
        return True

def test_pattern_detector():
    """Test the pattern detector with sample data"""
    print("🧪 Testing EXG Pattern Detector")
    print("=" * 50)
    
    # Initialize detector
    detector = EXGPatternDetector()
    
    # Train detector
    if detector.train_pattern_detector():
        # Save detector
        detector.save_detector()
        
        # Test with sample data
        print(f"\n🔍 Testing with sample EXG data:")
        
        test_samples = [
            {
                'name': 'Open-like sample',
                'data': (33100, 33200, 0.465, 0.475, 1, -1),
                'expected': 'Open'
            },
            {
                'name': 'Close-like sample',
                'data': (33000, 32800, 0.460, 0.445, -1, 1),
                'expected': 'Close'
            },
            {
                'name': 'Ambiguous sample',
                'data': (33050, 33000, 0.462, 0.460, 0, 0),
                'expected': 'Unknown'
            }
        ]
        
        for sample in test_samples:
            print(f"\n📊 Testing {sample['name']}:")
            print(f"   Raw EMG: {sample['data'][:2]}")
            print(f"   Normalized: {sample['data'][2:4]}")
            print(f"   Derivatives: {sample['data'][4:6]}")
            print(f"   Expected: {sample['expected']}")
            
            result = detector.detect_pattern(*sample['data'])
            
            if result:
                print(f"   🎯 Predicted: {result['gesture']}")
                print(f"   📊 Confidence: {result['confidence']:.3f}")
                print(f"   🔍 Closer to: {result['pattern_analysis']['closer_to_pattern']} pattern")
                print(f"   📈 Open prob: {result['probabilities']['open']:.3f}")
                print(f"   📉 Close prob: {result['probabilities']['close']:.3f}")
        
        print(f"\n🎉 Pattern Detector Test Complete!")
        return True
    
    return False

if __name__ == "__main__":
    test_pattern_detector()
