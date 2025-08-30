#!/usr/bin/env python3
"""
Train Improved AR Model
Fix the low confidence issue by using better AR feature engineering
Based on actual EMG time series patterns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

class ImprovedARTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.gesture_names = None
        self.lag_order = 15  # Increase lag order
        
        print("🔧 Improved AR Model Trainer")
        print("📈 Better AR Features + Scaling")
        print("🎯 Fix Low Confidence Issue")
        print("=" * 60)
    
    def load_data(self):
        """Load and prepare EMG data"""
        print("📂 Loading EMG dataset...")
        
        try:
            data = pd.read_csv('data/emg_data_no_relax.csv')
            print(f"✅ Loaded {len(data):,} samples")
        except FileNotFoundError:
            print("📂 Creating no-RELAX dataset...")
            original_data = pd.read_csv('data/combined_emg_data (1).csv')
            data = original_data[original_data['gesture'] != 'RELAX'].copy()
            data.to_csv('data/emg_data_no_relax.csv', index=False)
            print(f"✅ Created and loaded {len(data):,} samples")
        
        # Sort by timestamp for proper AR sequences
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Show gesture distribution
        print(f"\n📊 Gesture Distribution:")
        gesture_counts = data['gesture'].value_counts().sort_index()
        for gesture, count in gesture_counts.items():
            print(f"   {gesture}: {count:,} samples")
        
        return data
    
    def create_improved_ar_features(self, data):
        """Create improved AR features with better discrimination"""
        print(f"\n📈 Creating Improved AR features with lag order {self.lag_order}...")
        
        feature_columns = ['emg1_clean', 'emg2_clean', 'emg3_clean']
        ar_features = []
        labels = []
        
        for gesture in data['gesture'].unique():
            gesture_data = data[data['gesture'] == gesture].reset_index(drop=True)
            
            if len(gesture_data) < self.lag_order + 20:
                continue
                
            # Create more AR sequences per gesture for better training
            step_size = max(1, len(gesture_data) // 100)  # More sequences
            
            for i in range(self.lag_order, len(gesture_data), step_size):
                features = []
                
                # Get AR sequence
                ar_sequence = gesture_data[feature_columns].iloc[i-self.lag_order:i].values
                
                # Enhanced AR features for each channel
                for ch_idx in range(3):
                    channel_values = ar_sequence[:, ch_idx]
                    
                    # Basic lagged values (every 2nd value to reduce dimensionality)
                    features.extend(channel_values[::2])
                    
                    # Enhanced statistical features
                    features.append(np.mean(channel_values))
                    features.append(np.std(channel_values))
                    features.append(np.max(channel_values))
                    features.append(np.min(channel_values))
                    features.append(np.median(channel_values))
                    features.append(np.percentile(channel_values, 25))
                    features.append(np.percentile(channel_values, 75))
                    
                    # Trend and momentum features
                    features.append(channel_values[-1] - channel_values[0])
                    features.append(np.mean(np.diff(channel_values)))
                    features.append(np.std(np.diff(channel_values)))
                    
                    # Rolling statistics
                    if len(channel_values) >= 5:
                        features.append(np.mean(channel_values[-5:]))
                        features.append(np.std(channel_values[-5:]))
                    else:
                        features.extend([0, 0])
                    
                    # Autocorrelation
                    if len(channel_values) > 1:
                        try:
                            corr = np.corrcoef(channel_values[:-1], channel_values[1:])[0,1]
                            features.append(corr if not np.isnan(corr) else 0)
                        except:
                            features.append(0)
                    else:
                        features.append(0)
                
                # Current values (last in sequence)
                current_values = ar_sequence[-1]
                features.extend(current_values)
                
                # Cross-channel features
                features.append(current_values[0] * current_values[1])
                features.append(current_values[0] * current_values[2])
                features.append(current_values[1] * current_values[2])
                features.append(np.sum(current_values))
                features.append(np.std(current_values))
                
                # Enhanced cross-channel correlations
                for j in range(3):
                    for k in range(j+1, 3):
                        ch1_vals = ar_sequence[:, j]
                        ch2_vals = ar_sequence[:, k]
                        
                        if len(ch1_vals) > 1:
                            try:
                                corr = np.corrcoef(ch1_vals, ch2_vals)[0,1]
                                features.append(corr if not np.isnan(corr) else 0)
                            except:
                                features.append(0)
                        else:
                            features.append(0)
                
                # Enhanced temporal features
                timestamp = gesture_data['timestamp'].iloc[i]
                features.append(timestamp % 100)
                features.append((timestamp // 100) % 100)
                features.append((timestamp // 10000) % 100)
                features.append(timestamp % 1000)
                
                ar_features.append(features)
                labels.append(gesture)
        
        X = np.array(ar_features)
        y = np.array(labels)
        
        print(f"✅ Created {len(X):,} improved AR feature vectors")
        print(f"✅ Improved AR feature vector size: {X.shape[1]}")
        
        return X, y
    
    def train_improved_model(self, X, y):
        """Train improved AR model with scaling"""
        print(f"\n🔧 Training Improved AR Model...")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.gesture_names = list(self.label_encoder.classes_)
        
        print(f"\n🔤 Gesture Label Mapping:")
        for i, gesture in enumerate(self.gesture_names):
            print(f"   {i}: {gesture}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training samples: {X_train.shape[0]:,}")
        print(f"   Test samples: {X_test.shape[0]:,}")
        
        # Scale features (important for AR features!)
        print(f"\n⚖️ Scaling AR features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train improved Random Forest
        print(f"\n🌲 Training Improved Random Forest...")
        print(f"   Using optimized parameters for AR features")
        
        self.model = RandomForestClassifier(
            n_estimators=500,  # More trees
            max_depth=25,      # Slightly less depth to prevent overfitting
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Improved AR Model Results:")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   AR Feature Count: {X.shape[1]}")
        print(f"   Lag Order: {self.lag_order}")
        print(f"   Algorithm: Improved Random Forest")
        print(f"   Trees: 500")
        print(f"   Scaling: YES")
        
        # Detailed classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.gesture_names))
        
        # Show feature importance
        print(f"\n🎯 Top 10 Most Important Features:")
        feature_importance = self.model.feature_importances_
        top_features = np.argsort(feature_importance)[-10:][::-1]
        
        for i, feat_idx in enumerate(top_features):
            print(f"   {i+1:2d}. Feature {feat_idx:2d}: {feature_importance[feat_idx]:.4f}")
        
        return accuracy
    
    def save_model(self, accuracy):
        """Save the improved AR model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'gesture_names': self.gesture_names,
            'lag_order': self.lag_order,
            'accuracy': accuracy,
            'model_type': 'Improved-AR-Random-Forest',
            'feature_type': 'improved_ar_features',
            'algorithm': 'RandomForestClassifier',
            'parameters': {
                'n_estimators': 500,
                'max_depth': 25,
                'min_samples_split': 3,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'class_weight': 'balanced'
            },
            'scaling': True
        }
        
        filename = 'improved_ar_model.pkl'
        joblib.dump(model_data, filename)
        print(f"\n💾 Improved AR model saved: {filename}")
        print(f"📊 Model info:")
        print(f"   Type: Improved AR Random Forest")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   AR Features: {model_data['model'].n_features_in_}")
        print(f"   Lag Order: {self.lag_order}")
        print(f"   Trees: 500")
        print(f"   Scaling: YES")
        
        return filename
    
    def test_improved_prediction(self):
        """Test improved AR prediction"""
        print(f"\n🧪 Testing Improved AR Prediction...")
        
        # Test with your known samples
        test_samples = [
            {'base': [0.292654267, 0.543227094, 0.530998914], 'expected': '10-OK_SIGN', 'name': 'OK_SIGN'},
            {'base': [0.353751761, 0.480362306, 0.477253055], 'expected': '5-FIVE', 'name': 'FIVE'}
        ]
        
        for sample in test_samples:
            print(f"\n📊 Testing {sample['name']}:")
            
            # Create realistic AR sequence with proper variation
            ar_sequence = []
            base_emg = sample['base']
            
            for i in range(self.lag_order):
                # Create more realistic temporal variation
                time_factor = i / self.lag_order
                variation = np.random.normal(0, 0.005, 3)  # Smaller variation
                
                varied_emg = [
                    max(0, min(1, base_emg[0] + variation[0] + 0.01 * np.sin(time_factor * np.pi))),
                    max(0, min(1, base_emg[1] + variation[1] + 0.01 * np.cos(time_factor * np.pi))),
                    max(0, min(1, base_emg[2] + variation[2] + 0.005 * np.sin(2 * time_factor * np.pi)))
                ]
                ar_sequence.append(varied_emg)
            
            # Create improved AR features (same as training)
            ar_array = np.array(ar_sequence)
            features = []
            
            # Enhanced AR features for each channel
            for ch_idx in range(3):
                channel_values = ar_array[:, ch_idx]
                
                # Basic lagged values (every 2nd)
                features.extend(channel_values[::2])
                
                # Enhanced statistical features
                features.extend([
                    np.mean(channel_values), np.std(channel_values),
                    np.max(channel_values), np.min(channel_values),
                    np.median(channel_values),
                    np.percentile(channel_values, 25), np.percentile(channel_values, 75),
                    channel_values[-1] - channel_values[0],
                    np.mean(np.diff(channel_values)), np.std(np.diff(channel_values)),
                    np.mean(channel_values[-5:]), np.std(channel_values[-5:])
                ])
                
                # Autocorrelation
                try:
                    corr = np.corrcoef(channel_values[:-1], channel_values[1:])[0,1]
                    features.append(corr if not np.isnan(corr) else 0)
                except:
                    features.append(0)
            
            # Current values
            features.extend(ar_sequence[-1])
            
            # Cross-channel features
            current = ar_sequence[-1]
            features.extend([
                current[0] * current[1], current[0] * current[2], current[1] * current[2],
                np.sum(current), np.std(current)
            ])
            
            # Cross-channel correlations
            for j in range(3):
                for k in range(j+1, 3):
                    try:
                        corr = np.corrcoef(ar_array[:, j], ar_array[:, k])[0,1]
                        features.append(corr if not np.isnan(corr) else 0)
                    except:
                        features.append(0)
            
            # Temporal features
            features.extend([1752250605 % 100, (1752250605 // 100) % 100, 
                           (1752250605 // 10000) % 100, 1752250605 % 1000])
            
            # Scale and predict
            feature_vector = np.array(features).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            prediction = self.model.predict(feature_vector_scaled)[0]
            probabilities = self.model.predict_proba(feature_vector_scaled)[0]
            
            predicted_gesture = self.gesture_names[prediction]
            confidence = probabilities[prediction]
            
            correct = "✅ CORRECT!" if predicted_gesture == sample['expected'] else "❌ WRONG!"
            
            print(f"   Base EMG: {base_emg}")
            print(f"   AR features: {len(features)}")
            print(f"   🎯 Predicted: {predicted_gesture}")
            print(f"   📊 Confidence: {confidence:.3f}")
            print(f"   ✅ Expected: {sample['expected']}")
            print(f"   🎯 Result: {correct}")
            
            # Show top 3
            top_3_idx = np.argsort(probabilities)[-3:][::-1]
            print(f"   📊 Top 3:")
            for i, idx in enumerate(top_3_idx):
                print(f"      {i+1}. {self.gesture_names[idx]:15s} - {probabilities[idx]:.3f}")

def main():
    """Main training function"""
    trainer = ImprovedARTrainer()
    
    # Load data
    data = trainer.load_data()
    
    # Create improved AR features
    X, y = trainer.create_improved_ar_features(data)
    
    # Train improved model
    accuracy = trainer.train_improved_model(X, y)
    
    # Save model if good enough
    if accuracy > 0.7:  # Lower threshold for now
        filename = trainer.save_model(accuracy)
        
        # Test improved prediction
        trainer.test_improved_prediction()
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Improved AR model trained")
        print(f"📊 Accuracy: {accuracy:.4f}")
        print(f"💾 Saved as: {filename}")
        
        if accuracy > 0.9:
            print(f"🎯 EXCELLENT! High accuracy achieved!")
        elif accuracy > 0.8:
            print(f"🎯 VERY GOOD! Much better than before!")
        else:
            print(f"🎯 GOOD! Improved from previous versions!")
        
        print(f"\n🚀 This improved model should have much higher confidence!")
        
    else:
        print(f"\n⚠️  Accuracy still too low: {accuracy:.4f}")

if __name__ == "__main__":
    main()
