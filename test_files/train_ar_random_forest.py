#!/usr/bin/env python3
"""
Train AR Random Forest Model
Based on your memory: "AR Random Forest model with 98.92% test accuracy"
Uses Autoregressive features + Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

class ARRandomForestTrainer:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.gesture_names = None
        self.lag_order = 10  # AR lag order
        
        print("🌲 AR Random Forest Trainer")
        print("📈 Autoregressive Features + Random Forest")
        print("🎯 Target: 98.92% Accuracy")
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
        
        # Sort by timestamp for AR features
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Show gesture distribution
        print(f"\n📊 Gesture Distribution:")
        gesture_counts = data['gesture'].value_counts().sort_index()
        for gesture, count in gesture_counts.items():
            print(f"   {gesture}: {count:,} samples")
        
        return data
    
    def create_ar_features(self, data):
        """Create Autoregressive features from EMG data"""
        print(f"\n📈 Creating AR features with lag order {self.lag_order}...")
        
        feature_columns = ['emg1_clean', 'emg2_clean', 'emg3_clean']
        ar_features = []
        labels = []
        
        for gesture in data['gesture'].unique():
            gesture_data = data[data['gesture'] == gesture].reset_index(drop=True)
            
            if len(gesture_data) < self.lag_order + 10:
                continue
                
            for i in range(self.lag_order, len(gesture_data)):
                features = []
                
                # AR features for each channel
                for channel in feature_columns:
                    lagged_values = gesture_data[channel].iloc[i-self.lag_order:i].values
                    
                    # Lagged values (AR coefficients)
                    features.extend(lagged_values)
                    
                    # Statistical features of lagged values
                    features.append(np.mean(lagged_values))
                    features.append(np.std(lagged_values))
                    features.append(np.max(lagged_values))
                    features.append(np.min(lagged_values))
                    features.append(lagged_values[-1] - lagged_values[0])  # Trend
                
                # Current values
                current_values = gesture_data[feature_columns].iloc[i].values
                features.extend(current_values)
                
                # Cross-channel AR features
                for j in range(len(feature_columns)):
                    for k in range(j+1, len(feature_columns)):
                        # Correlation between channels over lag period
                        ch1_lag = gesture_data[feature_columns[j]].iloc[i-self.lag_order:i].values
                        ch2_lag = gesture_data[feature_columns[k]].iloc[i-self.lag_order:i].values
                        
                        if len(ch1_lag) > 1 and len(ch2_lag) > 1:
                            try:
                                corr = np.corrcoef(ch1_lag, ch2_lag)[0,1]
                                features.append(corr if not np.isnan(corr) else 0)
                            except:
                                features.append(0)
                        else:
                            features.append(0)
                
                # Temporal features
                timestamp = gesture_data['timestamp'].iloc[i]
                features.append(timestamp % 1000)
                features.append((timestamp // 1000) % 1000)
                
                ar_features.append(features)
                labels.append(gesture)
        
        X = np.array(ar_features)
        y = np.array(labels)
        
        print(f"✅ Created {len(X):,} AR feature vectors")
        print(f"✅ AR feature vector size: {X.shape[1]}")
        
        return X, y
    
    def train_ar_random_forest(self, X, y):
        """Train AR Random Forest with your optimal parameters"""
        print(f"\n🌲 Training AR Random Forest...")
        
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
        
        # Train AR Random Forest with your exact parameters
        print(f"\n🌲 Training AR Random Forest with your 98.92% parameters:")
        print(f"   n_estimators=400, max_depth=30, min_samples_split=5")
        print(f"   min_samples_leaf=2, max_features='sqrt'")
        
        self.model = RandomForestClassifier(
            n_estimators=400,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ AR Random Forest Results:")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   AR Feature Count: {X.shape[1]}")
        print(f"   Lag Order: {self.lag_order}")
        print(f"   Algorithm: AR Random Forest")
        print(f"   Trees: 400")
        
        # Detailed classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.gesture_names))
        
        # Show top AR features
        print(f"\n🎯 Top 10 Most Important AR Features:")
        feature_importance = self.model.feature_importances_
        top_features = np.argsort(feature_importance)[-10:][::-1]
        
        for i, feat_idx in enumerate(top_features):
            print(f"   {i+1:2d}. Feature {feat_idx:2d}: {feature_importance[feat_idx]:.4f}")
        
        return accuracy
    
    def save_model(self, accuracy):
        """Save the AR Random Forest model"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'gesture_names': self.gesture_names,
            'lag_order': self.lag_order,
            'accuracy': accuracy,
            'model_type': 'AR-Random-Forest',
            'feature_type': 'autoregressive_emg_features',
            'algorithm': 'RandomForestClassifier',
            'parameters': {
                'n_estimators': 400,
                'max_depth': 30,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt'
            }
        }
        
        filename = 'ar_random_forest_model.pkl'
        joblib.dump(model_data, filename)
        print(f"\n💾 AR Random Forest model saved: {filename}")
        print(f"📊 Model info:")
        print(f"   Type: AR Random Forest")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   AR Features: {model_data['model'].n_features_in_}")
        print(f"   Lag Order: {self.lag_order}")
        print(f"   Trees: 400")
        
        return filename
    
    def test_ar_prediction(self):
        """Test AR prediction with sample data"""
        print(f"\n🧪 Testing AR Prediction...")
        
        # Create a sample AR sequence (simulated)
        print(f"📊 Creating sample AR sequence for testing...")
        
        # Sample clean EMG values for OK_SIGN gesture
        base_emg = [0.292654267, 0.543227094, 0.530998914]
        
        # Create AR sequence with small variations
        ar_sequence = []
        for i in range(self.lag_order):
            variation = np.random.normal(0, 0.01, 3)
            varied_emg = [
                max(0, min(1, base_emg[0] + variation[0])),
                max(0, min(1, base_emg[1] + variation[1])),
                max(0, min(1, base_emg[2] + variation[2]))
            ]
            ar_sequence.append(varied_emg)
        
        # Create AR features (same as training)
        features = []
        
        # AR features for each channel
        for ch_idx in range(3):
            channel_values = np.array([seq[ch_idx] for seq in ar_sequence])
            
            # Lagged values
            features.extend(channel_values)
            
            # Statistical features
            features.append(np.mean(channel_values))
            features.append(np.std(channel_values))
            features.append(np.max(channel_values))
            features.append(np.min(channel_values))
            features.append(channel_values[-1] - channel_values[0])
        
        # Current values
        features.extend(ar_sequence[-1])
        
        # Cross-channel correlations
        for j in range(3):
            for k in range(j+1, 3):
                ch1_vals = np.array([seq[j] for seq in ar_sequence])
                ch2_vals = np.array([seq[k] for seq in ar_sequence])
                try:
                    corr = np.corrcoef(ch1_vals, ch2_vals)[0,1]
                    features.append(corr if not np.isnan(corr) else 0)
                except:
                    features.append(0)
        
        # Temporal features
        features.append(1752250605 % 1000)
        features.append((1752250605 // 1000) % 1000)
        
        # Predict
        feature_vector = np.array(features).reshape(1, -1)
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        predicted_gesture = self.gesture_names[prediction]
        confidence = probabilities[prediction]
        
        print(f"📊 AR sequence length: {len(ar_sequence)}")
        print(f"📊 AR features created: {len(features)}")
        print(f"🎯 Predicted: {predicted_gesture}")
        print(f"📊 Confidence: {confidence:.3f}")
        print(f"✅ Expected: 10-OK_SIGN (based on input)")
        
        # Show top 3
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        print(f"\n📊 Top 3 AR Predictions:")
        for i, idx in enumerate(top_3_idx):
            print(f"   {i+1}. {self.gesture_names[idx]:15s} - {probabilities[idx]:.3f}")

def main():
    """Main training function"""
    trainer = ARRandomForestTrainer()
    
    # Load data
    data = trainer.load_data()
    
    # Create AR features
    X, y = trainer.create_ar_features(data)
    
    # Train AR Random Forest
    accuracy = trainer.train_ar_random_forest(X, y)
    
    # Save model if good enough
    if accuracy > 0.8:  # 80%+ threshold
        filename = trainer.save_model(accuracy)
        
        # Test AR prediction
        trainer.test_ar_prediction()
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ AR Random Forest model trained")
        print(f"📊 Accuracy: {accuracy:.4f}")
        print(f"💾 Saved as: {filename}")
        
        if accuracy > 0.95:
            print(f"🎯 EXCELLENT! Close to your 98.92% target!")
        elif accuracy > 0.9:
            print(f"🎯 VERY GOOD! Getting close to 98.92%!")
        else:
            print(f"🎯 GOOD! Better than regular Random Forest!")
        
        print(f"\n🚀 Now you can use this AR Random Forest for real-time prediction!")
        
    else:
        print(f"\n⚠️  Accuracy still too low: {accuracy:.4f}")
        print(f"🔧 Try adjusting lag order or parameters")

if __name__ == "__main__":
    main()
