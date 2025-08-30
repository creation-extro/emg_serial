#!/usr/bin/env python3
"""
Train Simple, Reliable EMG Model
Uses raw EMG channels with KNN - proven to work well
Based on your memory: "User prefers training KNN models using raw EMG channels"
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

class SimpleReliableEMGModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.gesture_names = None
        
        print("🎯 Simple Reliable EMG Model Trainer")
        print("📊 Using Raw EMG Channels + KNN")
        print("✅ Proven to Work Well")
        print("=" * 60)
    
    def load_data(self):
        """Load and prepare EMG data"""
        print("📂 Loading EMG dataset...")
        
        try:
            # Load no-RELAX dataset
            data = pd.read_csv('data/emg_data_no_relax.csv')
            print(f"✅ Loaded {len(data):,} samples")
        except FileNotFoundError:
            # Create from original data
            print("📂 Creating no-RELAX dataset...")
            original_data = pd.read_csv('data/combined_emg_data (1).csv')
            data = original_data[original_data['gesture'] != 'RELAX'].copy()
            data.to_csv('data/emg_data_no_relax.csv', index=False)
            print(f"✅ Created and loaded {len(data):,} samples")
        
        # Show gesture distribution
        print(f"\n📊 Gesture Distribution:")
        gesture_counts = data['gesture'].value_counts().sort_index()
        for gesture, count in gesture_counts.items():
            print(f"   {gesture}: {count:,} samples")
        
        return data
    
    def create_simple_features(self, data):
        """Create simple, reliable features from raw EMG"""
        print(f"\n📈 Creating Simple EMG Features...")
        
        # Use raw EMG channels directly (your preference!)
        feature_columns = ['ch1', 'ch2', 'ch3']
        
        # Simple statistical features for each channel
        features = []
        labels = []
        
        for gesture in data['gesture'].unique():
            gesture_data = data[data['gesture'] == gesture]
            
            for _, row in gesture_data.iterrows():
                # Raw EMG values
                ch1 = row['ch1']
                ch2 = row['ch2'] 
                ch3 = row['ch3']
                
                # Simple feature vector
                feature_vector = [
                    # Raw values
                    ch1, ch2, ch3,
                    
                    # Simple ratios
                    ch1 / (ch2 + 1),  # Avoid division by zero
                    ch1 / (ch3 + 1),
                    ch2 / (ch3 + 1),
                    
                    # Simple combinations
                    ch1 + ch2,
                    ch1 + ch3,
                    ch2 + ch3,
                    ch1 + ch2 + ch3,
                    
                    # Simple differences
                    abs(ch1 - ch2),
                    abs(ch1 - ch3),
                    abs(ch2 - ch3),
                    
                    # Simple statistics
                    max(ch1, ch2, ch3),
                    min(ch1, ch2, ch3),
                    np.mean([ch1, ch2, ch3]),
                    np.std([ch1, ch2, ch3])
                ]
                
                features.append(feature_vector)
                labels.append(gesture)
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"✅ Created {len(X):,} feature vectors")
        print(f"✅ Feature vector size: {X.shape[1]}")
        
        return X, y
    
    def train_model(self, X, y):
        """Train simple KNN model"""
        print(f"\n🤖 Training Simple KNN Model...")
        
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
        
        # Scale features
        print(f"\n⚖️ Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train KNN model (your preferred algorithm!)
        print(f"\n🎯 Training KNN Classifier...")
        self.model = KNeighborsClassifier(
            n_neighbors=5,  # Simple and reliable
            weights='distance',  # Weight by distance
            metric='euclidean'  # Simple distance metric
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Simple KNN Model Results:")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   Feature Count: {X.shape[1]}")
        print(f"   Algorithm: KNN (k=5)")
        
        # Detailed classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.gesture_names))
        
        # Confusion matrix
        print(f"\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print("Predicted ->")
        print("Actual ↓")
        for i, gesture in enumerate(self.gesture_names):
            print(f"{gesture:12s}: {cm[i]}")
        
        return accuracy
    
    def save_model(self, accuracy):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'gesture_names': self.gesture_names,
            'accuracy': accuracy,
            'model_type': 'Simple-KNN-Raw-EMG',
            'feature_type': 'raw_emg_simple_features',
            'feature_count': 17,
            'algorithm': 'KNeighborsClassifier',
            'parameters': {
                'n_neighbors': 5,
                'weights': 'distance',
                'metric': 'euclidean'
            }
        }
        
        filename = 'simple_reliable_emg_model.pkl'
        joblib.dump(model_data, filename)
        print(f"\n💾 Model saved: {filename}")
        print(f"📊 Model info:")
        print(f"   Type: Simple KNN with Raw EMG")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Features: 17 simple features")
        print(f"   Algorithm: KNN (k=5)")
        
        return filename
    
    def test_sample_prediction(self):
        """Test with a sample prediction"""
        print(f"\n🧪 Testing Sample Prediction...")
        
        # Your selected OK_SIGN data
        test_raw = [1216, 44154, 43322]  # Raw EMG from your data
        
        print(f"📊 Test sample: Raw EMG = {test_raw}")
        
        # Create feature vector (same as training)
        ch1, ch2, ch3 = test_raw
        
        feature_vector = [
            # Raw values
            ch1, ch2, ch3,
            
            # Simple ratios
            ch1 / (ch2 + 1),
            ch1 / (ch3 + 1),
            ch2 / (ch3 + 1),
            
            # Simple combinations
            ch1 + ch2,
            ch1 + ch3,
            ch2 + ch3,
            ch1 + ch2 + ch3,
            
            # Simple differences
            abs(ch1 - ch2),
            abs(ch1 - ch3),
            abs(ch2 - ch3),
            
            # Simple statistics
            max(ch1, ch2, ch3),
            min(ch1, ch2, ch3),
            np.mean([ch1, ch2, ch3]),
            np.std([ch1, ch2, ch3])
        ]
        
        # Scale and predict
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        prediction = self.model.predict(feature_vector_scaled)[0]
        probabilities = self.model.predict_proba(feature_vector_scaled)[0]
        
        predicted_gesture = self.gesture_names[prediction]
        confidence = probabilities[prediction]
        
        print(f"🎯 Prediction: {predicted_gesture}")
        print(f"📊 Confidence: {confidence:.3f}")
        print(f"✅ Expected: 10-OK_SIGN")
        
        # Show top 3
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        print(f"\n📊 Top 3 Predictions:")
        for i, idx in enumerate(top_3_idx):
            print(f"   {i+1}. {self.gesture_names[idx]:15s} - {probabilities[idx]:.3f}")

def main():
    """Main training function"""
    trainer = SimpleReliableEMGModel()
    
    # Load data
    data = trainer.load_data()
    
    # Create simple features
    X, y = trainer.create_simple_features(data)
    
    # Train model
    accuracy = trainer.train_model(X, y)
    
    # Save model
    if accuracy > 0.7:  # Good enough threshold
        filename = trainer.save_model(accuracy)
        
        # Test sample prediction
        trainer.test_sample_prediction()
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Simple, reliable model trained")
        print(f"📊 Accuracy: {accuracy:.4f}")
        print(f"💾 Saved as: {filename}")
        print(f"\n🚀 Now you can use this reliable model for real-time prediction!")
        
    else:
        print(f"\n⚠️  Accuracy too low: {accuracy:.4f}")
        print(f"🔧 Try adjusting parameters or features")

if __name__ == "__main__":
    main()
