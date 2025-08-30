#!/usr/bin/env python3
"""
Train Simple Working Model
Go back to basics - use clean EMG features directly
No AR, no complex features - just what works
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

class SimpleWorkingTrainer:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.gesture_names = None
        
        print("✅ Simple Working Model Trainer")
        print("📊 Using Clean EMG Features Directly")
        print("🎯 Focus on What Actually Works")
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
        
        # Show gesture distribution
        print(f"\n📊 Gesture Distribution:")
        gesture_counts = data['gesture'].value_counts().sort_index()
        for gesture, count in gesture_counts.items():
            print(f"   {gesture}: {count:,} samples")
        
        return data
    
    def create_simple_features(self, data):
        """Create simple, working features"""
        print(f"\n📈 Creating Simple Working Features...")
        
        features = []
        labels = []
        
        for _, row in data.iterrows():
            # Use the clean EMG features directly (they're already in the dataset!)
            emg1_clean = row['emg1_clean']
            emg2_clean = row['emg2_clean']
            emg3_clean = row['emg3_clean']
            
            # Also use raw values for additional info
            ch1 = row['ch1']
            ch2 = row['ch2']
            ch3 = row['ch3']
            
            # Simple but effective feature vector
            feature_vector = [
                # Clean EMG values (most important!)
                emg1_clean,
                emg2_clean,
                emg3_clean,
                
                # Raw EMG values
                ch1,
                ch2,
                ch3,
                
                # Simple ratios of clean values
                emg1_clean / (emg2_clean + 0.001),
                emg1_clean / (emg3_clean + 0.001),
                emg2_clean / (emg3_clean + 0.001),
                
                # Simple ratios of raw values
                ch1 / (ch2 + 1),
                ch1 / (ch3 + 1),
                ch2 / (ch3 + 1),
                
                # Simple statistics
                max(emg1_clean, emg2_clean, emg3_clean),
                min(emg1_clean, emg2_clean, emg3_clean),
                np.mean([emg1_clean, emg2_clean, emg3_clean]),
                np.std([emg1_clean, emg2_clean, emg3_clean]),
                
                # Simple differences
                abs(emg1_clean - emg2_clean),
                abs(emg1_clean - emg3_clean),
                abs(emg2_clean - emg3_clean),
                
                # Simple products
                emg1_clean * emg2_clean,
                emg1_clean * emg3_clean,
                emg2_clean * emg3_clean,
                
                # Range
                max(emg1_clean, emg2_clean, emg3_clean) - min(emg1_clean, emg2_clean, emg3_clean),
            ]
            
            features.append(feature_vector)
            labels.append(row['gesture'])
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"✅ Created {len(X):,} simple feature vectors")
        print(f"✅ Simple feature vector size: {X.shape[1]}")
        
        return X, y
    
    def train_simple_model(self, X, y):
        """Train simple Random Forest model"""
        print(f"\n🌲 Training Simple Random Forest...")
        
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
        
        # Train simple Random Forest
        print(f"\n🌲 Training Simple Random Forest...")
        print(f"   Using simple, reliable parameters")
        
        self.model = RandomForestClassifier(
            n_estimators=100,  # Fewer trees
            max_depth=15,      # Shallower trees
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Simple Model Results:")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   Feature Count: {X.shape[1]}")
        print(f"   Algorithm: Simple Random Forest")
        print(f"   Trees: 100")
        print(f"   Scaling: NO (not needed)")
        
        # Detailed classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.gesture_names))
        
        # Confusion matrix
        print(f"\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print("Predicted ->")
        print("Actual ↓")
        for i, gesture in enumerate(self.gesture_names):
            correct = cm[i, i]
            total = np.sum(cm[i, :])
            accuracy_per_class = correct / total if total > 0 else 0
            print(f"{gesture:12s}: {correct:3d}/{total:3d} = {accuracy_per_class:.3f}")
        
        # Show feature importance
        print(f"\n🎯 Top 10 Most Important Features:")
        feature_importance = self.model.feature_importances_
        top_features = np.argsort(feature_importance)[-10:][::-1]
        
        feature_names = [
            'emg1_clean', 'emg2_clean', 'emg3_clean',
            'ch1', 'ch2', 'ch3',
            'emg1/emg2', 'emg1/emg3', 'emg2/emg3',
            'ch1/ch2', 'ch1/ch3', 'ch2/ch3',
            'max_clean', 'min_clean', 'mean_clean', 'std_clean',
            'diff_emg1_emg2', 'diff_emg1_emg3', 'diff_emg2_emg3',
            'prod_emg1_emg2', 'prod_emg1_emg3', 'prod_emg2_emg3',
            'range_clean'
        ]
        
        for i, feat_idx in enumerate(top_features):
            if feat_idx < len(feature_names):
                print(f"   {i+1:2d}. {feature_names[feat_idx]:15s}: {feature_importance[feat_idx]:.4f}")
        
        return accuracy
    
    def save_model(self, accuracy):
        """Save the simple working model"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'gesture_names': self.gesture_names,
            'accuracy': accuracy,
            'model_type': 'Simple-Working-Random-Forest',
            'feature_type': 'simple_clean_emg_features',
            'algorithm': 'RandomForestClassifier',
            'parameters': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt'
            },
            'scaling': False
        }
        
        filename = 'simple_working_model.pkl'
        joblib.dump(model_data, filename)
        print(f"\n💾 Simple working model saved: {filename}")
        print(f"📊 Model info:")
        print(f"   Type: Simple Random Forest")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Features: {model_data['model'].n_features_in_}")
        print(f"   Trees: 100")
        print(f"   Scaling: NO")
        
        return filename
    
    def test_simple_prediction(self):
        """Test simple prediction"""
        print(f"\n🧪 Testing Simple Prediction...")
        
        # Test with your known samples
        test_samples = [
            {
                'name': 'OK_SIGN',
                'clean': [0.292654267, 0.543227094, 0.530998914],
                'raw': [1216, 44154, 43322],
                'expected': '10-OK_SIGN'
            },
            {
                'name': 'FIVE',
                'clean': [0.353751761, 0.480362306, 0.477253055],
                'raw': [3456, 48234, 47892],
                'expected': '5-FIVE'
            }
        ]
        
        for sample in test_samples:
            print(f"\n📊 Testing {sample['name']}:")
            
            emg1_clean, emg2_clean, emg3_clean = sample['clean']
            ch1, ch2, ch3 = sample['raw']
            
            # Create simple features (same as training)
            feature_vector = [
                emg1_clean, emg2_clean, emg3_clean,
                ch1, ch2, ch3,
                emg1_clean/(emg2_clean+0.001), emg1_clean/(emg3_clean+0.001), emg2_clean/(emg3_clean+0.001),
                ch1/(ch2+1), ch1/(ch3+1), ch2/(ch3+1),
                max(emg1_clean,emg2_clean,emg3_clean), min(emg1_clean,emg2_clean,emg3_clean),
                np.mean([emg1_clean,emg2_clean,emg3_clean]), np.std([emg1_clean,emg2_clean,emg3_clean]),
                abs(emg1_clean-emg2_clean), abs(emg1_clean-emg3_clean), abs(emg2_clean-emg3_clean),
                emg1_clean*emg2_clean, emg1_clean*emg3_clean, emg2_clean*emg3_clean,
                max(emg1_clean,emg2_clean,emg3_clean)-min(emg1_clean,emg2_clean,emg3_clean)
            ]
            
            # Predict
            feature_vector = np.array(feature_vector).reshape(1, -1)
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]
            
            predicted_gesture = self.gesture_names[prediction]
            confidence = probabilities[prediction]
            
            correct = "✅ CORRECT!" if predicted_gesture == sample['expected'] else "❌ WRONG!"
            
            print(f"   Clean EMG: {sample['clean']}")
            print(f"   Raw EMG: {sample['raw']}")
            print(f"   Features: {len(feature_vector[0])}")
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
    trainer = SimpleWorkingTrainer()
    
    # Load data
    data = trainer.load_data()
    
    # Create simple features
    X, y = trainer.create_simple_features(data)
    
    # Train simple model
    accuracy = trainer.train_simple_model(X, y)
    
    # Save model if it works
    if accuracy > 0.5:  # 50%+ threshold (realistic)
        filename = trainer.save_model(accuracy)
        
        # Test simple prediction
        trainer.test_simple_prediction()
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Simple working model trained")
        print(f"📊 Accuracy: {accuracy:.4f}")
        print(f"💾 Saved as: {filename}")
        
        if accuracy > 0.8:
            print(f"🎯 EXCELLENT! High accuracy achieved!")
        elif accuracy > 0.7:
            print(f"🎯 VERY GOOD! Good accuracy!")
        elif accuracy > 0.6:
            print(f"🎯 GOOD! Decent accuracy!")
        else:
            print(f"🎯 OK! At least it works!")
        
        print(f"\n🚀 This simple model should actually work for real-time prediction!")
        
    else:
        print(f"\n⚠️  Accuracy too low: {accuracy:.4f}")
        print(f"🔧 There might be a fundamental issue with the dataset")

if __name__ == "__main__":
    main()
