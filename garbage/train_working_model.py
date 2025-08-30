#!/usr/bin/env python3
"""
Train Working EMG Model
Based on your memory: "AR Random Forest model with 98.92% test accuracy"
Uses better features and Random Forest algorithm
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

class WorkingEMGModel:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.gesture_names = None
        
        print("🌲 Working EMG Model Trainer")
        print("📊 Using Random Forest (Your 98.92% Model)")
        print("⚡ Better Feature Engineering")
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
    
    def create_better_features(self, data):
        """Create better discriminative features"""
        print(f"\n📈 Creating Better EMG Features...")
        
        features = []
        labels = []
        
        for gesture in data['gesture'].unique():
            gesture_data = data[data['gesture'] == gesture]
            
            for _, row in gesture_data.iterrows():
                # Raw EMG values
                ch1 = row['ch1']
                ch2 = row['ch2'] 
                ch3 = row['ch3']
                
                # Clean EMG values (normalized)
                emg1_clean = row['emg1_clean']
                emg2_clean = row['emg2_clean']
                emg3_clean = row['emg3_clean']
                
                # Enhanced feature vector
                feature_vector = [
                    # Raw EMG channels
                    ch1, ch2, ch3,
                    
                    # Clean EMG channels (normalized)
                    emg1_clean, emg2_clean, emg3_clean,
                    
                    # Raw channel ratios
                    ch1 / (ch2 + 1),
                    ch1 / (ch3 + 1),
                    ch2 / (ch3 + 1),
                    
                    # Clean channel ratios
                    emg1_clean / (emg2_clean + 0.001),
                    emg1_clean / (emg3_clean + 0.001),
                    emg2_clean / (emg3_clean + 0.001),
                    
                    # Raw statistics
                    max(ch1, ch2, ch3),
                    min(ch1, ch2, ch3),
                    np.mean([ch1, ch2, ch3]),
                    np.std([ch1, ch2, ch3]),
                    
                    # Clean statistics
                    max(emg1_clean, emg2_clean, emg3_clean),
                    min(emg1_clean, emg2_clean, emg3_clean),
                    np.mean([emg1_clean, emg2_clean, emg3_clean]),
                    np.std([emg1_clean, emg2_clean, emg3_clean]),
                    
                    # Raw differences
                    abs(ch1 - ch2),
                    abs(ch1 - ch3),
                    abs(ch2 - ch3),
                    
                    # Clean differences
                    abs(emg1_clean - emg2_clean),
                    abs(emg1_clean - emg3_clean),
                    abs(emg2_clean - emg3_clean),
                    
                    # Raw combinations
                    ch1 + ch2,
                    ch1 + ch3,
                    ch2 + ch3,
                    ch1 + ch2 + ch3,
                    
                    # Clean combinations
                    emg1_clean + emg2_clean,
                    emg1_clean + emg3_clean,
                    emg2_clean + emg3_clean,
                    emg1_clean + emg2_clean + emg3_clean,
                    
                    # Advanced features
                    ch1 * ch2 / (ch3 + 1),
                    ch1 * ch3 / (ch2 + 1),
                    ch2 * ch3 / (ch1 + 1),
                    
                    # Range features
                    max(ch1, ch2, ch3) - min(ch1, ch2, ch3),
                    max(emg1_clean, emg2_clean, emg3_clean) - min(emg1_clean, emg2_clean, emg3_clean),
                    
                    # Timestamp features (if available)
                    row.get('timestamp', 0) % 1000,
                    (row.get('timestamp', 0) // 1000) % 1000,
                ]
                
                features.append(feature_vector)
                labels.append(gesture)
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"✅ Created {len(X):,} feature vectors")
        print(f"✅ Feature vector size: {X.shape[1]}")
        
        return X, y
    
    def train_model(self, X, y):
        """Train Random Forest model with your optimal parameters"""
        print(f"\n🌲 Training Random Forest Model...")
        
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
        
        # Train Random Forest with your optimal parameters
        print(f"\n🌲 Training Random Forest with optimal parameters...")
        print(f"   Using your 98.92% accuracy parameters:")
        print(f"   n_estimators=400, max_depth=30, min_samples_split=5")
        
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
        
        print(f"\n✅ Random Forest Model Results:")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   Feature Count: {X.shape[1]}")
        print(f"   Algorithm: Random Forest")
        print(f"   Trees: 400")
        
        # Detailed classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.gesture_names))
        
        # Show feature importance
        print(f"\n🎯 Top 10 Most Important Features:")
        feature_importance = self.model.feature_importances_
        top_features = np.argsort(feature_importance)[-10:][::-1]
        
        feature_names = [
            'ch1', 'ch2', 'ch3', 'emg1_clean', 'emg2_clean', 'emg3_clean',
            'ch1/ch2', 'ch1/ch3', 'ch2/ch3', 'emg1/emg2', 'emg1/emg3', 'emg2/emg3',
            'max_raw', 'min_raw', 'mean_raw', 'std_raw',
            'max_clean', 'min_clean', 'mean_clean', 'std_clean',
            'diff_ch1_ch2', 'diff_ch1_ch3', 'diff_ch2_ch3',
            'diff_emg1_emg2', 'diff_emg1_emg3', 'diff_emg2_emg3',
            'sum_ch1_ch2', 'sum_ch1_ch3', 'sum_ch2_ch3', 'sum_all_raw',
            'sum_emg1_emg2', 'sum_emg1_emg3', 'sum_emg2_emg3', 'sum_all_clean',
            'prod_ch1_ch2_ch3', 'prod_ch1_ch3_ch2', 'prod_ch2_ch3_ch1',
            'range_raw', 'range_clean', 'timestamp_mod1000', 'timestamp_div1000'
        ]
        
        for i, feat_idx in enumerate(top_features):
            if feat_idx < len(feature_names):
                print(f"   {i+1:2d}. {feature_names[feat_idx]:15s}: {feature_importance[feat_idx]:.4f}")
        
        return accuracy
    
    def save_model(self, accuracy):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'gesture_names': self.gesture_names,
            'accuracy': accuracy,
            'model_type': 'Working-Random-Forest',
            'feature_type': 'enhanced_emg_features',
            'feature_count': 41,
            'algorithm': 'RandomForestClassifier',
            'parameters': {
                'n_estimators': 400,
                'max_depth': 30,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt'
            }
        }
        
        filename = 'working_emg_model.pkl'
        joblib.dump(model_data, filename)
        print(f"\n💾 Model saved: {filename}")
        print(f"📊 Model info:")
        print(f"   Type: Random Forest with Enhanced Features")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Features: 41 enhanced features")
        print(f"   Trees: 400")
        
        return filename
    
    def test_sample_prediction(self):
        """Test with sample predictions"""
        print(f"\n🧪 Testing Sample Predictions...")
        
        # Your selected samples with both raw and clean data
        test_samples = [
            {
                'name': 'OK_SIGN',
                'raw': [1216, 44154, 43322],
                'clean': [0.292654267, 0.543227094, 0.530998914],
                'expected': '10-OK_SIGN'
            },
            {
                'name': 'PEACE',
                'raw': [8930, 52764, 53100],
                'clean': [0.469815094, 0.613435699, 0.62751976],
                'expected': '6-PEACE'
            }
        ]
        
        for sample in test_samples:
            print(f"\n📊 Testing {sample['name']}:")
            print(f"   Raw: {sample['raw']}")
            print(f"   Clean: {sample['clean']}")
            print(f"   Expected: {sample['expected']}")
            
            # Create feature vector (same as training)
            ch1, ch2, ch3 = sample['raw']
            emg1_clean, emg2_clean, emg3_clean = sample['clean']
            
            feature_vector = [
                ch1, ch2, ch3, emg1_clean, emg2_clean, emg3_clean,
                ch1/(ch2+1), ch1/(ch3+1), ch2/(ch3+1),
                emg1_clean/(emg2_clean+0.001), emg1_clean/(emg3_clean+0.001), emg2_clean/(emg3_clean+0.001),
                max(ch1,ch2,ch3), min(ch1,ch2,ch3), np.mean([ch1,ch2,ch3]), np.std([ch1,ch2,ch3]),
                max(emg1_clean,emg2_clean,emg3_clean), min(emg1_clean,emg2_clean,emg3_clean), 
                np.mean([emg1_clean,emg2_clean,emg3_clean]), np.std([emg1_clean,emg2_clean,emg3_clean]),
                abs(ch1-ch2), abs(ch1-ch3), abs(ch2-ch3),
                abs(emg1_clean-emg2_clean), abs(emg1_clean-emg3_clean), abs(emg2_clean-emg3_clean),
                ch1+ch2, ch1+ch3, ch2+ch3, ch1+ch2+ch3,
                emg1_clean+emg2_clean, emg1_clean+emg3_clean, emg2_clean+emg3_clean, emg1_clean+emg2_clean+emg3_clean,
                ch1*ch2/(ch3+1), ch1*ch3/(ch2+1), ch2*ch3/(ch1+1),
                max(ch1,ch2,ch3)-min(ch1,ch2,ch3), max(emg1_clean,emg2_clean,emg3_clean)-min(emg1_clean,emg2_clean,emg3_clean),
                1752250605 % 1000, (1752250605 // 1000) % 1000
            ]
            
            # Predict
            feature_vector = np.array(feature_vector).reshape(1, -1)
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]
            
            predicted_gesture = self.gesture_names[prediction]
            confidence = probabilities[prediction]
            
            correct = "✅ CORRECT!" if predicted_gesture == sample['expected'] else "❌ WRONG!"
            
            print(f"   🎯 Predicted: {predicted_gesture}")
            print(f"   📊 Confidence: {confidence:.3f}")
            print(f"   🎯 Result: {correct}")

def main():
    """Main training function"""
    trainer = WorkingEMGModel()
    
    # Load data
    data = trainer.load_data()
    
    # Create better features
    X, y = trainer.create_better_features(data)
    
    # Train model
    accuracy = trainer.train_model(X, y)
    
    # Save model if good enough
    if accuracy > 0.8:  # 80%+ threshold
        filename = trainer.save_model(accuracy)
        
        # Test sample predictions
        trainer.test_sample_prediction()
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Working Random Forest model trained")
        print(f"📊 Accuracy: {accuracy:.4f}")
        print(f"💾 Saved as: {filename}")
        print(f"\n🚀 This should work much better for real-time prediction!")
        
    else:
        print(f"\n⚠️  Accuracy still too low: {accuracy:.4f}")
        print(f"🔧 Need to investigate data quality")

if __name__ == "__main__":
    main()
