#!/usr/bin/env python3
"""
Test Original Deep Learning Model with Domain-Adapted Dataset
This should be the model that showed good performance on synthetic data
"""

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import joblib

class OriginalDeepLearningTester:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.gesture_names = None
        self.preprocessing_data = None
        
    def load_original_model(self, model_filename='deep_learning_emg_model.h5', 
                           preprocessing_filename='deep_learning_emg_model_preprocessing.pkl'):
        """Load the original deep learning model"""
        print("📂 Loading original deep learning model...")
        
        if not os.path.exists(model_filename):
            print(f"❌ Model not found: {model_filename}")
            return False
        
        if not os.path.exists(preprocessing_filename):
            print(f"❌ Preprocessing data not found: {preprocessing_filename}")
            return False
        
        try:
            # Load Keras model
            self.model = keras.models.load_model(model_filename)
            print(f"✅ Deep learning model loaded: {model_filename}")
            
            # Load preprocessing data
            self.preprocessing_data = joblib.load(preprocessing_filename)
            self.scaler = self.preprocessing_data['scaler']
            self.label_encoder = self.preprocessing_data['label_encoder']
            self.gesture_names = self.preprocessing_data['gesture_names']
            
            print(f"✅ Preprocessing data loaded")
            print(f"🎯 Gestures: {len(self.gesture_names)}")
            print(f"🧠 Model input shape: {self.model.input_shape}")
            print(f"📊 Model parameters: {self.model.count_params():,}")
            print(f"🎯 Training data: {self.preprocessing_data.get('training_data', 'Unknown')}")
            print(f"📊 Test accuracy: {self.preprocessing_data.get('test_accuracy', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_domain_adapted_dataset(self, filename='domain_adapted_emg_dataset.csv'):
        """Load domain-adapted dataset"""
        print(f"\n📂 Loading domain-adapted dataset: {filename}")
        
        if not os.path.exists(filename):
            print(f"❌ Dataset not found: {filename}")
            return None
        
        try:
            df = pd.read_csv(filename)
            print(f"✅ Dataset loaded: {len(df):,} samples")
            print(f"📊 Gestures: {df['gesture'].nunique()}")
            print(f"🏷️ Gesture distribution:")
            
            gesture_counts = df['gesture'].value_counts().sort_index()
            for gesture, count in gesture_counts.items():
                print(f"   {gesture:15s}: {count:,} samples")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def create_features_from_domain_data(self, row):
        """Create features from domain-adapted data row (same as training)"""
        # Use the clean EMG values directly from the dataset
        emg1_clean = row['emg1_clean']
        emg2_clean = row['emg2_clean'] 
        emg3_clean = row['emg3_clean']
        
        # Create all features exactly as in training
        features = []
        
        # Base features
        features.extend([emg1_clean, emg2_clean, emg3_clean])
        
        # Ratios
        features.extend([
            emg1_clean / (emg2_clean + 1e-6),  # emg1_emg2_ratio
            emg1_clean / (emg3_clean + 1e-6),  # emg1_emg3_ratio
            emg2_clean / (emg3_clean + 1e-6)   # emg2_emg3_ratio
        ])
        
        # Statistical features
        emg_array = np.array([emg1_clean, emg2_clean, emg3_clean])
        features.extend([
            np.mean(emg_array),                # emg_mean
            np.std(emg_array),                 # emg_std
            np.max(emg_array),                 # emg_max
            np.min(emg_array),                 # emg_min
            np.max(emg_array) - np.min(emg_array)  # emg_range
        ])
        
        # Differences
        features.extend([
            abs(emg1_clean - emg2_clean),      # emg1_emg2_diff
            abs(emg1_clean - emg3_clean),      # emg1_emg3_diff
            abs(emg2_clean - emg3_clean)       # emg2_emg3_diff
        ])
        
        # Products (interaction terms)
        features.extend([
            emg1_clean * emg2_clean,           # emg1_emg2_product
            emg1_clean * emg3_clean,           # emg1_emg3_product
            emg2_clean * emg3_clean            # emg2_emg3_product
        ])
        
        # Power features
        features.extend([
            emg1_clean ** 2,                   # emg1_squared
            emg2_clean ** 2,                   # emg2_squared
            emg3_clean ** 2,                   # emg3_squared
            emg1_clean**2 + emg2_clean**2 + emg3_clean**2  # total_power
        ])
        
        # Subject and fatigue (default values for testing)
        features.extend([
            0.5,  # subject_encoded (default)
            1.0   # fatigue_factor (default)
        ])
        
        return np.array(features)
    
    def predict_gesture_from_domain_data(self, row):
        """Predict gesture using domain-adapted data row"""
        if self.model is None:
            return None
        
        try:
            # Create features
            features = self.create_features_from_domain_data(row)
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            pred_proba = self.model.predict(features_scaled, verbose=0)[0]
            pred_label = np.argmax(pred_proba)
            
            gesture_name = self.gesture_names[pred_label]
            confidence = pred_proba[pred_label]
            
            return {
                'gesture': gesture_name,
                'confidence': confidence,
                'probabilities': pred_proba,
                'features_count': len(features[0])
            }
            
        except Exception as e:
            print(f"❌ Deep learning prediction error: {e}")
            return None
    
    def test_specific_sample(self):
        """Test with the specific sample from user's selection"""
        print(f"\n🎯 Testing Specific Sample from User Selection:")
        print("=" * 60)

        # The sample from user's selection: 1752250562,1104,41210,39113,3-POINT,3
        # Convert to domain-adapted format
        raw_emg = [1104, 41210, 39113]

        # Normalize like domain-adapted data
        emg1_clean = (raw_emg[0] - 1000) / (15000 - 1000)
        emg2_clean = (raw_emg[1] - 15000) / (65000 - 15000)
        emg3_clean = (raw_emg[2] - 15000) / (65000 - 15000)

        # Clip to [0, 1]
        emg1_clean = np.clip(emg1_clean, 0, 1)
        emg2_clean = np.clip(emg2_clean, 0, 1)
        emg3_clean = np.clip(emg3_clean, 0, 1)

        # Create a sample row
        sample = {
            'ch1': raw_emg[0],
            'ch2': raw_emg[1],
            'ch3': raw_emg[2],
            'emg1_clean': emg1_clean,
            'emg2_clean': emg2_clean,
            'emg3_clean': emg3_clean,
            'gesture': '3-POINT',
            'sample_id': 'USER_SELECTED'
        }

        print(f"📊 User Selected Sample:")
        print(f"    Raw EMG: [{sample['ch1']:5d}, {sample['ch2']:5d}, {sample['ch3']:5d}]")
        print(f"    Clean EMG: [{sample['emg1_clean']:.3f}, {sample['emg2_clean']:.3f}, {sample['emg3_clean']:.3f}]")
        print(f"    Expected: {sample['gesture']}")

        result = self.predict_gesture_from_domain_data(sample)

        if result:
            predicted = result['gesture']
            confidence = result['confidence']
            correct = predicted == sample['gesture']

            status = "✅ CORRECT!" if correct else "❌ WRONG!"

            print(f"    🧠 Predicted: {predicted}")
            print(f"    📊 Confidence: {confidence:.3f}")
            print(f"    🎯 Result: {status}")

            # Show top 3 predictions
            probs = result['probabilities']
            top_3_idx = np.argsort(probs)[-3:][::-1]
            print(f"    📊 Top 3 predictions:")
            for j, idx in enumerate(top_3_idx):
                print(f"       {j+1}. {self.gesture_names[idx]:15s} - {probs[idx]:.3f}")

            return correct
        else:
            print(f"    ❌ Prediction failed")
            return False

    def test_domain_adapted_samples(self, df, num_samples=20):
        """Test random samples from domain-adapted dataset"""
        print(f"\n🧪 Testing {num_samples} Random Samples from Domain-Adapted Dataset:")
        print("=" * 80)
        
        # Get random samples from each gesture
        gestures = df['gesture'].unique()
        samples_per_gesture = max(1, num_samples // len(gestures))
        
        test_samples = []
        for gesture in sorted(gestures):
            gesture_data = df[df['gesture'] == gesture]
            if len(gesture_data) > 0:
                samples = gesture_data.sample(n=min(samples_per_gesture, len(gesture_data)))
                test_samples.extend(samples.to_dict('records'))
        
        # If we need more samples, add random ones
        if len(test_samples) < num_samples:
            remaining = num_samples - len(test_samples)
            additional = df.sample(n=remaining)
            test_samples.extend(additional.to_dict('records'))
        
        # Limit to requested number
        test_samples = test_samples[:num_samples]
        
        correct_count = 0
        total_confidence = 0
        
        print(f"📊 Testing {len(test_samples)} samples...")
        print()
        
        for i, sample in enumerate(test_samples, 1):
            expected_gesture = sample['gesture']
            
            print(f"{i:2d}. Sample ID: {sample['sample_id']}")
            print(f"    Raw EMG: [{sample['ch1']:5d}, {sample['ch2']:5d}, {sample['ch3']:5d}]")
            print(f"    Clean EMG: [{sample['emg1_clean']:.3f}, {sample['emg2_clean']:.3f}, {sample['emg3_clean']:.3f}]")
            print(f"    Expected: {expected_gesture}")
            
            result = self.predict_gesture_from_domain_data(sample)
            
            if result:
                predicted = result['gesture']
                confidence = result['confidence']
                correct = predicted == expected_gesture
                
                if correct:
                    correct_count += 1
                    status = "✅ CORRECT!"
                else:
                    status = "❌ WRONG!"
                
                total_confidence += confidence
                
                print(f"    🧠 Predicted: {predicted}")
                print(f"    📊 Confidence: {confidence:.3f}")
                print(f"    🎯 Result: {status}")
                
                # Show top 3 predictions
                probs = result['probabilities']
                top_3_idx = np.argsort(probs)[-3:][::-1]
                print(f"    📊 Top 3 predictions:")
                for j, idx in enumerate(top_3_idx):
                    print(f"       {j+1}. {self.gesture_names[idx]:15s} - {probs[idx]:.3f}")
            else:
                print(f"    ❌ Prediction failed")
            
            print()
        
        # Summary
        accuracy = correct_count / len(test_samples) * 100
        avg_confidence = total_confidence / len(test_samples) if len(test_samples) > 0 else 0
        
        print("🧠 Original Deep Learning Test Results Summary:")
        print("=" * 60)
        print(f"✅ Correct predictions: {correct_count}/{len(test_samples)}")
        print(f"📊 Accuracy: {accuracy:.2f}%")
        print(f"🎯 Average confidence: {avg_confidence:.3f}")
        print(f"📈 Model performance: {'EXCELLENT' if accuracy >= 90 else 'GOOD' if accuracy >= 80 else 'NEEDS IMPROVEMENT'}")
        
        return {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_samples': len(test_samples),
            'avg_confidence': avg_confidence
        }

def main():
    """Main function"""
    tester = OriginalDeepLearningTester()
    
    # Load original deep learning model
    if not tester.load_original_model():
        return
    
    # Test specific sample first
    print(f"\n🎯 Testing your specific sample first...")
    specific_correct = tester.test_specific_sample()

    # Load domain-adapted dataset
    df = tester.load_domain_adapted_dataset()
    if df is None:
        return

    # Test samples
    print(f"\n📋 Testing Options:")
    print(f"1. Test 10 random samples")
    print(f"2. Test 20 random samples")
    print(f"3. Test 50 random samples")
    print(f"4. Custom number of samples")
    print(f"5. Skip random testing")

    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        num_samples = 10
    elif choice == '2':
        num_samples = 20
    elif choice == '3':
        num_samples = 50
    elif choice == '4':
        try:
            num_samples = int(input("Enter number of samples: "))
        except ValueError:
            print("❌ Invalid number, using 20")
            num_samples = 20
    elif choice == '5':
        print(f"\n🎉 Testing completed!")
        print(f"🎯 Your specific sample result: {'✅ CORRECT' if specific_correct else '❌ WRONG'}")
        print(f"📊 This is the original deep learning model trained on domain-adapted data!")
        return
    else:
        print("❌ Invalid choice, using 20 samples")
        num_samples = 20

    # Run test
    results = tester.test_domain_adapted_samples(df, num_samples)

    print(f"\n🎉 Testing completed!")
    print(f"🎯 Your specific sample result: {'✅ CORRECT' if specific_correct else '❌ WRONG'}")
    print(f"📊 Random samples accuracy: {results['accuracy']:.2f}%")

    if results['accuracy'] > 90:
        print(f"🎉 EXCELLENT! This is the high-performance model you were looking for!")
    elif results['accuracy'] > 80:
        print(f"✅ GOOD performance - this model works well with domain-adapted data!")
    else:
        print(f"⚠️  Performance needs investigation")

if __name__ == "__main__":
    main()
