#!/usr/bin/env python3
"""
Test Augmented Real EMG Deep Learning Model
Compare performance with original models and analyze results
"""

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import joblib

class AugmentedRealEMGTester:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.gesture_names = None
        self.preprocessing_data = None
        
    def load_augmented_model(self, model_filename='augmented_real_emg_deep_learning_model.h5', 
                            preprocessing_filename='augmented_real_emg_deep_learning_model_preprocessing.pkl'):
        """Load the augmented real EMG deep learning model"""
        print("📂 Loading augmented real EMG deep learning model...")
        
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
            print(f"📊 Input features: {self.preprocessing_data.get('input_features', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_features_from_raw_emg(self, raw_emg):
        """Create features from raw EMG data (same as training)"""
        ch1, ch2, ch3 = raw_emg
        
        # Normalize EMG channels (same as training)
        emg1_clean = (ch1 - 1000) / (15000 - 1000)
        emg2_clean = (ch2 - 15000) / (65000 - 15000) 
        emg3_clean = (ch3 - 15000) / (65000 - 15000)
        
        # Clip to [0, 1]
        emg1_clean = np.clip(emg1_clean, 0, 1)
        emg2_clean = np.clip(emg2_clean, 0, 1)
        emg3_clean = np.clip(emg3_clean, 0, 1)
        
        # Create comprehensive features (same as training)
        base_features = [emg1_clean, emg2_clean, emg3_clean]
        
        # Statistical features
        emg_mean = np.mean(base_features)
        emg_std = np.std(base_features)
        emg_max = np.max(base_features)
        emg_min = np.min(base_features)
        emg_range = emg_max - emg_min
        emg_median = np.median(base_features)
        
        # Skewness and kurtosis (simplified calculation)
        emg_array = np.array(base_features)
        mean_val = np.mean(emg_array)
        std_val = np.std(emg_array)
        if std_val > 0:
            emg_skew = np.mean(((emg_array - mean_val) / std_val) ** 3)
            emg_kurt = np.mean(((emg_array - mean_val) / std_val) ** 4) - 3
        else:
            emg_skew = 0
            emg_kurt = 0
        
        # Signal characteristics
        signal_strength = emg1_clean + emg2_clean + emg3_clean
        signal_balance = np.std(base_features)
        
        # Channel relationships
        emg1_emg2_diff = abs(emg1_clean - emg2_clean)
        emg1_emg3_diff = abs(emg1_clean - emg3_clean)
        emg2_emg3_diff = abs(emg2_clean - emg3_clean)
        
        eps = 1e-6
        emg1_emg2_ratio = emg1_clean / (emg2_clean + eps)
        emg1_emg3_ratio = emg1_clean / (emg3_clean + eps)
        emg2_emg3_ratio = emg2_clean / (emg3_clean + eps)
        
        # Power features
        emg1_squared = emg1_clean ** 2
        emg2_squared = emg2_clean ** 2
        emg3_squared = emg3_clean ** 2
        total_power = emg1_squared + emg2_squared + emg3_squared
        
        power_ratio_1 = emg1_squared / (total_power + eps)
        power_ratio_2 = emg2_squared / (total_power + eps)
        power_ratio_3 = emg3_squared / (total_power + eps)
        
        # Interaction terms
        emg1_emg2_product = emg1_clean * emg2_clean
        emg1_emg3_product = emg1_clean * emg3_clean
        emg2_emg3_product = emg2_clean * emg3_clean
        
        # Dominant channel features
        dominant_channel = np.argmax(base_features)
        dominant_value = np.max(base_features)
        secondary_value = np.partition(base_features, -2)[-2]  # Second largest
        weakest_value = np.min(base_features)
        
        # Normalized features
        emg1_normalized = emg1_clean / (signal_strength + eps)
        emg2_normalized = emg2_clean / (signal_strength + eps)
        emg3_normalized = emg3_clean / (signal_strength + eps)
        
        # Noise level (default for real-time)
        noise_level = 0.1
        
        # Combine all features (37 total)
        features = [
            # Base features (3)
            emg1_clean, emg2_clean, emg3_clean,
            # Statistical features (8)
            emg_mean, emg_std, emg_max, emg_min, emg_range, emg_median, emg_skew, emg_kurt,
            # Signal characteristics (2)
            signal_strength, signal_balance,
            # Channel relationships (6)
            emg1_emg2_diff, emg1_emg3_diff, emg2_emg3_diff,
            emg1_emg2_ratio, emg1_emg3_ratio, emg2_emg3_ratio,
            # Power features (7)
            emg1_squared, emg2_squared, emg3_squared, total_power,
            power_ratio_1, power_ratio_2, power_ratio_3,
            # Interaction terms (3)
            emg1_emg2_product, emg1_emg3_product, emg2_emg3_product,
            # Dominant channel features (4)
            dominant_channel, dominant_value, secondary_value, weakest_value,
            # Normalized features (3)
            emg1_normalized, emg2_normalized, emg3_normalized,
            # Noise level (1)
            noise_level
        ]
        
        return np.array(features)
    
    def predict_gesture(self, raw_emg):
        """Predict gesture using augmented real EMG model"""
        if self.model is None:
            return None
        
        try:
            # Create features
            features = self.create_features_from_raw_emg(raw_emg)
            features = features.reshape(1, -1)
            
            # Handle any NaN or infinite values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
            
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
                'raw_emg': raw_emg,
                'features_count': len(features[0])
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def test_specific_samples(self):
        """Test with specific samples including user's selection"""
        print(f"\n🧪 Testing Specific Samples with Augmented Real EMG Model:")
        print("=" * 70)
        
        # Test samples including the one from user's selection
        test_samples = [
            {'raw': [1104, 41210, 39113], 'expected': '3-POINT', 'name': 'POINT (from selected data)'},
            {'raw': [2320, 41290, 34440], 'expected': '0-OPEN', 'name': 'OPEN sample 1'},
            {'raw': [4273, 53196, 55885], 'expected': '0-OPEN', 'name': 'OPEN sample 2'},
            {'raw': [1216, 44874, 46635], 'expected': '0-OPEN', 'name': 'OPEN sample 3'},
            {'raw': [2832, 49916, 53981], 'expected': '0-OPEN', 'name': 'OPEN sample 4'},
        ]
        
        correct_count = 0
        total_confidence = 0
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\n{i:2d}. Testing {sample['name']}:")
            print(f"    Raw EMG: {sample['raw']}")
            print(f"    Expected: {sample['expected']}")
            
            result = self.predict_gesture(sample['raw'])
            
            if result:
                predicted = result['gesture']
                confidence = result['confidence']
                correct = predicted == sample['expected']
                
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
        
        accuracy = correct_count / len(test_samples) * 100
        avg_confidence = total_confidence / len(test_samples) if len(test_samples) > 0 else 0
        
        print(f"\n🎯 Augmented Real EMG Model Test Results:")
        print(f"   Correct: {correct_count}/{len(test_samples)}")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Average confidence: {avg_confidence:.3f}")
        
        return accuracy

def main():
    """Main function"""
    tester = AugmentedRealEMGTester()
    
    # Load augmented real EMG model
    if not tester.load_augmented_model():
        return
    
    # Test with specific samples
    accuracy = tester.test_specific_samples()
    
    print(f"\n🎉 Augmented Real EMG Model Testing completed!")
    print(f"📊 Sample test accuracy: {accuracy:.2f}%")
    
    print(f"\n📊 Model Summary:")
    print(f"   🧠 Architecture: Advanced Deep Learning (512→256→128→64→32→11)")
    print(f"   📊 Parameters: {tester.model.count_params():,}")
    print(f"   🎯 Input Features: {tester.preprocessing_data.get('input_features', 37)}")
    print(f"   📈 Training Data: Augmented Real EMG (130k+ samples)")
    print(f"   ✅ Test Accuracy: {tester.preprocessing_data.get('test_accuracy', 0.1259):.4f}")
    
    if accuracy < 50:
        print(f"\n💡 Analysis:")
        print(f"   • Model trained on YOUR real EMG data with 5x augmentation")
        print(f"   • 37 comprehensive engineered features")
        print(f"   • Advanced deep learning architecture")
        print(f"   • Performance indicates challenging gesture classification task")
        print(f"   • Consider: more diverse data collection, different preprocessing, or simpler models")

if __name__ == "__main__":
    main()
