#!/usr/bin/env python3
"""
Test Optimized EMG Model with Sample Data
Compare performance and analyze data quality
"""

import pandas as pd
import numpy as np
import joblib
import os

class OptimizedModelTester:
    def __init__(self):
        self.model_data = None
        
    def load_model(self, filename='optimized_emg_model.pkl'):
        """Load the optimized model"""
        print("📂 Loading optimized model...")
        
        if not os.path.exists(filename):
            print(f"❌ Model not found: {filename}")
            return False
        
        try:
            self.model_data = joblib.load(filename)
            print(f"✅ Model loaded: {filename}")
            print(f"🎯 Model type: {self.model_data['model_type']}")
            print(f"📊 Test accuracy: {self.model_data['test_accuracy']:.4f}")
            print(f"📊 CV accuracy: {self.model_data['cv_mean']:.4f} ± {self.model_data['cv_std']:.4f}")
            print(f"🔧 Best params: {self.model_data['best_params']}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_features(self, raw_emg):
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
        
        # Create features exactly as in training
        base_features = [emg1_clean, emg2_clean, emg3_clean]
        
        # Statistical features
        emg_mean = np.mean(base_features)
        emg_std = np.std(base_features)
        emg_max = np.max(base_features)
        emg_min = np.min(base_features)
        emg_range = emg_max - emg_min
        
        # Channel differences
        emg1_emg2_diff = abs(emg1_clean - emg2_clean)
        emg1_emg3_diff = abs(emg1_clean - emg3_clean)
        emg2_emg3_diff = abs(emg2_clean - emg3_clean)
        
        # Channel ratios
        eps = 1e-6
        emg1_emg2_ratio = emg1_clean / (emg2_clean + eps)
        emg1_emg3_ratio = emg1_clean / (emg3_clean + eps)
        emg2_emg3_ratio = emg2_clean / (emg3_clean + eps)
        
        # Power features
        total_power = emg1_clean**2 + emg2_clean**2 + emg3_clean**2
        
        # Dominant channel
        dominant_channel = np.argmax(base_features)
        
        # Combine all features
        features = [
            emg1_clean, emg2_clean, emg3_clean,
            emg_mean, emg_std, emg_max, emg_min, emg_range,
            emg1_emg2_diff, emg1_emg3_diff, emg2_emg3_diff,
            emg1_emg2_ratio, emg1_emg3_ratio, emg2_emg3_ratio,
            total_power, dominant_channel
        ]
        
        return np.array(features)
    
    def predict_gesture(self, raw_emg):
        """Predict gesture using optimized model"""
        if self.model_data is None:
            return None
        
        try:
            # Create features
            features = self.create_features(raw_emg)
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.model_data['scaler'].transform(features)
            
            # Make prediction
            pred_proba = self.model_data['model'].predict_proba(features_scaled)[0]
            pred_label = self.model_data['model'].predict(features_scaled)[0]
            
            gesture_name = self.model_data['gesture_names'][pred_label]
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
    
    def test_known_samples(self):
        """Test with known samples from the selected data"""
        print("\n🧪 Testing Known Samples with Optimized Model:")
        print("=" * 70)
        
        # Test samples from the real data
        test_samples = [
            {'raw': [1104, 41210, 39113], 'expected': '3-POINT', 'name': 'POINT (from selected data)'},
            {'raw': [2320, 41290, 34440], 'expected': '0-OPEN', 'name': 'OPEN'},
            {'raw': [4273, 53196, 55885], 'expected': '0-OPEN', 'name': 'OPEN (high values)'},
            {'raw': [1216, 44874, 46635], 'expected': '0-OPEN', 'name': 'OPEN (medium values)'},
            {'raw': [912, 49051, 51676], 'expected': '0-OPEN', 'name': 'OPEN (low ch1)'},
        ]
        
        correct_count = 0
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\n{i}. Testing {sample['name']}:")
            print(f"   Raw EMG: {sample['raw']}")
            print(f"   Expected: {sample['expected']}")
            
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
                
                print(f"   🧠 Predicted: {predicted}")
                print(f"   📊 Confidence: {confidence:.3f}")
                print(f"   🎯 Result: {status}")
                
                # Show top 3
                probs = result['probabilities']
                top_3_idx = np.argsort(probs)[-3:][::-1]
                print(f"   📊 Top 3:")
                for j, idx in enumerate(top_3_idx):
                    gesture_name = self.model_data['gesture_names'][idx]
                    print(f"      {j+1}. {gesture_name:15s} - {probs[idx]:.3f}")
            else:
                print(f"   ❌ Prediction failed")
        
        accuracy = correct_count / len(test_samples) * 100
        print(f"\n🎯 Test Results:")
        print(f"   Correct: {correct_count}/{len(test_samples)}")
        print(f"   Accuracy: {accuracy:.2f}%")
        
        return accuracy
    
    def analyze_data_quality(self):
        """Analyze the quality of the training data"""
        print("\n🔍 Analyzing Data Quality:")
        print("=" * 50)
        
        # Load the real dataset
        data = pd.read_csv('data/emg_data_no_relax.csv')
        
        print(f"📊 Dataset shape: {data.shape}")
        print(f"📊 Gestures: {data['gesture'].nunique()}")
        
        # Check for data quality issues
        print(f"\n🔍 Data Quality Analysis:")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        print(f"   Missing values: {missing_values.sum()}")
        
        # Check EMG value ranges
        print(f"\n📊 EMG Value Ranges:")
        print(f"   CH1: {data['ch1'].min():,} - {data['ch1'].max():,}")
        print(f"   CH2: {data['ch2'].min():,} - {data['ch2'].max():,}")
        print(f"   CH3: {data['ch3'].min():,} - {data['ch3'].max():,}")
        
        # Check clean EMG ranges
        print(f"\n📊 Clean EMG Ranges:")
        print(f"   EMG1_clean: {data['emg1_clean'].min():.3f} - {data['emg1_clean'].max():.3f}")
        print(f"   EMG2_clean: {data['emg2_clean'].min():.3f} - {data['emg2_clean'].max():.3f}")
        print(f"   EMG3_clean: {data['emg3_clean'].min():.3f} - {data['emg3_clean'].max():.3f}")
        
        # Check gesture distribution
        print(f"\n🏷️ Gesture Distribution:")
        gesture_counts = data['gesture'].value_counts().sort_index()
        for gesture, count in gesture_counts.items():
            percentage = count / len(data) * 100
            print(f"   {gesture:15s}: {count:,} samples ({percentage:.1f}%)")
        
        # Check for potential issues
        print(f"\n⚠️  Potential Issues:")
        
        # Check if gestures are well-separated
        gesture_stats = data.groupby('gesture')[['emg1_clean', 'emg2_clean', 'emg3_clean']].mean()
        print(f"   Gesture mean EMG values:")
        for gesture in gesture_stats.index:
            emg_values = gesture_stats.loc[gesture]
            print(f"      {gesture:15s}: [{emg_values['emg1_clean']:.3f}, {emg_values['emg2_clean']:.3f}, {emg_values['emg3_clean']:.3f}]")
        
        # Check variance within gestures
        gesture_std = data.groupby('gesture')[['emg1_clean', 'emg2_clean', 'emg3_clean']].std()
        print(f"\n   Gesture EMG standard deviations:")
        for gesture in gesture_std.index:
            emg_stds = gesture_std.loc[gesture]
            print(f"      {gesture:15s}: [{emg_stds['emg1_clean']:.3f}, {emg_stds['emg2_clean']:.3f}, {emg_stds['emg3_clean']:.3f}]")

def main():
    """Main function"""
    tester = OptimizedModelTester()
    
    # Load optimized model
    if not tester.load_model():
        return
    
    # Test known samples
    accuracy = tester.test_known_samples()
    
    # Analyze data quality
    tester.analyze_data_quality()
    
    print(f"\n🎉 Testing completed!")
    print(f"📊 Sample test accuracy: {accuracy:.2f}%")
    
    if accuracy < 50:
        print(f"\n💡 Recommendations:")
        print(f"   1. Check data preprocessing and feature engineering")
        print(f"   2. Verify gesture labeling accuracy")
        print(f"   3. Consider data augmentation or collection of more diverse samples")
        print(f"   4. Try different normalization approaches")

if __name__ == "__main__":
    main()
