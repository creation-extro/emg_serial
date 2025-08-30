#!/usr/bin/env python3
"""
Test AR Random Forest Model with Sample Data
Use the existing high-accuracy AR model for testing
"""

import pandas as pd
import numpy as np
import joblib
import os

class ARModelTester:
    def __init__(self):
        self.model_data = None
        
    def load_ar_model(self, filename='ar_random_forest_model.pkl'):
        """Load the AR Random Forest model"""
        print("📂 Loading AR Random Forest model...")
        
        if not os.path.exists(filename):
            print(f"❌ Model not found: {filename}")
            return False
        
        try:
            self.model_data = joblib.load(filename)
            print(f"✅ Model loaded: {filename}")
            
            # Check what's in the model data
            if isinstance(self.model_data, dict):
                print(f"🎯 Model type: {self.model_data.get('model_type', 'Unknown')}")
                if 'test_accuracy' in self.model_data:
                    print(f"📊 Test accuracy: {self.model_data['test_accuracy']:.4f}")
                if 'cv_mean' in self.model_data:
                    print(f"📊 CV accuracy: {self.model_data['cv_mean']:.4f}")
                print(f"🔧 Available keys: {list(self.model_data.keys())}")
            else:
                print(f"🎯 Model type: Direct model object")
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_gesture(self, raw_emg):
        """Predict gesture using AR model"""
        if self.model_data is None:
            return None
        
        try:
            # Handle different model formats
            if isinstance(self.model_data, dict):
                model = self.model_data.get('model')
                scaler = self.model_data.get('scaler')
                label_encoder = self.model_data.get('label_encoder')
                gesture_names = self.model_data.get('gesture_names', self.model_data.get('classes'))
            else:
                # Direct model object
                model = self.model_data
                scaler = None
                label_encoder = None
                gesture_names = None
            
            if model is None:
                print("❌ No model found in loaded data")
                return None
            
            # Create features from raw EMG
            features = self.create_ar_features(raw_emg)
            features = features.reshape(1, -1)
            
            # Scale features if scaler is available
            if scaler is not None:
                features = scaler.transform(features)
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(features)[0]
                pred_label = model.predict(features)[0]
                confidence = pred_proba[pred_label]
            else:
                pred_label = model.predict(features)[0]
                pred_proba = None
                confidence = 1.0
            
            # Get gesture name
            if gesture_names is not None:
                if label_encoder is not None:
                    gesture_name = gesture_names[pred_label]
                else:
                    gesture_name = gesture_names[pred_label] if pred_label < len(gesture_names) else f"Class_{pred_label}"
            else:
                gesture_name = f"Class_{pred_label}"
            
            return {
                'gesture': gesture_name,
                'confidence': confidence,
                'probabilities': pred_proba,
                'raw_emg': raw_emg,
                'features_count': len(features[0])
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_ar_features(self, raw_emg):
        """Create AR features from raw EMG data"""
        ch1, ch2, ch3 = raw_emg
        
        # Simple approach: use raw channels directly (as mentioned in memories)
        # This is what works best according to user preferences
        features = [ch1, ch2, ch3]
        
        # Add some basic derived features
        features.extend([
            ch1 + ch2 + ch3,  # total
            max(ch1, ch2, ch3),  # max
            min(ch1, ch2, ch3),  # min
            abs(ch1 - ch2),  # diff 1-2
            abs(ch1 - ch3),  # diff 1-3
            abs(ch2 - ch3),  # diff 2-3
        ])
        
        return np.array(features)
    
    def test_sample_data(self):
        """Test with sample data from the real dataset"""
        print("\n🧪 Testing Sample Data with AR Random Forest Model:")
        print("=" * 70)
        
        # Load some real samples from the dataset
        try:
            data = pd.read_csv('data/emg_data_no_relax.csv')
            print(f"📂 Loaded {len(data)} samples from real dataset")
            
            # Get a few samples from each gesture
            test_samples = []
            for gesture in data['gesture'].unique()[:5]:  # Test first 5 gestures
                gesture_data = data[data['gesture'] == gesture].head(2)  # 2 samples per gesture
                for _, row in gesture_data.iterrows():
                    test_samples.append({
                        'raw': [int(row['ch1']), int(row['ch2']), int(row['ch3'])],
                        'expected': row['gesture'],
                        'name': f"{row['gesture']} sample"
                    })
            
            correct_count = 0
            
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
                    
                    print(f"    🧠 Predicted: {predicted}")
                    print(f"    📊 Confidence: {confidence:.3f}")
                    print(f"    🎯 Result: {status}")
                    
                    # Show top 3 if probabilities available
                    if result['probabilities'] is not None:
                        probs = result['probabilities']
                        top_3_idx = np.argsort(probs)[-3:][::-1]
                        print(f"    📊 Top 3:")
                        for j, idx in enumerate(top_3_idx):
                            if isinstance(self.model_data, dict) and 'gesture_names' in self.model_data:
                                gesture_name = self.model_data['gesture_names'][idx]
                            else:
                                gesture_name = f"Class_{idx}"
                            print(f"       {j+1}. {gesture_name:15s} - {probs[idx]:.3f}")
                else:
                    print(f"    ❌ Prediction failed")
            
            accuracy = correct_count / len(test_samples) * 100
            print(f"\n🎯 AR Model Test Results:")
            print(f"   Correct: {correct_count}/{len(test_samples)}")
            print(f"   Accuracy: {accuracy:.2f}%")
            
            return accuracy
            
        except Exception as e:
            print(f"❌ Error testing samples: {e}")
            return 0
    
    def test_specific_samples(self):
        """Test with specific samples including the selected one"""
        print("\n🧪 Testing Specific Samples:")
        print("=" * 50)
        
        # Test samples including the one from user's selection
        test_samples = [
            {'raw': [1104, 41210, 39113], 'expected': '3-POINT', 'name': 'POINT (from selected data)'},
            {'raw': [2320, 41290, 34440], 'expected': '0-OPEN', 'name': 'OPEN sample 1'},
            {'raw': [4273, 53196, 55885], 'expected': '0-OPEN', 'name': 'OPEN sample 2'},
            {'raw': [1216, 44874, 46635], 'expected': '0-OPEN', 'name': 'OPEN sample 3'},
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
            else:
                print(f"   ❌ Prediction failed")
        
        accuracy = correct_count / len(test_samples) * 100
        print(f"\n🎯 Specific Sample Results:")
        print(f"   Correct: {correct_count}/{len(test_samples)}")
        print(f"   Accuracy: {accuracy:.2f}%")
        
        return accuracy

def main():
    """Main function"""
    tester = ARModelTester()
    
    # Load AR model
    if not tester.load_ar_model():
        return
    
    # Test with sample data from dataset
    dataset_accuracy = tester.test_sample_data()
    
    # Test with specific samples
    specific_accuracy = tester.test_specific_samples()
    
    print(f"\n🎉 AR Model Testing completed!")
    print(f"📊 Dataset sample accuracy: {dataset_accuracy:.2f}%")
    print(f"📊 Specific sample accuracy: {specific_accuracy:.2f}%")
    
    if dataset_accuracy > 80:
        print(f"✅ AR model shows good performance!")
    else:
        print(f"⚠️  AR model performance needs investigation")

if __name__ == "__main__":
    main()
