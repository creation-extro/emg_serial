#!/usr/bin/env python3
"""
Test AR + LightGBM Model with Sample EMG Signals
Input your EMG signals with timestamps to get gesture predictions
"""

import numpy as np
import joblib
import os
from collections import deque

class EMGSampleTester:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.gesture_names = None
        self.lag_order = 15
        self.sample_buffer = deque(maxlen=50)
        
        print("🧪 AR + LightGBM EMG Sample Tester")
        print("🎯 Test Your EMG Signals with Timestamps")
        print("=" * 60)
    
    def load_model(self):
        """Load the AR + LightGBM model"""
        print("📂 Loading AR + LightGBM model...")
        
        # Look for LightGBM model
        model_files = [f for f in os.listdir('.') if 'lightgbm' in f.lower() and f.endswith('.pkl')]
        
        if not model_files:
            print("❌ LightGBM model not found!")
            return False
        
        model_file = model_files[0]
        print(f"📂 Loading: {model_file}")
        
        try:
            model_data = joblib.load(model_file)
            
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.label_encoder = model_data['label_encoder']
            self.gesture_names = model_data['gesture_names']
            self.lag_order = model_data.get('lag_order', 15)
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 Model type: {model_data.get('model_type', 'Unknown')}")
            print(f"📊 Accuracy: {model_data.get('accuracy', 'Unknown'):.4f}")
            print(f"📈 Lag order: {self.lag_order}")
            
            print(f"\n🎯 Available Gestures:")
            for i, gesture in enumerate(self.gesture_names):
                print(f"   {i}: {gesture}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_enhanced_ar_features(self, emg_sequence, timestamp):
        """Create enhanced AR features for LightGBM"""
        if len(emg_sequence) != self.lag_order:
            return None
        
        features = []
        emg_array = np.array(emg_sequence)
        
        # Enhanced AR features for each channel
        for ch_idx in range(3):
            channel_values = emg_array[:, ch_idx]
            
            # Basic lagged values (every 2nd value)
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
        
        # Current values
        current_values = emg_sequence[-1]
        features.extend(current_values)
        
        # Cross-channel features
        features.append(current_values[0] * current_values[1])
        features.append(current_values[0] * current_values[2])
        features.append(current_values[1] * current_values[2])
        features.append(np.sum(current_values))
        features.append(np.std(current_values))
        
        # Enhanced timestamp features
        features.append(timestamp % 100)
        features.append((timestamp // 100) % 100)
        features.append((timestamp // 10000) % 100)
        features.append(timestamp % 1000)
        
        return np.array(features)
    
    def predict_gesture(self, emg_sequence, timestamp):
        """Predict gesture from EMG sequence"""
        if self.model is None:
            return None
        
        try:
            # Create AR features
            ar_features = self.create_enhanced_ar_features(emg_sequence, timestamp)
            if ar_features is None:
                return None
            
            ar_features = ar_features.reshape(1, -1)
            
            print(f"🔍 Created {ar_features.shape[1]} AR features")
            
            # Scale if needed
            if self.scaler is not None:
                ar_features = self.scaler.transform(ar_features)
            
            # Make prediction
            pred_label = self.model.predict(ar_features)[0]
            pred_proba = self.model.predict_proba(ar_features)[0]
            
            gesture_name = self.gesture_names[pred_label]
            confidence = pred_proba[pred_label]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(pred_proba)[-3:][::-1]
            top_3_predictions = []
            
            for idx in top_3_indices:
                top_3_predictions.append({
                    'gesture': self.gesture_names[idx],
                    'confidence': pred_proba[idx]
                })
            
            return {
                'gesture': gesture_name,
                'confidence': confidence,
                'top_3': top_3_predictions,
                'timestamp': timestamp,
                'features_count': ar_features.shape[1]
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def create_sequence_from_sample(self, base_emg, timestamp):
        """Create AR sequence from a single EMG sample"""
        sequence = []
        
        # Create a realistic sequence by adding small variations
        for i in range(self.lag_order):
            # Add small random variation to simulate time series
            variation = np.random.normal(0, 0.01, 3)
            varied_emg = [
                max(0, min(1, base_emg[0] + variation[0])),
                max(0, min(1, base_emg[1] + variation[1])),
                max(0, min(1, base_emg[2] + variation[2]))
            ]
            sequence.append(varied_emg)
        
        return sequence
    
    def test_single_sample(self, emg_clean, timestamp, expected_gesture=None):
        """Test a single EMG sample"""
        print(f"\n🧪 Testing EMG Sample:")
        print(f"   Clean EMG: {emg_clean}")
        print(f"   Timestamp: {timestamp}")
        if expected_gesture:
            print(f"   Expected: {expected_gesture}")
        
        # Create AR sequence
        emg_sequence = self.create_sequence_from_sample(emg_clean, timestamp)
        
        # Make prediction
        result = self.predict_gesture(emg_sequence, timestamp)
        
        if result:
            print(f"\n🎯 AR + LightGBM Prediction:")
            print(f"   Predicted: {result['gesture']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   AR Features: {result['features_count']}")
            
            if expected_gesture:
                correct = "✅ CORRECT!" if result['gesture'] == expected_gesture else "❌ WRONG!"
                print(f"   Result: {correct}")
            
            print(f"\n📊 Top 3 Predictions:")
            for i, pred in enumerate(result['top_3'], 1):
                print(f"   {i}. {pred['gesture']:15s} - {pred['confidence']:.3f}")
            
            # Store in history
            self.sample_buffer.append({
                'input': emg_clean,
                'timestamp': timestamp,
                'predicted': result['gesture'],
                'confidence': result['confidence'],
                'expected': expected_gesture
            })
            
            return result
        else:
            print("❌ Prediction failed")
            return None
    
    def show_known_examples(self):
        """Show known EMG examples for testing"""
        print("\n📊 Known EMG Examples for Testing:")
        print("=" * 60)
        
        examples = [
            {
                'name': 'OK_SIGN',
                'emg_clean': [0.292654267, 0.543227094, 0.530998914],
                'timestamp': 1752250605,
                'expected': '10-OK_SIGN'
            },
            {
                'name': 'PEACE',
                'emg_clean': [0.469815094, 0.613435699, 0.62751976],
                'timestamp': 1752250579,
                'expected': '6-PEACE'
            },
            {
                'name': 'POINT',
                'emg_clean': [0.262597128, 0.540141474, 0.530530678],
                'timestamp': 1752250562,
                'expected': '3-POINT'
            },
            {
                'name': 'FIVE',
                'emg_clean': [0.353751761, 0.480362306, 0.477253055],
                'timestamp': 1752250572,
                'expected': '5-FIVE'
            },
            {
                'name': 'CLOSE',
                'emg_clean': [0.253636884, 0.412346916, 0.407494172],
                'timestamp': 1752251517,
                'expected': '1-CLOSE'
            }
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example['name']} ({example['expected']}):")
            print(f"   EMG: {example['emg_clean']}")
            print(f"   Timestamp: {example['timestamp']}")
            print(f"   Input format: {example['emg_clean'][0]} {example['emg_clean'][1]} {example['emg_clean'][2]} {example['timestamp']}")
            print()
    
    def test_all_examples(self):
        """Test all known examples"""
        print("\n🧪 Testing All Known Examples:")
        print("=" * 60)
        
        examples = [
            ([0.292654267, 0.543227094, 0.530998914], 1752250605, '10-OK_SIGN'),
            ([0.469815094, 0.613435699, 0.62751976], 1752250579, '6-PEACE'),
            ([0.262597128, 0.540141474, 0.530530678], 1752250562, '3-POINT'),
            ([0.353751761, 0.480362306, 0.477253055], 1752250572, '5-FIVE'),
            ([0.253636884, 0.412346916, 0.407494172], 1752251517, '1-CLOSE')
        ]
        
        correct_count = 0
        total_count = len(examples)
        
        for i, (emg_clean, timestamp, expected) in enumerate(examples, 1):
            print(f"\n--- Test {i}/{total_count} ---")
            result = self.test_single_sample(emg_clean, timestamp, expected)
            
            if result and result['gesture'] == expected:
                correct_count += 1
        
        accuracy = correct_count / total_count * 100
        print(f"\n🎯 Test Summary:")
        print(f"   Total Tests: {total_count}")
        print(f"   Correct: {correct_count}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        return accuracy
    
    def interactive_mode(self):
        """Interactive testing mode"""
        print("\n🎮 Interactive EMG Sample Testing")
        print("=" * 60)
        print("Enter your EMG signals with timestamps")
        print("Format: emg1_clean emg2_clean emg3_clean timestamp")
        print("Example: 0.292654267 0.543227094 0.530998914 1752250605")
        print("Commands: 'examples' to see examples, 'test-all' to test all examples, 'quit' to exit")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n📡 Enter EMG sample: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'examples':
                    self.show_known_examples()
                    continue
                
                if user_input.lower() == 'test-all':
                    self.test_all_examples()
                    continue
                
                # Parse EMG input
                values = user_input.split()
                if len(values) < 4:
                    print("❌ Please enter 4 values: emg1_clean emg2_clean emg3_clean timestamp")
                    continue
                
                emg1, emg2, emg3, timestamp = values[:4]
                emg_clean = [float(emg1), float(emg2), float(emg3)]
                timestamp = int(timestamp)
                
                # Optional expected gesture
                expected = values[4] if len(values) > 4 else None
                
                # Test the sample
                self.test_single_sample(emg_clean, timestamp, expected)
                
            except ValueError:
                print("❌ Invalid input. Please enter 3 float values + 1 integer timestamp")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    tester = EMGSampleTester()
    
    # Load model
    if not tester.load_model():
        return
    
    print(f"\n📋 Testing Options:")
    print(f"1. 🧪 Test all known examples")
    print(f"2. 📊 Show known examples")
    print(f"3. 🎮 Interactive mode - Enter your samples")
    print(f"4. 🚪 Exit")
    
    while True:
        try:
            choice = input(f"\nChoose option (1-4): ").strip()
            
            if choice == '1':
                tester.test_all_examples()
                
            elif choice == '2':
                tester.show_known_examples()
                
            elif choice == '3':
                tester.interactive_mode()
                
            elif choice == '4':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
