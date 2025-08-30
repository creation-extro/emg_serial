#!/usr/bin/env python3
"""
Simple Real-Time EMG Tester
Uses the reliable KNN model with raw EMG channels
Much simpler and more reliable than LightGBM
"""

import serial
import time
import numpy as np
import joblib
import os

class SimpleRealtimeTester:
    def __init__(self, port='COM8', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        
        # Model components
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.gesture_names = None
        
        print("🎯 Simple Real-Time EMG Tester")
        print("📊 Using Reliable KNN Model")
        print("⚡ Raw EMG Channels Only")
        print("=" * 60)
    
    def load_model(self):
        """Load the simple reliable model"""
        print("📂 Loading Simple Reliable Model...")
        
        model_file = 'simple_reliable_emg_model.pkl'
        
        if not os.path.exists(model_file):
            print("❌ Simple model not found!")
            print("🔧 Please run: python train_simple_reliable_model.py")
            return False
        
        try:
            model_data = joblib.load(model_file)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.gesture_names = model_data['gesture_names']
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 Model type: {model_data['model_type']}")
            print(f"📊 Accuracy: {model_data['accuracy']:.4f}")
            print(f"📊 Algorithm: {model_data['algorithm']}")
            print(f"📊 Features: {model_data['feature_count']}")
            
            print(f"\n🎯 Available Gestures:")
            for i, gesture in enumerate(self.gesture_names):
                print(f"   {i}: {gesture}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def connect_pico(self):
        """Connect to Pico"""
        print(f"🔌 Connecting to Pico on {self.port}...")
        
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            
            if self.serial_conn.is_open:
                print("✅ Pico connected successfully!")
                self.is_connected = True
                return True
            else:
                print("❌ Failed to connect")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def create_simple_features(self, raw_emg):
        """Create simple features from raw EMG (same as training)"""
        ch1, ch2, ch3 = raw_emg
        
        # Same feature creation as training
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
        
        return np.array(feature_vector)
    
    def predict_gesture(self, raw_emg):
        """Predict gesture from raw EMG"""
        if self.model is None:
            return None
        
        try:
            # Create features
            features = self.create_simple_features(raw_emg)
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            pred_label = self.model.predict(features_scaled)[0]
            pred_proba = self.model.predict_proba(features_scaled)[0]
            
            gesture_name = self.gesture_names[pred_label]
            confidence = pred_proba[pred_label]
            
            return {
                'gesture': gesture_name,
                'confidence': confidence,
                'probabilities': pred_proba
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def test_known_samples(self):
        """Test with known good samples"""
        print("\n🧪 Testing Known Samples:")
        print("=" * 50)
        
        # Your selected samples
        test_samples = [
            {'raw': [1216, 44154, 43322], 'expected': '10-OK_SIGN', 'name': 'OK_SIGN'},
            {'raw': [8930, 52764, 53100], 'expected': '6-PEACE', 'name': 'PEACE'},
            {'raw': [1536, 61951, 61487], 'expected': '3-POINT', 'name': 'POINT'},
            {'raw': [3456, 48234, 47892], 'expected': '5-FIVE', 'name': 'FIVE'},
            {'raw': [1424, 20869, 21061], 'expected': '1-CLOSE', 'name': 'CLOSE'}
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
                
                print(f"   🎯 Predicted: {predicted}")
                print(f"   📊 Confidence: {confidence:.3f}")
                print(f"   🎯 Result: {status}")
                
                # Show top 3 for wrong predictions
                if not correct:
                    probs = result['probabilities']
                    top_3_idx = np.argsort(probs)[-3:][::-1]
                    print(f"   📊 Top 3:")
                    for j, idx in enumerate(top_3_idx):
                        print(f"      {j+1}. {self.gesture_names[idx]:12s} - {probs[idx]:.3f}")
            else:
                print(f"   ❌ Prediction failed")
        
        accuracy = correct_count / len(test_samples) * 100
        print(f"\n🎯 Test Results:")
        print(f"   Correct: {correct_count}/{len(test_samples)}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        return accuracy
    
    def start_realtime_prediction(self):
        """Start real-time prediction"""
        if not self.is_connected or self.model is None:
            print("❌ Not ready for real-time prediction")
            return
        
        print("\n🎮 Starting Real-Time Prediction...")
        print("💪 Perform hand gestures!")
        print("🛑 Press Ctrl+C to stop")
        print("-" * 60)
        
        prediction_count = 0
        
        try:
            while True:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Skip status messages
                    if line.startswith('#') or line.startswith('🔧') or line.startswith('💪'):
                        continue
                    
                    # Parse EMG data
                    if ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 4:
                            try:
                                timestamp = int(parts[0])
                                ch1 = int(parts[1])
                                ch2 = int(parts[2])
                                ch3 = int(parts[3])
                                
                                # Predict gesture
                                result = self.predict_gesture([ch1, ch2, ch3])
                                
                                if result and result['confidence'] > 0.4:  # Lower threshold
                                    prediction_count += 1
                                    
                                    print(f"\n🎯 Prediction #{prediction_count}")
                                    print(f"   Raw EMG: [{ch1}, {ch2}, {ch3}]")
                                    print(f"   💪 Gesture: {result['gesture']}")
                                    print(f"   📊 Confidence: {result['confidence']:.3f}")
                                    
                                    # Show top 3 if confidence is moderate
                                    if result['confidence'] < 0.7:
                                        probs = result['probabilities']
                                        top_3_idx = np.argsort(probs)[-3:][::-1]
                                        print(f"   📊 Top 3:")
                                        for i, idx in enumerate(top_3_idx):
                                            print(f"      {i+1}. {self.gesture_names[idx]:12s} - {probs[idx]:.3f}")
                            
                            except ValueError:
                                pass
                
                time.sleep(0.1)  # Small delay
                
        except KeyboardInterrupt:
            print(f"\n🛑 Real-time prediction stopped")
            print(f"📊 Total predictions: {prediction_count}")
        finally:
            if self.serial_conn:
                self.serial_conn.close()

def main():
    """Main function"""
    tester = SimpleRealtimeTester()
    
    # Load model
    if not tester.load_model():
        return
    
    print(f"\n📋 Testing Options:")
    print(f"1. 🧪 Test known samples")
    print(f"2. 🔌 Real-time prediction with Pico")
    print(f"3. 🚪 Exit")
    
    while True:
        try:
            choice = input(f"\nChoose option (1-3): ").strip()
            
            if choice == '1':
                tester.test_known_samples()
                
            elif choice == '2':
                port = input("Enter COM port (default COM8): ").strip() or 'COM8'
                tester.port = port
                
                if tester.connect_pico():
                    print("\n💪 Instructions:")
                    print("1. Perform different hand gestures")
                    print("2. Hold each gesture for 2-3 seconds")
                    print("3. Try: OK sign, Peace, Point, Fist, Open hand")
                    
                    input("Press Enter to start...")
                    tester.start_realtime_prediction()
                
            elif choice == '3':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
