#!/usr/bin/env python3
"""
Working Real-Time EMG Tester
Uses the working Random Forest model with enhanced features
"""

import serial
import time
import numpy as np
import joblib
import os

class WorkingRealtimeTester:
    def __init__(self, port='COM8', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        
        # Model components
        self.model = None
        self.label_encoder = None
        self.gesture_names = None
        
        print("🌲 Working Real-Time EMG Tester")
        print("📊 Using Random Forest Model")
        print("⚡ Enhanced Features")
        print("=" * 60)
    
    def load_model(self):
        """Load the working model"""
        print("📂 Loading Working Model...")
        
        model_file = 'working_emg_model.pkl'
        
        if not os.path.exists(model_file):
            print("❌ Working model not found!")
            print("🔧 Please run: python train_working_model.py")
            return False
        
        try:
            model_data = joblib.load(model_file)
            
            self.model = model_data['model']
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
    
    def normalize_emg(self, raw_emg):
        """Convert raw EMG to clean EMG (same as dataset)"""
        # Simple normalization (you can adjust based on your data range)
        ch1, ch2, ch3 = raw_emg
        
        # Normalize to 0-1 range (adjust max values based on your sensors)
        emg1_clean = ch1 / 65535.0
        emg2_clean = ch2 / 65535.0
        emg3_clean = ch3 / 65535.0
        
        return [emg1_clean, emg2_clean, emg3_clean]
    
    def create_enhanced_features(self, raw_emg, clean_emg, timestamp=1752250605):
        """Create enhanced features (same as training)"""
        ch1, ch2, ch3 = raw_emg
        emg1_clean, emg2_clean, emg3_clean = clean_emg
        
        # Same feature creation as training
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
            
            # Timestamp features
            timestamp % 1000,
            (timestamp // 1000) % 1000,
        ]
        
        return np.array(feature_vector)
    
    def predict_gesture(self, raw_emg, timestamp):
        """Predict gesture from raw EMG"""
        if self.model is None:
            return None
        
        try:
            # Convert to clean EMG
            clean_emg = self.normalize_emg(raw_emg)
            
            # Create enhanced features
            features = self.create_enhanced_features(raw_emg, clean_emg, timestamp)
            features = features.reshape(1, -1)
            
            # Make prediction (Random Forest doesn't need scaling)
            pred_label = self.model.predict(features)[0]
            pred_proba = self.model.predict_proba(features)[0]
            
            gesture_name = self.gesture_names[pred_label]
            confidence = pred_proba[pred_label]
            
            return {
                'gesture': gesture_name,
                'confidence': confidence,
                'probabilities': pred_proba,
                'raw_emg': raw_emg,
                'clean_emg': clean_emg
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
            
            result = self.predict_gesture(sample['raw'], 1752250605)
            
            if result:
                predicted = result['gesture']
                confidence = result['confidence']
                correct = predicted == sample['expected']
                
                if correct:
                    correct_count += 1
                    status = "✅ CORRECT!"
                else:
                    status = "❌ WRONG!"
                
                print(f"   Clean EMG: {[f'{x:.3f}' for x in result['clean_emg']]}")
                print(f"   🎯 Predicted: {predicted}")
                print(f"   📊 Confidence: {confidence:.3f}")
                print(f"   🎯 Result: {status}")
                
                # Show top 3 for all predictions
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
    
    def start_realtime_prediction(self):
        """Start real-time prediction"""
        if not self.is_connected or self.model is None:
            print("❌ Not ready for real-time prediction")
            return
        
        print("\n🌲 Starting Real-Time Random Forest Prediction...")
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
                                result = self.predict_gesture([ch1, ch2, ch3], timestamp)
                                
                                if result and result['confidence'] > 0.3:  # Lower threshold
                                    prediction_count += 1
                                    
                                    print(f"\n🌲 Prediction #{prediction_count}")
                                    print(f"   Raw EMG: {result['raw_emg']}")
                                    print(f"   Clean EMG: {[f'{x:.3f}' for x in result['clean_emg']]}")
                                    print(f"   💪 Gesture: {result['gesture']}")
                                    print(f"   📊 Confidence: {result['confidence']:.3f}")
                                    
                                    # Show top 3 if confidence is moderate
                                    if result['confidence'] < 0.8:
                                        probs = result['probabilities']
                                        top_3_idx = np.argsort(probs)[-3:][::-1]
                                        print(f"   📊 Top 3:")
                                        for i, idx in enumerate(top_3_idx):
                                            print(f"      {i+1}. {self.gesture_names[idx]:12s} - {probs[idx]:.3f}")
                            
                            except ValueError:
                                pass
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Real-time prediction stopped")
            print(f"📊 Total predictions: {prediction_count}")
        finally:
            if self.serial_conn:
                self.serial_conn.close()

def main():
    """Main function"""
    tester = WorkingRealtimeTester()
    
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
