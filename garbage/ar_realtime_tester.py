#!/usr/bin/env python3
"""
AR Random Forest Real-Time Tester
Uses Autoregressive features with Random Forest for real-time EMG prediction
Based on your 98.92% accuracy model
"""

import serial
import time
import numpy as np
import joblib
import os
from collections import deque

class ARRealtimeTester:
    def __init__(self, port='COM8', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        
        # Model components
        self.model = None
        self.label_encoder = None
        self.gesture_names = None
        self.lag_order = 10
        
        # AR data buffer
        self.emg_buffer = deque(maxlen=50)  # Store recent EMG samples
        
        print("🌲 AR Random Forest Real-Time Tester")
        print("📈 Autoregressive Features + Random Forest")
        print("🎯 Your 98.92% Accuracy Model")
        print("=" * 60)
    
    def load_model(self):
        """Load the improved AR model"""
        print("📂 Loading Improved AR model...")

        # Try improved model first
        model_file = 'improved_ar_model.pkl'

        if not os.path.exists(model_file):
            # Fall back to original AR model
            model_file = 'ar_random_forest_model.pkl'
            if not os.path.exists(model_file):
                print("❌ No AR model found!")
                print("🔧 Please run: python train_improved_ar_model.py")
                return False

        try:
            model_data = joblib.load(model_file)

            self.model = model_data['model']
            self.scaler = model_data.get('scaler')  # May not exist in old models
            self.label_encoder = model_data['label_encoder']
            self.gesture_names = model_data['gesture_names']
            self.lag_order = model_data['lag_order']

            print(f"✅ AR model loaded!")
            print(f"📊 Model type: {model_data['model_type']}")
            print(f"📊 Accuracy: {model_data['accuracy']:.4f}")
            print(f"📊 Algorithm: {model_data['algorithm']}")
            print(f"📈 Lag order: {self.lag_order}")
            print(f"⚖️ Scaling: {'YES' if self.scaler else 'NO'}")

            print(f"\n🎯 Available Gestures:")
            for i, gesture in enumerate(self.gesture_names):
                print(f"   {i}: {gesture}")

            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def normalize_emg(self, raw_emg):
        """Convert raw EMG to clean EMG"""
        ch1, ch2, ch3 = raw_emg
        
        # Normalize to 0-1 range
        emg1_clean = ch1 / 65535.0
        emg2_clean = ch2 / 65535.0
        emg3_clean = ch3 / 65535.0
        
        return [emg1_clean, emg2_clean, emg3_clean]
    
    def create_ar_features(self, emg_sequence, timestamp):
        """Create improved AR features from EMG sequence"""
        if len(emg_sequence) != self.lag_order:
            return None

        features = []
        ar_array = np.array(emg_sequence)

        # Enhanced AR features for each channel
        for ch_idx in range(3):
            channel_values = ar_array[:, ch_idx]

            # Basic lagged values (every 2nd value to reduce dimensionality)
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

        # Current values (last in sequence)
        current_values = emg_sequence[-1]
        features.extend(current_values)

        # Cross-channel features
        features.append(current_values[0] * current_values[1])
        features.append(current_values[0] * current_values[2])
        features.append(current_values[1] * current_values[2])
        features.append(np.sum(current_values))
        features.append(np.std(current_values))

        # Enhanced cross-channel correlations
        for j in range(3):
            for k in range(j+1, 3):
                ch1_vals = ar_array[:, j]
                ch2_vals = ar_array[:, k]

                if len(ch1_vals) > 1:
                    try:
                        corr = np.corrcoef(ch1_vals, ch2_vals)[0,1]
                        features.append(corr if not np.isnan(corr) else 0)
                    except:
                        features.append(0)
                else:
                    features.append(0)

        # Enhanced temporal features
        features.append(timestamp % 100)
        features.append((timestamp // 100) % 100)
        features.append((timestamp // 10000) % 100)
        features.append(timestamp % 1000)

        return np.array(features)
    
    def predict_gesture(self, emg_sequence, timestamp):
        """Predict gesture using improved AR model"""
        if self.model is None:
            return None

        try:
            # Create AR features
            ar_features = self.create_ar_features(emg_sequence, timestamp)
            if ar_features is None:
                return None

            ar_features = ar_features.reshape(1, -1)

            # Scale features if scaler exists
            if self.scaler is not None:
                ar_features = self.scaler.transform(ar_features)

            # Make prediction
            pred_label = self.model.predict(ar_features)[0]
            pred_proba = self.model.predict_proba(ar_features)[0]

            gesture_name = self.gesture_names[pred_label]
            confidence = pred_proba[pred_label]

            return {
                'gesture': gesture_name,
                'confidence': confidence,
                'probabilities': pred_proba,
                'ar_features_count': len(ar_features[0])
            }

        except Exception as e:
            print(f"❌ AR prediction error: {e}")
            return None
    
    def test_known_samples(self):
        """Test with known samples using AR features"""
        print("\n🧪 Testing Known Samples with AR Features:")
        print("=" * 60)
        
        # Create AR sequences from known samples
        test_samples = [
            {'base': [0.292654267, 0.543227094, 0.530998914], 'expected': '10-OK_SIGN', 'name': 'OK_SIGN'},
            {'base': [0.469815094, 0.613435699, 0.62751976], 'expected': '6-PEACE', 'name': 'PEACE'},
            {'base': [0.262597128, 0.540141474, 0.530530678], 'expected': '3-POINT', 'name': 'POINT'},
            {'base': [0.353751761, 0.480362306, 0.477253055], 'expected': '5-FIVE', 'name': 'FIVE'},
            {'base': [0.253636884, 0.412346916, 0.407494172], 'expected': '1-CLOSE', 'name': 'CLOSE'}
        ]
        
        correct_count = 0
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\n{i}. Testing {sample['name']} with AR features:")
            print(f"   Base EMG: {sample['base']}")
            print(f"   Expected: {sample['expected']}")
            
            # Create AR sequence with variations
            ar_sequence = []
            for j in range(self.lag_order):
                variation = np.random.normal(0, 0.01, 3)
                varied_emg = [
                    max(0, min(1, sample['base'][0] + variation[0])),
                    max(0, min(1, sample['base'][1] + variation[1])),
                    max(0, min(1, sample['base'][2] + variation[2]))
                ]
                ar_sequence.append(varied_emg)
            
            result = self.predict_gesture(ar_sequence, 1752250605)
            
            if result:
                predicted = result['gesture']
                confidence = result['confidence']
                correct = predicted == sample['expected']
                
                if correct:
                    correct_count += 1
                    status = "✅ CORRECT!"
                else:
                    status = "❌ WRONG!"
                
                print(f"   AR sequence length: {len(ar_sequence)}")
                print(f"   AR features: {result['ar_features_count']}")
                print(f"   🎯 Predicted: {predicted}")
                print(f"   📊 Confidence: {confidence:.3f}")
                print(f"   🎯 Result: {status}")
                
                # Show top 3
                probs = result['probabilities']
                top_3_idx = np.argsort(probs)[-3:][::-1]
                print(f"   📊 Top 3:")
                for j, idx in enumerate(top_3_idx):
                    print(f"      {j+1}. {self.gesture_names[idx]:12s} - {probs[idx]:.3f}")
            else:
                print(f"   ❌ AR prediction failed")
        
        accuracy = correct_count / len(test_samples) * 100
        print(f"\n🎯 AR Test Results:")
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
    
    def start_ar_realtime_prediction(self):
        """Start real-time AR prediction"""
        if not self.is_connected or self.model is None:
            print("❌ Not ready for AR real-time prediction")
            return
        
        print("\n🌲 Starting AR Random Forest Real-Time Prediction...")
        print("📈 Collecting AR sequence data...")
        print("💪 Perform hand gestures and hold for 2-3 seconds!")
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
                                
                                # Convert to clean EMG
                                clean_emg = self.normalize_emg([ch1, ch2, ch3])
                                
                                # Add to AR buffer
                                self.emg_buffer.append({
                                    'clean_emg': clean_emg,
                                    'timestamp': timestamp,
                                    'raw': [ch1, ch2, ch3]
                                })
                                
                                # Make AR prediction when we have enough data
                                if len(self.emg_buffer) >= self.lag_order:
                                    # Get last lag_order samples for AR sequence
                                    recent_samples = list(self.emg_buffer)[-self.lag_order:]
                                    ar_sequence = [sample['clean_emg'] for sample in recent_samples]
                                    
                                    # Predict gesture using AR features
                                    result = self.predict_gesture(ar_sequence, timestamp)
                                    
                                    if result and result['confidence'] > 0.4:  # AR threshold
                                        prediction_count += 1
                                        
                                        print(f"\n🌲 AR Prediction #{prediction_count}")
                                        print(f"   Raw EMG: [{ch1}, {ch2}, {ch3}]")
                                        print(f"   Clean EMG: {[f'{x:.3f}' for x in clean_emg]}")
                                        print(f"   AR sequence: {self.lag_order} samples")
                                        print(f"   AR features: {result['ar_features_count']}")
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
                
                time.sleep(0.05)  # Slightly slower for AR processing
                
        except KeyboardInterrupt:
            print(f"\n🛑 AR real-time prediction stopped")
            print(f"📊 Total AR predictions: {prediction_count}")
        finally:
            if self.serial_conn:
                self.serial_conn.close()

def main():
    """Main function"""
    tester = ARRealtimeTester()
    
    # Load AR model
    if not tester.load_model():
        return
    
    print(f"\n📋 AR Testing Options:")
    print(f"1. 🧪 Test known samples with AR features")
    print(f"2. 🔌 Real-time AR prediction with Pico")
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
                    print("\n💪 AR Instructions:")
                    print("1. AR model needs sequential data")
                    print("2. Hold each gesture for 3-4 seconds")
                    print("3. Wait for AR sequence to build up")
                    print("4. Try: OK sign, Peace, Point, Fist, Open hand")
                    
                    input("Press Enter to start AR prediction...")
                    tester.start_ar_realtime_prediction()
                
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
