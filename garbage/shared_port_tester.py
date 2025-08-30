#!/usr/bin/env python3
"""
Shared Port EMG Tester
Works even when Thonny is connected to the same port
"""

import serial
import time
import numpy as np
import joblib
import os
from collections import deque

class SharedPortEMGTester:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.gesture_names = None
        self.lag_order = 15
        self.emg_buffer = deque(maxlen=100)
        
        print("🎮 Shared Port EMG Tester")
        print("🔧 Works even with Thonny connected")
        print("=" * 50)
    
    def load_model(self):
        """Load the AR + LightGBM model"""
        print("📂 Loading AR + LightGBM model...")
        
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
            print(f"📊 Accuracy: {model_data.get('accuracy', 'Unknown'):.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def try_connect_port(self, port, max_attempts=3):
        """Try to connect to port with multiple attempts"""
        for attempt in range(max_attempts):
            try:
                print(f"🔌 Attempt {attempt + 1}: Connecting to {port}...")
                ser = serial.Serial(port, 115200, timeout=1)
                time.sleep(0.5)
                print(f"✅ Connected to {port}")
                return ser
            except serial.SerialException as e:
                if "Access is denied" in str(e):
                    print(f"⚠️  Port {port} busy, waiting...")
                    time.sleep(2)
                else:
                    print(f"❌ Port {port} error: {e}")
                    break
            except Exception as e:
                print(f"❌ Error: {e}")
                break
        
        return None
    
    def normalize_emg(self, raw_values):
        """Convert raw EMG values to clean normalized values"""
        clean_values = []
        
        for i in range(3):
            raw_val = raw_values[i]
            # Normalize to 0-1 range
            clean_val = raw_val / 65535.0
            clean_val = max(0, min(1, clean_val))
            clean_values.append(clean_val)
        
        return clean_values
    
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
            
            # Scale if needed
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
                'timestamp': timestamp,
                'probabilities': pred_proba
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def test_with_manual_input(self):
        """Test with manual EMG input (when port is busy)"""
        print("\n🎮 Manual EMG Input Mode")
        print("📊 Enter EMG data from Thonny output")
        print("Format: timestamp,ch1,ch2,ch3")
        print("Example: 1234567,1250,42000,38000")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                line = input("📡 Enter EMG data: ").strip()
                
                if line.lower() == 'quit':
                    break
                
                if ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 4:
                        timestamp = int(parts[0])
                        ch1 = int(parts[1])
                        ch2 = int(parts[2])
                        ch3 = int(parts[3])
                        
                        # Convert to clean EMG
                        clean_emg = self.normalize_emg([ch1, ch2, ch3])
                        
                        # Add to buffer
                        self.emg_buffer.append({
                            'clean_emg': clean_emg,
                            'timestamp': timestamp,
                            'raw': [ch1, ch2, ch3]
                        })
                        
                        print(f"✅ Added sample: Raw={[ch1, ch2, ch3]}, Clean={[f'{x:.3f}' for x in clean_emg]}")
                        
                        # Make prediction when we have enough data
                        if len(self.emg_buffer) >= self.lag_order:
                            recent_samples = list(self.emg_buffer)[-self.lag_order:]
                            emg_sequence = [sample['clean_emg'] for sample in recent_samples]
                            
                            prediction = self.predict_gesture(emg_sequence, timestamp)
                            
                            if prediction:
                                print(f"\n🎯 Prediction:")
                                print(f"   💪 Gesture: {prediction['gesture']}")
                                print(f"   📊 Confidence: {prediction['confidence']:.3f}")
                                
                                if prediction['confidence'] < 0.8:
                                    probs = prediction['probabilities']
                                    top_3_idx = np.argsort(probs)[-3:][::-1]
                                    print(f"   📊 Top 3:")
                                    for i, idx in enumerate(top_3_idx):
                                        print(f"      {i+1}. {self.gesture_names[idx]:12s} - {probs[idx]:.3f}")
                        else:
                            print(f"   Need {self.lag_order - len(self.emg_buffer)} more samples for prediction")
                    else:
                        print("❌ Invalid format. Use: timestamp,ch1,ch2,ch3")
                else:
                    print("❌ Invalid format. Use: timestamp,ch1,ch2,ch3")
                    
            except ValueError:
                print("❌ Invalid numbers. Use: timestamp,ch1,ch2,ch3")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def start_testing(self, port):
        """Start testing with the specified port"""
        # Try to connect
        ser = self.try_connect_port(port)
        
        if ser is None:
            print(f"\n⚠️  Cannot connect to {port}")
            print("🔧 Options:")
            print("1. Close Thonny and try again")
            print("2. Use manual input mode")
            
            choice = input("Choose (1/2): ").strip()
            if choice == '2':
                self.test_with_manual_input()
            return
        
        # Connected successfully - start real-time testing
        print(f"\n🎮 Real-time testing on {port}")
        print("💪 Perform gestures!")
        print("🛑 Press Ctrl+C to stop")
        
        prediction_count = 0
        
        try:
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line and ',' in line and not line.startswith('#'):
                        parts = line.split(',')
                        if len(parts) >= 4:
                            try:
                                timestamp = int(parts[0])
                                ch1 = int(parts[1])
                                ch2 = int(parts[2])
                                ch3 = int(parts[3])
                                
                                clean_emg = self.normalize_emg([ch1, ch2, ch3])
                                
                                self.emg_buffer.append({
                                    'clean_emg': clean_emg,
                                    'timestamp': timestamp,
                                    'raw': [ch1, ch2, ch3]
                                })
                                
                                if len(self.emg_buffer) >= self.lag_order:
                                    recent_samples = list(self.emg_buffer)[-self.lag_order:]
                                    emg_sequence = [sample['clean_emg'] for sample in recent_samples]
                                    
                                    prediction = self.predict_gesture(emg_sequence, timestamp)
                                    
                                    if prediction and prediction['confidence'] > 0.6:
                                        prediction_count += 1
                                        print(f"\n🎯 Prediction #{prediction_count}")
                                        print(f"   Raw: {[ch1, ch2, ch3]}")
                                        print(f"   💪 Gesture: {prediction['gesture']}")
                                        print(f"   📊 Confidence: {prediction['confidence']:.3f}")
                            
                            except ValueError:
                                pass
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Stopped. Made {prediction_count} predictions")
        finally:
            ser.close()

def main():
    """Main function"""
    tester = SharedPortEMGTester()
    
    if not tester.load_model():
        return
    
    print(f"\n📋 Testing Options:")
    print(f"1. 🔌 Try connecting to COM port")
    print(f"2. 📊 Manual input mode (copy from Thonny)")
    print(f"3. 🚪 Exit")
    
    while True:
        try:
            choice = input(f"\nChoose option (1-3): ").strip()
            
            if choice == '1':
                port = input("Enter COM port (e.g., COM8): ").strip()
                tester.start_testing(port)
                
            elif choice == '2':
                tester.test_with_manual_input()
                
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
