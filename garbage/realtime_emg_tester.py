#!/usr/bin/env python3
"""
Real-Time EMG Gesture Prediction with Pico
Connects to Pico, collects muscle signals, and predicts gestures using AR + LightGBM
"""

import serial
import time
import numpy as np
import joblib
import os
from collections import deque
import threading
from datetime import datetime

class RealTimeEMGTester:
    def __init__(self, port='COM3', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self.is_collecting = False
        
        # Model components
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.gesture_names = None
        self.lag_order = 15
        
        # Data buffers
        self.emg_buffer = deque(maxlen=100)  # Store recent EMG samples
        self.prediction_history = deque(maxlen=20)  # Store recent predictions
        
        # EMG processing parameters
        self.emg_ranges = {
            'ch1': {'min': 0, 'max': 65535},
            'ch2': {'min': 0, 'max': 65535},
            'ch3': {'min': 0, 'max': 65535}
        }
        
        print("🎮 Real-Time EMG Gesture Tester")
        print("💪 Muscle Signal → AR + LightGBM → Gesture")
        print("🔌 Pico Connection Ready")
        print("=" * 60)
    
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
    
    def connect_pico(self):
        """Connect to Pico via serial"""
        print(f"🔌 Connecting to Pico on {self.port}...")
        
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for connection to stabilize
            
            if self.serial_conn.is_open:
                print("✅ Pico connected successfully!")
                print("💪 Pico should be calibrating baseline...")
                self.is_connected = True
                return True
            else:
                print("❌ Failed to open serial connection")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            print("🔧 Make sure:")
            print("   1. Pico is connected via USB")
            print("   2. Pico code is uploaded and running")
            print("   3. Correct COM port is specified")
            return False
    
    def disconnect_pico(self):
        """Disconnect from Pico"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.is_connected = False
            print("🔌 Pico disconnected")
    
    def normalize_emg(self, raw_values):
        """Convert raw EMG values to clean normalized values"""
        clean_values = []
        
        channels = ['ch1', 'ch2', 'ch3']
        for i, channel in enumerate(channels):
            raw_val = raw_values[i]
            min_val = self.emg_ranges[channel]['min']
            max_val = self.emg_ranges[channel]['max']
            
            # Normalize to 0-1 range
            clean_val = (raw_val - min_val) / (max_val - min_val)
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
    
    def read_emg_data(self):
        """Read EMG data from Pico"""
        if not self.is_connected:
            return None
        
        try:
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8').strip()
                
                # Skip status messages (lines starting with #)
                if line.startswith('#'):
                    print(f"📊 {line}")
                    return None
                
                # Parse EMG data (expecting format: "timestamp,ch1,ch2,ch3")
                if ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 4:
                        timestamp = int(parts[0])
                        ch1 = int(parts[1])
                        ch2 = int(parts[2])
                        ch3 = int(parts[3])
                        
                        return {
                            'timestamp': timestamp,
                            'raw': [ch1, ch2, ch3]
                        }
        except Exception as e:
            print(f"❌ Data reading error: {e}")
        
        return None
    
    def start_realtime_prediction(self):
        """Start real-time prediction loop"""
        if not self.is_connected or self.model is None:
            print("❌ Not ready for real-time prediction")
            return
        
        print("\n🎮 Starting Real-Time EMG Prediction...")
        print("💪 Perform hand gestures with your muscles!")
        print("📊 Predictions will appear when enough data is collected")
        print("🛑 Press Ctrl+C to stop")
        print("-" * 60)
        
        self.is_collecting = True
        prediction_count = 0
        
        try:
            while self.is_collecting:
                # Read EMG data
                emg_data = self.read_emg_data()
                
                if emg_data:
                    # Convert to clean EMG
                    clean_emg = self.normalize_emg(emg_data['raw'])
                    timestamp = emg_data['timestamp']
                    
                    # Add to buffer
                    self.emg_buffer.append({
                        'clean_emg': clean_emg,
                        'timestamp': timestamp,
                        'raw': emg_data['raw']
                    })
                    
                    # Make prediction when we have enough data
                    if len(self.emg_buffer) >= self.lag_order:
                        # Get last lag_order samples
                        recent_samples = list(self.emg_buffer)[-self.lag_order:]
                        emg_sequence = [sample['clean_emg'] for sample in recent_samples]
                        
                        # Predict gesture
                        prediction = self.predict_gesture(emg_sequence, timestamp)
                        
                        if prediction and prediction['confidence'] > 0.5:  # Only show confident predictions
                            prediction_count += 1
                            
                            # Store prediction
                            self.prediction_history.append(prediction)
                            
                            # Display prediction
                            current_time = datetime.now().strftime("%H:%M:%S")
                            print(f"\n🎯 Prediction #{prediction_count} at {current_time}")
                            print(f"   Raw EMG: {emg_data['raw']}")
                            print(f"   Clean EMG: {[f'{x:.3f}' for x in clean_emg]}")
                            print(f"   💪 Gesture: {prediction['gesture']}")
                            print(f"   📊 Confidence: {prediction['confidence']:.3f}")
                            
                            # Show top 3 if confidence is moderate
                            if prediction['confidence'] < 0.8:
                                probs = prediction['probabilities']
                                top_3_idx = np.argsort(probs)[-3:][::-1]
                                print(f"   📊 Top 3:")
                                for i, idx in enumerate(top_3_idx):
                                    print(f"      {i+1}. {self.gesture_names[idx]:12s} - {probs[idx]:.3f}")
                
                time.sleep(0.01)  # Small delay
                
        except KeyboardInterrupt:
            print(f"\n🛑 Real-time prediction stopped")
            print(f"📊 Total predictions made: {prediction_count}")
        except Exception as e:
            print(f"\n❌ Real-time prediction error: {e}")
        finally:
            self.is_collecting = False

def main():
    """Main function"""
    tester = RealTimeEMGTester()
    
    # Load model
    if not tester.load_model():
        return
    
    print(f"\n📋 Real-Time EMG Testing:")
    print(f"1. 🔌 Connect to Pico and start real-time prediction")
    print(f"2. 🚪 Exit")
    
    while True:
        try:
            choice = input(f"\nChoose option (1-2): ").strip()
            
            if choice == '1':
                port = input("Enter COM port (default COM3): ").strip() or 'COM3'
                tester.port = port
                
                if tester.connect_pico():
                    print("\n💪 Instructions:")
                    print("1. Keep muscles relaxed during calibration")
                    print("2. After calibration, perform different hand gestures")
                    print("3. Hold each gesture for 2-3 seconds")
                    print("4. Try: OK sign, Peace, Point, Fist, Open hand")
                    
                    input("\nPress Enter when ready to start...")
                    tester.start_realtime_prediction()
                    tester.disconnect_pico()
                
            elif choice == '2':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1 or 2")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
