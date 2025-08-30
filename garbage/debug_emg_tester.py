#!/usr/bin/env python3
"""
Debug EMG Tester - Shows exactly what's happening
Helps identify why predictions aren't being made
"""

import serial
import time
import numpy as np
import joblib
import os
from collections import deque
from datetime import datetime

class DebugEMGTester:
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
        self.lag_order = 15
        
        # Data buffers
        self.emg_buffer = deque(maxlen=100)
        self.raw_data_count = 0
        self.valid_data_count = 0
        
        print("🔍 Debug EMG Tester")
        print("🕵️ Shows exactly what's happening")
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
            print(f"🎯 Gestures: {len(self.gesture_names)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def connect_pico(self):
        """Connect to Pico via serial"""
        print(f"🔌 Connecting to Pico on {self.port}...")
        
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            
            if self.serial_conn.is_open:
                print("✅ Pico connected successfully!")
                self.is_connected = True
                return True
            else:
                print("❌ Failed to open serial connection")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
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
                'probabilities': pred_proba,
                'features_count': len(ar_features[0])
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def start_debug_session(self):
        """Start debug session with detailed output"""
        if not self.is_connected or self.model is None:
            print("❌ Not ready for debugging")
            return
        
        print("\n🔍 Starting Debug Session...")
        print("🕵️ Showing all data received and processing steps")
        print("🛑 Press Ctrl+C to stop")
        print("-" * 60)
        
        last_status_time = time.time()
        
        try:
            while True:
                # Show status every 5 seconds
                current_time = time.time()
                if current_time - last_status_time > 5:
                    print(f"\n📊 Status Update:")
                    print(f"   Raw data lines received: {self.raw_data_count}")
                    print(f"   Valid EMG samples: {self.valid_data_count}")
                    print(f"   Buffer size: {len(self.emg_buffer)}/{self.lag_order}")
                    print(f"   Ready for prediction: {'YES' if len(self.emg_buffer) >= self.lag_order else 'NO'}")
                    last_status_time = current_time
                
                # Read data from Pico
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    self.raw_data_count += 1
                    
                    print(f"\n📡 Raw line #{self.raw_data_count}: {line}")
                    
                    # Skip status messages
                    if line.startswith('#') or line.startswith('🔧') or line.startswith('💪'):
                        print(f"   ℹ️  Status message - skipping")
                        continue
                    
                    # Parse EMG data
                    if ',' in line:
                        parts = line.split(',')
                        print(f"   🔍 Split into {len(parts)} parts: {parts}")
                        
                        if len(parts) >= 4:
                            try:
                                timestamp = int(parts[0])
                                ch1 = int(parts[1])
                                ch2 = int(parts[2])
                                ch3 = int(parts[3])
                                
                                print(f"   ✅ Parsed: timestamp={timestamp}, ch1={ch1}, ch2={ch2}, ch3={ch3}")
                                
                                # Convert to clean EMG
                                clean_emg = self.normalize_emg([ch1, ch2, ch3])
                                print(f"   🧹 Clean EMG: {[f'{x:.3f}' for x in clean_emg]}")
                                
                                # Add to buffer
                                self.emg_buffer.append({
                                    'clean_emg': clean_emg,
                                    'timestamp': timestamp,
                                    'raw': [ch1, ch2, ch3]
                                })
                                
                                self.valid_data_count += 1
                                print(f"   📊 Added to buffer. Buffer size: {len(self.emg_buffer)}")
                                
                                # Try prediction when we have enough data
                                if len(self.emg_buffer) >= self.lag_order:
                                    print(f"   🎯 Attempting prediction...")
                                    
                                    # Get last lag_order samples
                                    recent_samples = list(self.emg_buffer)[-self.lag_order:]
                                    emg_sequence = [sample['clean_emg'] for sample in recent_samples]
                                    
                                    # Predict gesture
                                    prediction = self.predict_gesture(emg_sequence, timestamp)
                                    
                                    if prediction:
                                        print(f"   🎯 PREDICTION MADE:")
                                        print(f"      Gesture: {prediction['gesture']}")
                                        print(f"      Confidence: {prediction['confidence']:.3f}")
                                        print(f"      Features: {prediction['features_count']}")
                                        
                                        # Show top 3 always in debug mode
                                        probs = prediction['probabilities']
                                        top_3_idx = np.argsort(probs)[-3:][::-1]
                                        print(f"      Top 3:")
                                        for i, idx in enumerate(top_3_idx):
                                            print(f"         {i+1}. {self.gesture_names[idx]:12s} - {probs[idx]:.3f}")
                                    else:
                                        print(f"   ❌ Prediction failed")
                                else:
                                    needed = self.lag_order - len(self.emg_buffer)
                                    print(f"   ⏳ Need {needed} more samples for prediction")
                                
                            except ValueError as e:
                                print(f"   ❌ Parse error: {e}")
                        else:
                            print(f"   ⚠️  Not enough parts (need 4, got {len(parts)})")
                    else:
                        print(f"   ⚠️  No comma found in line")
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Debug session stopped")
            print(f"📊 Final Stats:")
            print(f"   Raw data lines: {self.raw_data_count}")
            print(f"   Valid EMG samples: {self.valid_data_count}")
            print(f"   Buffer size: {len(self.emg_buffer)}")
        except Exception as e:
            print(f"\n❌ Debug session error: {e}")
        finally:
            if self.serial_conn:
                self.serial_conn.close()
                self.is_connected = False

def main():
    """Main function"""
    tester = DebugEMGTester()
    
    # Load model
    if not tester.load_model():
        return
    
    # Connect to Pico
    port = input("Enter COM port (default COM8): ").strip() or 'COM8'
    tester.port = port
    
    if not tester.connect_pico():
        return
    
    print(f"\n🔍 Debug Mode Instructions:")
    print(f"1. Make sure your Pico code is running in Thonny")
    print(f"2. You should see EMG data streaming")
    print(f"3. Perform hand gestures")
    print(f"4. Watch the debug output to see what's happening")
    
    input("Press Enter to start debug session...")
    
    tester.start_debug_session()

if __name__ == "__main__":
    main()
