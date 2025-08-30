#!/usr/bin/env python3
"""
Live EMG Gesture Prediction System
Connects to Pico, processes real-time EMG data, and predicts gestures
Using AR + LightGBM model
"""

import serial
import time
import numpy as np
import pandas as pd
from collections import deque
import threading
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LiveEMGPredictor:
    def __init__(self, port='COM3', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self.is_predicting = False
        
        # Data buffers
        self.raw_buffer = deque(maxlen=1000)  # Store raw EMG data
        self.clean_buffer = deque(maxlen=100)  # Store clean EMG data for AR
        self.prediction_history = deque(maxlen=50)  # Store recent predictions
        
        # Model components
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.gesture_names = None
        self.lag_order = 15
        
        # EMG processing parameters
        self.emg_ranges = {
            'ch1': {'min': 0, 'max': 65535},
            'ch2': {'min': 0, 'max': 65535},
            'ch3': {'min': 0, 'max': 65535}
        }
        
        print("🎮 Live EMG Gesture Prediction System")
        print("⚡ AR + LightGBM Real-Time Processing")
        print("🔌 Pico Connection Ready")
        print("=" * 60)
    
    def load_model(self):
        """Load the trained AR + LightGBM model"""
        print("📂 Loading AR + LightGBM model...")
        
        # Look for LightGBM model first
        model_files = [f for f in os.listdir('.') if 'lightgbm' in f.lower() and f.endswith('.pkl')]
        
        if not model_files:
            # Look for any optimized AR model
            model_files = [f for f in os.listdir('.') if f.startswith('optimized_ar_emg') and f.endswith('.pkl')]
        
        if not model_files:
            print("❌ No suitable model found!")
            print("🔧 Please train AR + LightGBM model first")
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
            
            # Test connection
            if self.serial_conn.is_open:
                print("✅ Pico connected successfully!")
                self.is_connected = True
                return True
            else:
                print("❌ Failed to open serial connection")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            print("🔧 Make sure Pico is connected and port is correct")
            return False
    
    def disconnect_pico(self):
        """Disconnect from Pico"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.is_connected = False
            print("🔌 Pico disconnected")
    
    def normalize_emg(self, raw_values):
        """Convert raw EMG values to clean normalized values (0-1)"""
        clean_values = []
        
        channels = ['ch1', 'ch2', 'ch3']
        for i, channel in enumerate(channels):
            raw_val = raw_values[i]
            min_val = self.emg_ranges[channel]['min']
            max_val = self.emg_ranges[channel]['max']
            
            # Normalize to 0-1 range
            clean_val = (raw_val - min_val) / (max_val - min_val)
            clean_val = max(0, min(1, clean_val))  # Clamp to [0, 1]
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
                
                # Parse EMG data (expecting format: "ch1,ch2,ch3")
                if ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        ch1 = int(parts[0])
                        ch2 = int(parts[1])
                        ch3 = int(parts[2])
                        
                        timestamp = int(time.time() * 1000)  # Current timestamp in ms
                        
                        return {
                            'raw': [ch1, ch2, ch3],
                            'timestamp': timestamp
                        }
        except Exception as e:
            print(f"❌ Data reading error: {e}")
        
        return None
    
    def start_live_prediction(self):
        """Start live prediction loop"""
        if not self.is_connected or self.model is None:
            print("❌ Not ready for live prediction")
            return
        
        print("\n🎮 Starting Live EMG Prediction...")
        print("📊 Real-time gesture recognition active")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        
        self.is_predicting = True
        prediction_count = 0
        
        try:
            while self.is_predicting:
                # Read EMG data
                emg_data = self.read_emg_data()
                
                if emg_data:
                    # Convert to clean EMG
                    clean_emg = self.normalize_emg(emg_data['raw'])
                    timestamp = emg_data['timestamp']
                    
                    # Add to clean buffer
                    self.clean_buffer.append(clean_emg)
                    
                    # Store raw data for display
                    self.raw_buffer.append({
                        'raw': emg_data['raw'],
                        'clean': clean_emg,
                        'timestamp': timestamp
                    })
                    
                    # Make prediction when we have enough data
                    if len(self.clean_buffer) >= self.lag_order:
                        # Get last lag_order samples
                        emg_sequence = list(self.clean_buffer)[-self.lag_order:]
                        
                        # Predict gesture
                        prediction = self.predict_gesture(emg_sequence, timestamp)
                        
                        if prediction:
                            prediction_count += 1
                            
                            # Store prediction
                            self.prediction_history.append(prediction)
                            
                            # Display prediction
                            print(f"\n📊 Prediction #{prediction_count}")
                            print(f"   Raw EMG: {emg_data['raw']}")
                            print(f"   Clean EMG: {[f'{x:.3f}' for x in clean_emg]}")
                            print(f"   🎯 Gesture: {prediction['gesture']}")
                            print(f"   📊 Confidence: {prediction['confidence']:.3f}")
                            print(f"   ⏰ Timestamp: {timestamp}")
                            
                            # Show top 3 if confidence is low
                            if prediction['confidence'] < 0.7:
                                probs = prediction['probabilities']
                                top_3_idx = np.argsort(probs)[-3:][::-1]
                                print(f"   📊 Top 3:")
                                for i, idx in enumerate(top_3_idx):
                                    print(f"      {i+1}. {self.gesture_names[idx]:12s} - {probs[idx]:.3f}")
                
                time.sleep(0.1)  # Small delay to prevent overwhelming
                
        except KeyboardInterrupt:
            print("\n🛑 Live prediction stopped by user")
        except Exception as e:
            print(f"\n❌ Live prediction error: {e}")
        finally:
            self.is_predicting = False
    
    def simulate_live_data(self):
        """Simulate live EMG data for testing (when Pico not connected)"""
        print("\n🧪 Simulation Mode - Testing without Pico")
        print("📊 Generating simulated EMG data")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        
        # Sample EMG patterns for different gestures
        sample_patterns = [
            {'name': '10-OK_SIGN', 'raw': [1216, 44154, 43322], 'clean': [0.293, 0.543, 0.531]},
            {'name': '6-PEACE', 'raw': [8930, 52764, 53100], 'clean': [0.470, 0.613, 0.628]},
            {'name': '3-POINT', 'raw': [1536, 61951, 61487], 'clean': [0.263, 0.540, 0.531]},
            {'name': '5-FIVE', 'raw': [3456, 48234, 47892], 'clean': [0.354, 0.480, 0.477]},
            {'name': '1-CLOSE', 'raw': [1424, 20869, 21061], 'clean': [0.254, 0.412, 0.407]}
        ]
        
        prediction_count = 0
        
        try:
            while True:
                # Pick random pattern
                pattern = np.random.choice(sample_patterns)
                
                # Add some noise
                noise = np.random.normal(0, 0.02, 3)
                clean_emg = [max(0, min(1, pattern['clean'][i] + noise[i])) for i in range(3)]
                
                timestamp = int(time.time() * 1000)
                
                # Add to buffer
                self.clean_buffer.append(clean_emg)
                
                # Make prediction
                if len(self.clean_buffer) >= self.lag_order:
                    emg_sequence = list(self.clean_buffer)[-self.lag_order:]
                    prediction = self.predict_gesture(emg_sequence, timestamp)
                    
                    if prediction:
                        prediction_count += 1
                        
                        print(f"\n📊 Simulation #{prediction_count}")
                        print(f"   Expected: {pattern['name']}")
                        print(f"   Clean EMG: {[f'{x:.3f}' for x in clean_emg]}")
                        print(f"   🎯 Predicted: {prediction['gesture']}")
                        print(f"   📊 Confidence: {prediction['confidence']:.3f}")
                        
                        correct = "✅" if prediction['gesture'] == pattern['name'] else "❌"
                        print(f"   🎯 Result: {correct}")
                
                time.sleep(1)  # 1 second between predictions
                
        except KeyboardInterrupt:
            print("\n🛑 Simulation stopped by user")

def main():
    """Main function"""
    predictor = LiveEMGPredictor()
    
    # Load model
    if not predictor.load_model():
        return
    
    print(f"\n📋 Live Prediction Options:")
    print(f"1. 🔌 Connect to Pico and start live prediction")
    print(f"2. 🧪 Simulate live data (testing mode)")
    print(f"3. 🚪 Exit")
    
    while True:
        try:
            choice = input(f"\nChoose option (1-3): ").strip()
            
            if choice == '1':
                port = input("Enter COM port (default COM3): ").strip() or 'COM3'
                predictor.port = port
                
                if predictor.connect_pico():
                    predictor.start_live_prediction()
                    predictor.disconnect_pico()
                
            elif choice == '2':
                predictor.simulate_live_data()
                
            elif choice == '3':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
