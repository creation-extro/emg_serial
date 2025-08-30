#!/usr/bin/env python3
"""
Real-Time Deep Learning EMG Tester
Use trained neural network for real-time EMG gesture recognition
"""

import serial
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import os

class DeepLearningRealtimeTester:
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
        self.feature_columns = None
        
        print("🧠 Deep Learning Real-Time EMG Tester")
        print("⚡ Neural Network Gesture Recognition")
        print("🎯 High-Accuracy Real-Time Prediction")
        print("=" * 70)
    
    def load_model(self, model_filename='deep_learning_emg_model.h5', 
                   preprocessing_filename='deep_learning_emg_model_preprocessing.pkl'):
        """Load the trained deep learning model"""
        print("📂 Loading deep learning model...")
        
        if not os.path.exists(model_filename):
            print(f"❌ Model not found: {model_filename}")
            print("🔧 Please run: python train_deep_learning_model.py")
            return False
        
        if not os.path.exists(preprocessing_filename):
            print(f"❌ Preprocessing data not found: {preprocessing_filename}")
            return False
        
        try:
            # Load Keras model
            self.model = keras.models.load_model(model_filename)
            
            # Load preprocessing components
            preprocessing_data = joblib.load(preprocessing_filename)
            self.scaler = preprocessing_data['scaler']
            self.label_encoder = preprocessing_data['label_encoder']
            self.gesture_names = preprocessing_data['gesture_names']
            self.feature_columns = preprocessing_data['feature_columns']
            
            print(f"✅ Deep learning model loaded!")
            print(f"📊 Model type: {preprocessing_data['model_type']}")
            print(f"📊 Architecture: {preprocessing_data['architecture']}")
            print(f"📊 Parameters: {preprocessing_data['total_params']:,}")
            print(f"📊 Test Accuracy: {preprocessing_data['test_accuracy']:.4f}")
            print(f"📊 Input Features: {preprocessing_data['input_features']}")
            
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
        
        # Normalize to 0-1 range (same as training data)
        emg1_clean = ch1 / 65535.0
        emg2_clean = ch2 / 65535.0
        emg3_clean = ch3 / 65535.0
        
        return [emg1_clean, emg2_clean, emg3_clean]
    
    def create_features(self, raw_emg, subject_id=0, fatigue_factor=1.0):
        """Create feature vector (same as training)"""
        clean_emg = self.normalize_emg(raw_emg)
        emg1_clean, emg2_clean, emg3_clean = clean_emg
        
        # Create all features (same as training)
        features = {
            'emg1_clean': emg1_clean,
            'emg2_clean': emg2_clean,
            'emg3_clean': emg3_clean,
            
            # Ratios
            'emg1_emg2_ratio': emg1_clean / (emg2_clean + 1e-6),
            'emg1_emg3_ratio': emg1_clean / (emg3_clean + 1e-6),
            'emg2_emg3_ratio': emg2_clean / (emg3_clean + 1e-6),
            
            # Statistical features
            'emg_mean': np.mean([emg1_clean, emg2_clean, emg3_clean]),
            'emg_std': np.std([emg1_clean, emg2_clean, emg3_clean]),
            'emg_max': max(emg1_clean, emg2_clean, emg3_clean),
            'emg_min': min(emg1_clean, emg2_clean, emg3_clean),
            'emg_range': max(emg1_clean, emg2_clean, emg3_clean) - min(emg1_clean, emg2_clean, emg3_clean),
            
            # Differences
            'emg1_emg2_diff': abs(emg1_clean - emg2_clean),
            'emg1_emg3_diff': abs(emg1_clean - emg3_clean),
            'emg2_emg3_diff': abs(emg2_clean - emg3_clean),
            
            # Products
            'emg1_emg2_product': emg1_clean * emg2_clean,
            'emg1_emg3_product': emg1_clean * emg3_clean,
            'emg2_emg3_product': emg2_clean * emg3_clean,
            
            # Power features
            'emg1_squared': emg1_clean ** 2,
            'emg2_squared': emg2_clean ** 2,
            'emg3_squared': emg3_clean ** 2,
            'total_power': emg1_clean**2 + emg2_clean**2 + emg3_clean**2,
            
            # Subject and fatigue
            'subject_encoded': subject_id / 4.0,
            'fatigue_factor': fatigue_factor
        }
        
        # Create feature vector in correct order
        feature_vector = [features[col] for col in self.feature_columns]
        
        return np.array(feature_vector)
    
    def predict_gesture(self, raw_emg):
        """Predict gesture using deep learning model"""
        if self.model is None:
            return None
        
        try:
            # Create features
            features = self.create_features(raw_emg)
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            pred_proba = self.model.predict(features_scaled, verbose=0)[0]
            pred_label = np.argmax(pred_proba)
            
            gesture_name = self.gesture_names[pred_label]
            confidence = pred_proba[pred_label]
            
            return {
                'gesture': gesture_name,
                'confidence': confidence,
                'probabilities': pred_proba,
                'raw_emg': raw_emg,
                'clean_emg': self.normalize_emg(raw_emg),
                'features_count': len(features[0])
            }
            
        except Exception as e:
            print(f"❌ Deep learning prediction error: {e}")
            return None
    
    def test_known_samples(self):
        """Test with known samples"""
        print("\n🧪 Testing Known Samples with Deep Learning:")
        print("=" * 60)
        
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
            print(f"\n{i}. Testing {sample['name']} with Deep Learning:")
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
                
                print(f"   Clean EMG: {[f'{x:.3f}' for x in result['clean_emg']]}")
                print(f"   Features: {result['features_count']}")
                print(f"   🧠 Predicted: {predicted}")
                print(f"   📊 Confidence: {confidence:.3f}")
                print(f"   🎯 Result: {status}")
                
                # Show top 3
                probs = result['probabilities']
                top_3_idx = np.argsort(probs)[-3:][::-1]
                print(f"   📊 Top 3:")
                for j, idx in enumerate(top_3_idx):
                    print(f"      {j+1}. {self.gesture_names[idx]:12s} - {probs[idx]:.3f}")
            else:
                print(f"   ❌ Deep learning prediction failed")
        
        accuracy = correct_count / len(test_samples) * 100
        print(f"\n🧠 Deep Learning Test Results:")
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
    
    def start_deep_learning_prediction(self):
        """Start real-time deep learning prediction"""
        if not self.is_connected or self.model is None:
            print("❌ Not ready for deep learning prediction")
            return
        
        print("\n🧠 Starting Deep Learning Real-Time Prediction...")
        print("⚡ Neural Network Processing...")
        print("💪 Perform hand gestures!")
        print("🛑 Press Ctrl+C to stop")
        print("-" * 70)
        
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
                                
                                # Predict gesture using deep learning
                                result = self.predict_gesture([ch1, ch2, ch3])
                                
                                if result and result['confidence'] > 0.3:  # Lower threshold for your signals
                                    prediction_count += 1
                                    
                                    print(f"\n🧠 Deep Learning Prediction #{prediction_count}")
                                    print(f"   Raw EMG: {result['raw_emg']}")
                                    print(f"   Clean EMG: {[f'{x:.3f}' for x in result['clean_emg']]}")
                                    print(f"   Features: {result['features_count']}")
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
                
                time.sleep(0.05)  # Small delay for processing
                
        except KeyboardInterrupt:
            print(f"\n🛑 Deep learning prediction stopped")
            print(f"📊 Total predictions: {prediction_count}")
        finally:
            if self.serial_conn:
                self.serial_conn.close()

def main():
    """Main function"""
    tester = DeepLearningRealtimeTester()
    
    # Load deep learning model
    if not tester.load_model():
        return
    
    print(f"\n📋 Deep Learning Testing Options:")
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
                    print("\n🧠 Deep Learning Instructions:")
                    print("1. Neural network processes complex patterns")
                    print("2. Hold each gesture for 2-3 seconds")
                    print("3. Expect high accuracy predictions")
                    print("4. Try: OK sign, Peace, Point, Fist, Open hand")
                    
                    input("Press Enter to start deep learning prediction...")
                    tester.start_deep_learning_prediction()
                
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
