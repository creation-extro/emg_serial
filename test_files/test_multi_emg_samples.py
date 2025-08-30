#!/usr/bin/env python3
"""
Multi-Sample EMG Tester for AR + LightGBM Model
Input 4-5 EMG signals with timestamps to get gesture predictions
Uses real sequential EMG data for better AR feature creation
"""

import numpy as np
import joblib
import os
from collections import deque

class MultiEMGSampleTester:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.gesture_names = None
        self.lag_order = 15
        
        print("🎯 Multi-Sample EMG Tester for AR + LightGBM")
        print("📊 Input 4-5 EMG Signals with Timestamps")
        print("⚡ Real Sequential Data Processing")
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
    
    def normalize_emg(self, raw_values):
        """Convert raw EMG values to clean normalized values"""
        # Using the same normalization as in your dataset
        clean_values = []
        
        # Approximate ranges from your data
        emg_ranges = {
            'ch1': {'min': 0, 'max': 65535},
            'ch2': {'min': 0, 'max': 65535}, 
            'ch3': {'min': 0, 'max': 65535}
        }
        
        channels = ['ch1', 'ch2', 'ch3']
        for i, channel in enumerate(channels):
            raw_val = raw_values[i]
            min_val = emg_ranges[channel]['min']
            max_val = emg_ranges[channel]['max']
            
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
    
    def parse_emg_line(self, line):
        """Parse EMG data line from CSV format"""
        try:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                timestamp = int(parts[0])
                ch1 = int(parts[1])
                ch2 = int(parts[2]) 
                ch3 = int(parts[3])
                expected_gesture = parts[4]
                
                # Convert to clean EMG
                clean_emg = self.normalize_emg([ch1, ch2, ch3])
                
                return {
                    'timestamp': timestamp,
                    'raw_emg': [ch1, ch2, ch3],
                    'clean_emg': clean_emg,
                    'expected_gesture': expected_gesture
                }
        except Exception as e:
            print(f"❌ Error parsing line: {e}")
        
        return None
    
    def test_multi_samples(self, emg_lines, show_details=True):
        """Test multiple EMG samples in sequence"""
        print(f"\n🎯 Testing {len(emg_lines)} EMG Samples:")
        print("=" * 60)
        
        # Parse all samples
        samples = []
        for i, line in enumerate(emg_lines):
            sample = self.parse_emg_line(line)
            if sample:
                samples.append(sample)
                if show_details:
                    print(f"Sample {i+1}: {sample['expected_gesture']} - Raw: {sample['raw_emg']} - Clean: {[f'{x:.3f}' for x in sample['clean_emg']]}")
        
        if len(samples) < self.lag_order:
            print(f"❌ Need at least {self.lag_order} samples for AR features, got {len(samples)}")
            return
        
        print(f"\n📈 Creating AR sequence from {len(samples)} samples...")
        
        # Use the samples to create AR sequence
        emg_sequence = [sample['clean_emg'] for sample in samples[-self.lag_order:]]
        final_sample = samples[-1]
        
        # Make prediction using the last sample's timestamp
        result = self.predict_gesture(emg_sequence, final_sample['timestamp'])
        
        if result:
            print(f"\n🎯 AR + LightGBM Prediction:")
            print(f"   Input Samples: {len(samples)}")
            print(f"   AR Sequence Length: {len(emg_sequence)}")
            print(f"   Expected: {final_sample['expected_gesture']}")
            print(f"   Predicted: {result['gesture']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   AR Features: {result['features_count']}")
            
            correct = "✅ CORRECT!" if result['gesture'] == final_sample['expected_gesture'] else "❌ WRONG!"
            print(f"   Result: {correct}")
            
            print(f"\n📊 Top 3 Predictions:")
            for i, pred in enumerate(result['top_3'], 1):
                print(f"   {i}. {pred['gesture']:15s} - {pred['confidence']:.3f}")
            
            return result
        else:
            print("❌ Prediction failed")
            return None
    
    def show_example_format(self):
        """Show example input format"""
        print("\n📊 Example Input Format:")
        print("=" * 60)
        print("Enter 4-5 lines of EMG data in CSV format:")
        print("timestamp,ch1,ch2,ch3,expected_gesture")
        print()
        print("Example (3-POINT gesture sequence):")
        print("1752250560,4425,61743,61406,3-POINT")
        print("1752250561,4449,61751,61414,3-POINT") 
        print("1752250562,4473,61759,61422,3-POINT")
        print("1752250563,4497,61767,61430,3-POINT")
        print("1752250564,4521,61775,61438,3-POINT")
        print()
        print("Or use data from your CSV file!")
    
    def interactive_mode(self):
        """Interactive multi-sample testing mode"""
        print("\n🎮 Interactive Multi-Sample EMG Testing")
        print("=" * 60)
        print("Enter 4-5 EMG samples (one per line)")
        print("Format: timestamp,ch1,ch2,ch3,expected_gesture")
        print("Commands: 'example' to see format, 'done' to process, 'quit' to exit")
        print("-" * 60)
        
        while True:
            try:
                print(f"\n📊 Enter your EMG samples (need {self.lag_order} for AR features):")
                emg_lines = []
                
                while True:
                    line = input(f"Sample {len(emg_lines)+1} (or 'done'/'example'/'quit'): ").strip()
                    
                    if line.lower() == 'quit':
                        print("👋 Goodbye!")
                        return
                    
                    if line.lower() == 'example':
                        self.show_example_format()
                        continue
                    
                    if line.lower() == 'done':
                        if len(emg_lines) >= self.lag_order:
                            break
                        else:
                            print(f"❌ Need at least {self.lag_order} samples, got {len(emg_lines)}")
                            continue
                    
                    if line and ',' in line:
                        emg_lines.append(line)
                        print(f"✅ Added sample {len(emg_lines)}")
                        
                        if len(emg_lines) >= 20:  # Limit to prevent too many samples
                            print("📊 Maximum 20 samples reached. Processing...")
                            break
                    else:
                        print("❌ Invalid format. Use: timestamp,ch1,ch2,ch3,expected_gesture")
                
                if emg_lines:
                    self.test_multi_samples(emg_lines)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def test_sample_sequences(self):
        """Test some sample sequences"""
        print("\n🧪 Testing Sample EMG Sequences:")
        print("=" * 60)
        
        # Sample sequence 1: 3-POINT
        print("\n1. Testing 3-POINT sequence:")
        point_sequence = [
            "1752250560,4425,61743,61406,3-POINT",
            "1752250561,4449,61751,61414,3-POINT", 
            "1752250562,4473,61759,61422,3-POINT",
            "1752250563,4497,61767,61430,3-POINT",
            "1752250564,4521,61775,61438,3-POINT"
        ]
        self.test_multi_samples(point_sequence, show_details=False)
        
        # Sample sequence 2: OK_SIGN
        print("\n2. Testing 10-OK_SIGN sequence:")
        ok_sequence = [
            "1752250603,1200,44100,43300,10-OK_SIGN",
            "1752250604,1208,44127,43327,10-OK_SIGN",
            "1752250605,1216,44154,43354,10-OK_SIGN",
            "1752250606,1224,44181,43381,10-OK_SIGN",
            "1752250607,1232,44208,43408,10-OK_SIGN"
        ]
        self.test_multi_samples(ok_sequence, show_details=False)

def main():
    """Main function"""
    tester = MultiEMGSampleTester()
    
    # Load model
    if not tester.load_model():
        return
    
    print(f"\n📋 Multi-Sample Testing Options:")
    print(f"1. 🧪 Test sample sequences")
    print(f"2. 📊 Show input format example")
    print(f"3. 🎮 Interactive mode - Enter your samples")
    print(f"4. 🚪 Exit")
    
    while True:
        try:
            choice = input(f"\nChoose option (1-4): ").strip()
            
            if choice == '1':
                tester.test_sample_sequences()
                
            elif choice == '2':
                tester.show_example_format()
                
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
