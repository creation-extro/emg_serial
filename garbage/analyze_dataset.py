#!/usr/bin/env python3
"""
Analyze Your EMG Dataset
Check data quality and find the best approach for your dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class DatasetAnalyzer:
    def __init__(self):
        print("🔍 EMG Dataset Analyzer")
        print("📊 Analyzing Your Dataset Quality")
        print("=" * 60)
    
    def load_and_analyze_data(self):
        """Load and analyze the dataset"""
        print("📂 Loading your EMG dataset...")
        
        data = pd.read_csv('data/combined_emg_data (1).csv')
        print(f"✅ Loaded {len(data):,} samples")
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Shape: {data.shape}")
        print(f"   Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Gesture distribution
        print(f"\n🎯 Gesture Distribution:")
        gesture_counts = data['gesture'].value_counts().sort_index()
        for gesture, count in gesture_counts.items():
            percentage = count / len(data) * 100
            print(f"   {gesture:15s}: {count:5,} samples ({percentage:5.1f}%)")
        
        # Data quality check
        print(f"\n🔍 Data Quality Check:")
        print(f"   Missing values: {data.isnull().sum().sum()}")
        print(f"   Duplicate rows: {data.duplicated().sum()}")
        
        # EMG channel statistics
        print(f"\n📈 EMG Channel Statistics:")
        emg_channels = ['ch1', 'ch2', 'ch3', 'emg1_clean', 'emg2_clean', 'emg3_clean']
        for channel in emg_channels:
            if channel in data.columns:
                stats = data[channel].describe()
                print(f"   {channel:12s}: min={stats['min']:8.3f}, max={stats['max']:8.3f}, mean={stats['mean']:8.3f}, std={stats['std']:8.3f}")
        
        return data
    
    def test_different_approaches(self, data):
        """Test different modeling approaches"""
        print(f"\n🧪 Testing Different Modeling Approaches...")
        
        # Remove RELAX samples
        data_no_relax = data[data['gesture'] != 'RELAX'].copy()
        print(f"📊 After removing RELAX: {len(data_no_relax):,} samples")
        
        approaches = [
            {
                'name': 'Raw EMG Only',
                'features': ['ch1', 'ch2', 'ch3']
            },
            {
                'name': 'Clean EMG Only', 
                'features': ['emg1_clean', 'emg2_clean', 'emg3_clean']
            },
            {
                'name': 'Raw + Clean EMG',
                'features': ['ch1', 'ch2', 'ch3', 'emg1_clean', 'emg2_clean', 'emg3_clean']
            },
            {
                'name': 'All Available Features',
                'features': ['ch1', 'ch2', 'ch3', 'emg1_clean', 'emg2_clean', 'emg3_clean', 
                           'emg1_notch', 'emg2_notch', 'emg3_notch']
            }
        ]
        
        results = []
        
        for approach in approaches:
            print(f"\n🔬 Testing: {approach['name']}")
            
            # Check if all features exist
            available_features = [f for f in approach['features'] if f in data_no_relax.columns]
            if len(available_features) != len(approach['features']):
                missing = set(approach['features']) - set(available_features)
                print(f"   ⚠️  Missing features: {missing}")
                continue
            
            # Prepare data
            X = data_no_relax[available_features].values
            y = data_no_relax['gesture'].values
            
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Train simple Random Forest
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"   📊 Features: {len(available_features)}")
            print(f"   📊 Accuracy: {accuracy:.4f}")
            
            results.append({
                'approach': approach['name'],
                'features': available_features,
                'accuracy': accuracy,
                'model': rf,
                'label_encoder': le
            })
        
        # Find best approach
        best_result = max(results, key=lambda x: x['accuracy'])
        print(f"\n🏆 Best Approach: {best_result['approach']}")
        print(f"   📊 Accuracy: {best_result['accuracy']:.4f}")
        print(f"   📊 Features: {best_result['features']}")
        
        return best_result, data_no_relax
    
    def save_best_model(self, best_result, data):
        """Save the best performing model"""
        print(f"\n💾 Saving Best Model...")
        
        import joblib
        
        model_data = {
            'model': best_result['model'],
            'label_encoder': best_result['label_encoder'],
            'gesture_names': list(best_result['label_encoder'].classes_),
            'features': best_result['features'],
            'accuracy': best_result['accuracy'],
            'model_type': f"Best-{best_result['approach'].replace(' ', '-')}",
            'algorithm': 'RandomForestClassifier'
        }
        
        filename = 'best_emg_model.pkl'
        joblib.dump(model_data, filename)
        
        print(f"✅ Best model saved: {filename}")
        print(f"📊 Model details:")
        print(f"   Approach: {best_result['approach']}")
        print(f"   Accuracy: {best_result['accuracy']:.4f}")
        print(f"   Features: {len(best_result['features'])}")
        
        return filename
    
    def test_sample_predictions(self, best_result):
        """Test with sample predictions"""
        print(f"\n🧪 Testing Sample Predictions...")
        
        # Your known samples
        test_samples = [
            {'raw': [1216, 44154, 43322], 'clean': [0.251244313, 0.532674987, 0.601380024], 'expected': '0-OPEN'},
            {'raw': [2320, 41290, 34440], 'clean': [0.297480812, 0.493316386, 0.491653875], 'expected': '0-OPEN'}
        ]
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\n{i}. Testing sample:")
            print(f"   Raw: {sample['raw']}")
            print(f"   Clean: {sample['clean']}")
            print(f"   Expected: {sample['expected']}")
            
            # Create feature vector based on best approach
            if 'Raw EMG Only' in best_result['approach']:
                features = sample['raw']
            elif 'Clean EMG Only' in best_result['approach']:
                features = sample['clean']
            elif 'Raw + Clean EMG' in best_result['approach']:
                features = sample['raw'] + sample['clean']
            else:
                features = sample['raw'] + sample['clean'] + [0, 0, 0]  # Add dummy notch values
            
            # Predict
            feature_vector = np.array(features).reshape(1, -1)
            prediction = best_result['model'].predict(feature_vector)[0]
            probabilities = best_result['model'].predict_proba(feature_vector)[0]
            
            predicted_gesture = best_result['label_encoder'].classes_[prediction]
            confidence = probabilities[prediction]
            
            correct = "✅ CORRECT!" if predicted_gesture == sample['expected'] else "❌ WRONG!"
            
            print(f"   🎯 Predicted: {predicted_gesture}")
            print(f"   📊 Confidence: {confidence:.3f}")
            print(f"   🎯 Result: {correct}")

def main():
    """Main analysis function"""
    analyzer = DatasetAnalyzer()
    
    # Load and analyze data
    data = analyzer.load_and_analyze_data()
    
    # Test different approaches
    best_result, clean_data = analyzer.test_different_approaches(data)
    
    # Save best model
    if best_result['accuracy'] > 0.5:
        filename = analyzer.save_best_model(best_result, clean_data)
        
        # Test sample predictions
        analyzer.test_sample_predictions(best_result)
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Found working approach: {best_result['approach']}")
        print(f"📊 Accuracy: {best_result['accuracy']:.4f}")
        print(f"💾 Saved as: {filename}")
        
        print(f"\n🚀 Recommendations:")
        if best_result['accuracy'] > 0.8:
            print(f"   🎯 EXCELLENT! Your dataset works very well!")
        elif best_result['accuracy'] > 0.7:
            print(f"   🎯 GOOD! Your dataset is usable for real-time prediction!")
        elif best_result['accuracy'] > 0.6:
            print(f"   🎯 OK! Your dataset needs some improvement but is workable!")
        else:
            print(f"   🎯 POOR! Consider collecting more data or improving sensors!")
        
        print(f"   📊 Use features: {best_result['features']}")
        print(f"   🌲 Use Random Forest with these exact features")
        print(f"   ⚡ Skip complex AR features - they don't help!")
        
    else:
        print(f"\n⚠️  All approaches failed!")
        print(f"🔧 Your dataset might have fundamental issues")
        print(f"💡 Consider:")
        print(f"   1. Checking sensor connections")
        print(f"   2. Collecting more diverse data")
        print(f"   3. Using external datasets")

if __name__ == "__main__":
    main()
