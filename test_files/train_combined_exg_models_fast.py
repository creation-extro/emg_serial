#!/usr/bin/env python3
"""
Fast Training of Combined EXG Data on Various ML Models
Optimized for speed with subset sampling and efficient AR models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

class FastEXGModelTrainer:
    def __init__(self):
        self.results = {}
        self.scaler = None
        
        print("⚡ Fast Combined EXG Data - Multi-Model Trainer")
        print("📊 Optimized for speed with comprehensive model testing")
        print("=" * 70)
    
    def load_and_sample_dataset(self, filename='combined_exg_data.csv', sample_size=20000):
        """Load dataset and create balanced sample for faster training"""
        print(f"📂 Loading combined EXG dataset: {filename}")
        
        if not os.path.exists(filename):
            print(f"❌ Dataset not found: {filename}")
            return None
        
        data = pd.read_csv(filename)
        print(f"✅ Loaded full dataset: {data.shape}")
        
        # Create balanced sample
        if len(data) > sample_size:
            print(f"🔄 Creating balanced sample of {sample_size:,} samples...")
            
            # Sample equally from each label
            samples_per_label = sample_size // 2
            
            label_0 = data[data['label'] == 0].sample(n=samples_per_label, random_state=42)
            label_1 = data[data['label'] == 1].sample(n=samples_per_label, random_state=42)
            
            data = pd.concat([label_0, label_1]).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"✅ Sampled dataset: {data.shape}")
        
        # Show label distribution
        print(f"🏷️ Label distribution:")
        label_counts = data['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            percentage = count / len(data) * 100
            print(f"   Label {label}: {count:,} samples ({percentage:.1f}%)")
        
        return data
    
    def create_simple_ar_features(self, data, lag_order=3):
        """Create simple AR features for faster processing"""
        print(f"\n🔧 Creating simple AR features with lag order {lag_order}...")
        
        # Sort by file_id to maintain time series order within each file
        data_sorted = data.sort_values(['file_id']).reset_index(drop=True)
        
        ar_features = []
        ar_labels = []
        
        # Process each file separately
        for file_id in data_sorted['file_id'].unique():
            file_data = data_sorted[data_sorted['file_id'] == file_id].reset_index(drop=True)
            
            if len(file_data) < lag_order + 1:
                continue
            
            # Extract main signals
            raw_emg1 = file_data['raw_emg1'].values
            raw_emg2 = file_data['raw_emg2'].values
            norm_emg1 = file_data['normalized_emg1'].values
            norm_emg2 = file_data['normalized_emg2'].values
            
            # Create AR features
            for i in range(lag_order, len(file_data)):
                ar_row = []
                
                # Add lagged values
                for lag in range(1, lag_order + 1):
                    ar_row.extend([
                        raw_emg1[i - lag],
                        raw_emg2[i - lag],
                        norm_emg1[i - lag],
                        norm_emg2[i - lag]
                    ])
                
                # Add current values
                ar_row.extend([
                    raw_emg1[i], raw_emg2[i],
                    norm_emg1[i], norm_emg2[i]
                ])
                
                ar_features.append(ar_row)
                ar_labels.append(file_data.iloc[i]['label'])
        
        ar_features = np.array(ar_features)
        ar_labels = np.array(ar_labels)
        
        print(f"✅ Created AR features: {ar_features.shape}")
        print(f"📊 AR samples: {len(ar_features):,}")
        
        return ar_features, ar_labels
    
    def prepare_standard_features(self, data):
        """Prepare standard features"""
        print(f"\n📈 Preparing standard features...")
        
        feature_columns = ['raw_emg1', 'derivative_emg1', 'normalized_emg1', 
                          'raw_emg2', 'derivative_emg2', 'normalized_emg2']
        
        X = data[feature_columns].values
        y = data['label'].values
        
        print(f"✅ Standard features: {X.shape}")
        return X, y
    
    def train_models_fast(self, X, y, model_type="Standard"):
        """Train models with optimized settings for speed"""
        print(f"\n🚀 Fast training {model_type} models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fast model configurations
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=20),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=200),
            'SVM (Linear)': SVC(kernel='linear', random_state=42, C=1.0)
        }
        
        results = {}
        
        for name, model in models.items():
            start_time = time.time()
            print(f"🔄 Training {name}...")
            
            try:
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Test
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Quick CV (3-fold for speed)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3)
                
                training_time = time.time() - start_time
                
                results[name] = {
                    'model': model,
                    'scaler': scaler,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_time': training_time,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                
                print(f"   ✅ {name}: {accuracy:.4f} (CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}) [{training_time:.1f}s]")
                
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
        
        return results
    
    def create_comprehensive_summary(self, standard_results, ar_results):
        """Create comprehensive results summary"""
        print(f"\n📊 Creating Comprehensive Results Summary...")
        
        all_results = []
        
        # Standard results
        for name, result in standard_results.items():
            all_results.append({
                'Model Type': 'Standard',
                'Model Name': name,
                'Test Accuracy': result['accuracy'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std'],
                'Training Time (s)': result['training_time']
            })
        
        # AR results
        for name, result in ar_results.items():
            all_results.append({
                'Model Type': 'AR (AutoRegressive)',
                'Model Name': name,
                'Test Accuracy': result['accuracy'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std'],
                'Training Time (s)': result['training_time']
            })
        
        # Create DataFrame and sort
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('Test Accuracy', ascending=False)
        
        return results_df, {**standard_results, **ar_results}
    
    def display_results(self, results_df):
        """Display comprehensive results"""
        print(f"\n🏆 COMPREHENSIVE MODEL PERFORMANCE RESULTS")
        print("=" * 100)
        print(f"{'Rank':<4} {'Type':<20} {'Model':<20} {'Test Acc':<10} {'CV Mean':<10} {'CV Std':<8} {'Time(s)':<8}")
        print("-" * 100)
        
        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            print(f"{i:<4} {row['Model Type']:<20} {row['Model Name']:<20} "
                  f"{row['Test Accuracy']:<10.4f} {row['CV Mean']:<10.4f} "
                  f"{row['CV Std']:<8.4f} {row['Training Time (s)']:<8.1f}")
        
        # Best models summary
        print(f"\n🥇 TOP 5 BEST PERFORMING MODELS:")
        print("-" * 60)
        top_5 = results_df.head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. {row['Model Type']} - {row['Model Name']}: {row['Test Accuracy']:.4f}")
        
        # Best by category
        print(f"\n🏆 BEST BY CATEGORY:")
        print("-" * 40)
        
        standard_best = results_df[results_df['Model Type'] == 'Standard'].iloc[0]
        ar_best = results_df[results_df['Model Type'] == 'AR (AutoRegressive)'].iloc[0]
        
        print(f"🔹 Best Standard Model: {standard_best['Model Name']} - {standard_best['Test Accuracy']:.4f}")
        print(f"🔹 Best AR Model: {ar_best['Model Name']} - {ar_best['Test Accuracy']:.4f}")
        
        return results_df
    
    def save_best_models(self, all_results, results_df):
        """Save the best performing models"""
        print(f"\n💾 Saving best models...")
        
        # Overall best
        best_overall = results_df.iloc[0]
        best_model_key = f"{best_overall['Model Name']}"
        
        # Find the actual result
        best_result = None
        for name, result in all_results.items():
            if name == best_model_key:
                best_result = result
                break
        
        if best_result:
            model_data = {
                'model': best_result['model'],
                'scaler': best_result['scaler'],
                'model_name': f"{best_overall['Model Type']} - {best_overall['Model Name']}",
                'test_accuracy': best_result['accuracy'],
                'cv_mean': best_result['cv_mean'],
                'cv_std': best_result['cv_std'],
                'training_time': best_result['training_time'],
                'training_data': 'combined_exg_data.csv',
                'model_type': 'EXG-Binary-Classification'
            }
            
            filename = 'best_combined_exg_model_fast.pkl'
            joblib.dump(model_data, filename)
            
            print(f"✅ Best model saved: {filename}")
            print(f"🎯 Model: {best_overall['Model Type']} - {best_overall['Model Name']}")
            print(f"📊 Accuracy: {best_result['accuracy']:.4f}")
            
            return filename
        
        return None

def main():
    """Main training function"""
    trainer = FastEXGModelTrainer()
    
    # Load and sample dataset
    data = trainer.load_and_sample_dataset(sample_size=20000)
    if data is None:
        return
    
    # Prepare standard features
    X_standard, y_standard = trainer.prepare_standard_features(data)
    
    # Train standard models
    print(f"\n" + "="*70)
    print(f"🔄 TRAINING STANDARD MODELS")
    print(f"="*70)
    standard_results = trainer.train_models_fast(X_standard, y_standard, "Standard")
    
    # Create AR features and train AR models
    print(f"\n" + "="*70)
    print(f"🔄 TRAINING AR (AUTOREGRESSIVE) MODELS")
    print(f"="*70)
    X_ar, y_ar = trainer.create_simple_ar_features(data, lag_order=3)
    ar_results = trainer.train_models_fast(X_ar, y_ar, "AR")
    
    # Create comprehensive summary
    results_df, all_results = trainer.create_comprehensive_summary(standard_results, ar_results)
    
    # Display results
    final_results = trainer.display_results(results_df)
    
    # Save best models
    best_model_file = trainer.save_best_models(all_results, results_df)
    
    print(f"\n🎉 FAST TRAINING COMPLETE!")
    print(f"📊 Total models tested: {len(all_results)}")
    print(f"🏆 Best overall: {results_df.iloc[0]['Model Type']} - {results_df.iloc[0]['Model Name']}")
    print(f"🎯 Best accuracy: {results_df.iloc[0]['Test Accuracy']:.4f}")
    if best_model_file:
        print(f"💾 Saved as: {best_model_file}")

if __name__ == "__main__":
    main()
