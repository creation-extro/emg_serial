#!/usr/bin/env python3
"""
Train Combined EXG Data on Various ML Models including AR Models
Test multiple algorithms and compare accuracy results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class CombinedEXGModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        print("🧠 Combined EXG Data - Multi-Model Trainer")
        print("📊 Testing various ML algorithms including AR models")
        print("=" * 70)
    
    def load_combined_dataset(self, filename='combined_exg_data.csv'):
        """Load the combined EXG dataset"""
        print(f"📂 Loading combined EXG dataset: {filename}")
        
        if not os.path.exists(filename):
            print(f"❌ Dataset not found: {filename}")
            return None
        
        data = pd.read_csv(filename)
        print(f"✅ Loaded dataset: {data.shape}")
        print(f"📊 Total samples: {len(data):,}")
        print(f"📊 Features: {len(data.columns)}")
        
        # Show label distribution
        print(f"🏷️ Label distribution:")
        label_counts = data['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            percentage = count / len(data) * 100
            print(f"   Label {label}: {count:,} samples ({percentage:.1f}%)")
        
        return data
    
    def create_ar_features(self, data, lag_order=5):
        """Create AutoRegressive features from time series data"""
        print(f"\n🔧 Creating AR features with lag order {lag_order}...")
        
        # Sort by source file and reset index to ensure proper time series order
        data_sorted = data.sort_values(['file_id', data.index]).reset_index(drop=True)
        
        ar_features = []
        ar_labels = []
        
        # Create AR features for each file separately
        for file_id in data_sorted['file_id'].unique():
            file_data = data_sorted[data_sorted['file_id'] == file_id].reset_index(drop=True)
            
            # Extract time series features
            emg1_series = file_data['raw_emg1'].values
            emg2_series = file_data['raw_emg2'].values
            norm1_series = file_data['normalized_emg1'].values
            norm2_series = file_data['normalized_emg2'].values
            deriv1_series = file_data['derivative_emg1'].values
            deriv2_series = file_data['derivative_emg2'].values
            
            # Create AR features starting from lag_order
            for i in range(lag_order, len(file_data)):
                # AR features for each signal
                ar_feature_row = []
                
                # Raw EMG AR features
                for lag in range(1, lag_order + 1):
                    ar_feature_row.extend([
                        emg1_series[i - lag],
                        emg2_series[i - lag]
                    ])
                
                # Normalized EMG AR features
                for lag in range(1, lag_order + 1):
                    ar_feature_row.extend([
                        norm1_series[i - lag],
                        norm2_series[i - lag]
                    ])
                
                # Derivative AR features
                for lag in range(1, lag_order + 1):
                    ar_feature_row.extend([
                        deriv1_series[i - lag],
                        deriv2_series[i - lag]
                    ])
                
                # Current values
                ar_feature_row.extend([
                    emg1_series[i], emg2_series[i],
                    norm1_series[i], norm2_series[i],
                    deriv1_series[i], deriv2_series[i]
                ])
                
                ar_features.append(ar_feature_row)
                ar_labels.append(file_data.iloc[i]['label'])
        
        ar_features = np.array(ar_features)
        ar_labels = np.array(ar_labels)
        
        print(f"✅ Created AR features: {ar_features.shape}")
        print(f"📊 AR samples: {len(ar_features):,}")
        print(f"📊 AR features per sample: {ar_features.shape[1]}")
        
        return ar_features, ar_labels
    
    def prepare_standard_features(self, data):
        """Prepare standard features (non-AR)"""
        print(f"\n📈 Preparing standard features...")
        
        # Select feature columns (exclude metadata)
        feature_columns = ['raw_emg1', 'derivative_emg1', 'normalized_emg1', 
                          'raw_emg2', 'derivative_emg2', 'normalized_emg2']
        
        X = data[feature_columns].values
        y = data['label'].values
        
        print(f"✅ Standard features prepared: {X.shape}")
        return X, y, feature_columns
    
    def initialize_models(self):
        """Initialize various ML models"""
        print(f"\n🏗️ Initializing ML models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
            'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        }
        
        print(f"✅ Initialized {len(self.models)} models")
        for name in self.models.keys():
            print(f"   • {name}")
    
    def train_and_evaluate_models(self, X, y, model_type="Standard"):
        """Train and evaluate all models"""
        print(f"\n🚀 Training and evaluating {model_type} models...")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training samples: {len(self.X_train):,}")
        print(f"📊 Test samples: {len(self.X_test):,}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, self.y_train)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, self.y_train, cv=5)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'y_pred': y_pred
                }
                
                print(f"   ✅ {name}: {accuracy:.4f} (CV: {cv_mean:.4f} ± {cv_std:.4f})")
                
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
                continue
        
        return results
    
    def optimize_best_models(self, results, X_train_scaled, X_test_scaled):
        """Optimize the top performing models"""
        print(f"\n🔧 Optimizing top performing models...")
        
        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        top_3 = sorted_results[:3]
        
        print(f"🏆 Top 3 models for optimization:")
        for i, (name, result) in enumerate(top_3, 1):
            print(f"   {i}. {name}: {result['accuracy']:.4f}")
        
        optimized_results = {}
        
        # Optimize Random Forest if in top 3
        for name, result in top_3:
            if 'Random Forest' in name:
                print(f"\n🔧 Optimizing Random Forest...")
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                
                rf = RandomForestClassifier(random_state=42, n_jobs=-1)
                grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train_scaled, self.y_train)
                
                optimized_pred = grid_search.predict(X_test_scaled)
                optimized_accuracy = accuracy_score(self.y_test, optimized_pred)
                
                optimized_results['Optimized Random Forest'] = {
                    'model': grid_search.best_estimator_,
                    'accuracy': optimized_accuracy,
                    'best_params': grid_search.best_params_,
                    'y_pred': optimized_pred
                }
                
                print(f"   ✅ Optimized RF: {optimized_accuracy:.4f}")
                print(f"   🔧 Best params: {grid_search.best_params_}")
                break
        
        return optimized_results
    
    def create_results_summary(self, standard_results, ar_results, optimized_results=None):
        """Create comprehensive results summary"""
        print(f"\n📊 Creating Results Summary...")
        
        # Combine all results
        all_results = {}
        
        # Add standard results
        for name, result in standard_results.items():
            all_results[f"Standard - {name}"] = result
        
        # Add AR results
        for name, result in ar_results.items():
            all_results[f"AR - {name}"] = result
        
        # Add optimized results
        if optimized_results:
            for name, result in optimized_results.items():
                all_results[name] = result
        
        # Create summary DataFrame
        summary_data = []
        for name, result in all_results.items():
            summary_data.append({
                'Model': name,
                'Test Accuracy': result['accuracy'],
                'CV Mean': result.get('cv_mean', 0),
                'CV Std': result.get('cv_std', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test Accuracy', ascending=False)
        
        print(f"\n🏆 FINAL RESULTS RANKING:")
        print("=" * 80)
        print(f"{'Rank':<4} {'Model':<35} {'Test Acc':<10} {'CV Mean':<10} {'CV Std':<10}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(summary_df.iterrows(), 1):
            print(f"{i:<4} {row['Model']:<35} {row['Test Accuracy']:<10.4f} {row['CV Mean']:<10.4f} {row['CV Std']:<10.4f}")
        
        return summary_df, all_results
    
    def save_best_model(self, all_results, summary_df):
        """Save the best performing model"""
        best_model_name = summary_df.iloc[0]['Model']
        best_result = all_results[best_model_name]
        
        print(f"\n💾 Saving best model: {best_model_name}")
        
        model_data = {
            'model': best_result['model'],
            'scaler': self.scaler,
            'model_name': best_model_name,
            'test_accuracy': best_result['accuracy'],
            'cv_mean': best_result.get('cv_mean', 0),
            'cv_std': best_result.get('cv_std', 0),
            'training_data': 'combined_exg_data.csv',
            'model_type': 'EXG-Classification'
        }
        
        filename = 'best_combined_exg_model.pkl'
        joblib.dump(model_data, filename)
        
        print(f"✅ Best model saved: {filename}")
        print(f"🎯 Model: {best_model_name}")
        print(f"📊 Accuracy: {best_result['accuracy']:.4f}")
        
        return filename

def main():
    """Main training function"""
    trainer = CombinedEXGModelTrainer()
    
    # Load dataset
    data = trainer.load_combined_dataset()
    if data is None:
        return
    
    # Initialize models
    trainer.initialize_models()
    
    # Prepare standard features
    X_standard, y_standard, feature_cols = trainer.prepare_standard_features(data)
    
    # Train standard models
    print(f"\n" + "="*70)
    print(f"🔄 TRAINING STANDARD MODELS")
    print(f"="*70)
    standard_results = trainer.train_and_evaluate_models(X_standard, y_standard, "Standard")
    
    # Create AR features and train AR models
    print(f"\n" + "="*70)
    print(f"🔄 TRAINING AR (AUTOREGRESSIVE) MODELS")
    print(f"="*70)
    X_ar, y_ar = trainer.create_ar_features(data, lag_order=5)
    ar_results = trainer.train_and_evaluate_models(X_ar, y_ar, "AR")
    
    # Optimize best models
    X_train_scaled = trainer.scaler.transform(trainer.X_train)
    X_test_scaled = trainer.scaler.transform(trainer.X_test)
    optimized_results = trainer.optimize_best_models(standard_results, X_train_scaled, X_test_scaled)
    
    # Create final summary
    summary_df, all_results = trainer.create_results_summary(standard_results, ar_results, optimized_results)
    
    # Save best model
    best_model_file = trainer.save_best_model(all_results, summary_df)
    
    print(f"\n🎉 Training Complete!")
    print(f"📊 Total models tested: {len(all_results)}")
    print(f"🏆 Best model: {summary_df.iloc[0]['Model']}")
    print(f"🎯 Best accuracy: {summary_df.iloc[0]['Test Accuracy']:.4f}")
    print(f"💾 Saved as: {best_model_file}")

if __name__ == "__main__":
    main()
