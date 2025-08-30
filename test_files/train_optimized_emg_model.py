#!/usr/bin/env python3
"""
Train Optimized EMG Model using Random Forest
Focus on simplicity and effectiveness for EMG gesture classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

class OptimizedEMGTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.gesture_names = None
        
        print("🎯 Optimized EMG Gesture Trainer")
        print("📊 Using Random Forest for reliable performance")
        print("=" * 60)
        
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def load_real_dataset(self, filename='data/emg_data_no_relax.csv'):
        """Load the real EMG dataset"""
        print(f"📂 Loading real EMG dataset...")

        if not os.path.exists(filename):
            print(f"❌ Dataset not found: {filename}")
            return None
        
        data = pd.read_csv(filename)
        print(f"✅ Loaded dataset: {data.shape}")
        print(f"📊 Samples: {len(data):,}")
        print(f"📊 Features: {len(data.columns)}")
        print(f"📊 Gestures: {data['gesture'].nunique()}")
        
        # Show gesture distribution
        print(f"🏷️ Gesture distribution:")
        gesture_counts = data['gesture'].value_counts().sort_index()
        for gesture, count in gesture_counts.items():
            print(f"   {gesture:15s}: {count:,} samples")
        
        return data
    
    def create_optimized_features(self, data):
        """Create optimized features for Random Forest"""
        print(f"\n🔧 Creating optimized features...")
        
        # Use the clean EMG features that are already in the dataset
        base_features = ['emg1_clean', 'emg2_clean', 'emg3_clean']
        
        # Simple but effective features
        data['emg_mean'] = data[base_features].mean(axis=1)
        data['emg_std'] = data[base_features].std(axis=1)
        data['emg_max'] = data[base_features].max(axis=1)
        data['emg_min'] = data[base_features].min(axis=1)
        data['emg_range'] = data['emg_max'] - data['emg_min']
        
        # Channel differences (important for gesture discrimination)
        data['emg1_emg2_diff'] = np.abs(data['emg1_clean'] - data['emg2_clean'])
        data['emg1_emg3_diff'] = np.abs(data['emg1_clean'] - data['emg3_clean'])
        data['emg2_emg3_diff'] = np.abs(data['emg2_clean'] - data['emg3_clean'])
        
        # Channel ratios (with small epsilon to avoid division by zero)
        eps = 1e-6
        data['emg1_emg2_ratio'] = data['emg1_clean'] / (data['emg2_clean'] + eps)
        data['emg1_emg3_ratio'] = data['emg1_clean'] / (data['emg3_clean'] + eps)
        data['emg2_emg3_ratio'] = data['emg2_clean'] / (data['emg3_clean'] + eps)
        
        # Power features
        data['total_power'] = data['emg1_clean']**2 + data['emg2_clean']**2 + data['emg3_clean']**2
        
        # Dominant channel (which channel has highest activation)
        data['dominant_channel'] = data[base_features].idxmax(axis=1).map({
            'emg1_clean': 0, 'emg2_clean': 1, 'emg3_clean': 2
        })
        
        # Feature columns for training (keep it simple but effective)
        feature_columns = [
            'emg1_clean', 'emg2_clean', 'emg3_clean',
            'emg_mean', 'emg_std', 'emg_max', 'emg_min', 'emg_range',
            'emg1_emg2_diff', 'emg1_emg3_diff', 'emg2_emg3_diff',
            'emg1_emg2_ratio', 'emg1_emg3_ratio', 'emg2_emg3_ratio',
            'total_power', 'dominant_channel'
        ]
        
        print(f"✅ Created {len(feature_columns)} optimized features")
        
        # Prepare features and labels
        X = data[feature_columns].values
        y = data['gesture'].values
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.gesture_names = list(self.label_encoder.classes_)
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"🎯 Number of classes: {len(self.gesture_names)}")
        print(f"🏷️ Gesture classes: {self.gesture_names}")
        
        return X, y_encoded, feature_columns
    
    def train_optimized_model(self, X, y, feature_columns):
        """Train optimized Random Forest model"""
        print(f"\n🚀 Training optimized Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training samples: {len(X_train):,}")
        print(f"📊 Test samples: {len(X_test):,}")
        
        # Scale features (Random Forest doesn't strictly need it, but it helps)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Optimized Random Forest parameters
        print(f"🔧 Finding optimal hyperparameters...")
        
        # Quick grid search for best parameters
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Use best model
        self.model = grid_search.best_estimator_
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        test_accuracy = self.model.score(X_test_scaled, y_test)
        y_pred = self.model.predict(X_test_scaled)
        
        print(f"\n🎯 Optimized Model Results:")
        print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"📊 CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.gesture_names))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:20s}: {row['importance']:.4f}")
        
        return {
            'test_accuracy': test_accuracy,
            'cv_scores': cv_scores,
            'feature_columns': feature_columns,
            'feature_importance': feature_importance,
            'y_test': y_test,
            'y_pred': y_pred,
            'best_params': grid_search.best_params_
        }
    
    def save_model(self, results, filename='optimized_emg_model'):
        """Save the optimized model"""
        print(f"\n💾 Saving optimized model...")
        
        # Save model and preprocessing
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'gesture_names': self.gesture_names,
            'feature_columns': results['feature_columns'],
            'test_accuracy': results['test_accuracy'],
            'cv_mean': results['cv_scores'].mean(),
            'cv_std': results['cv_scores'].std(),
            'model_type': 'Optimized-Random-Forest',
            'best_params': results['best_params'],
            'training_data': 'emg_data_no_relax.csv'
        }
        
        model_filename = f"{filename}.pkl"
        joblib.dump(model_data, model_filename)
        
        print(f"✅ Model saved: {model_filename}")
        print(f"🎯 Test accuracy: {results['test_accuracy']:.4f}")
        print(f"📊 CV accuracy: {results['cv_scores'].mean():.4f} ± {results['cv_scores'].std():.4f}")
        
        return model_filename
    
    def create_visualization(self, results):
        """Create visualization of results"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Confusion matrix
            plt.subplot(2, 2, 1)
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.gesture_names,
                       yticklabels=self.gesture_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            # Feature importance
            plt.subplot(2, 2, 2)
            top_features = results['feature_importance'].head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Feature Importance')
            plt.gca().invert_yaxis()
            
            # Accuracy comparison
            plt.subplot(2, 2, 3)
            accuracies = [results['test_accuracy'], results['cv_scores'].mean()]
            labels = ['Test Accuracy', 'CV Accuracy']
            colors = ['skyblue', 'lightcoral']
            bars = plt.bar(labels, accuracies, color=colors)
            plt.ylabel('Accuracy')
            plt.title('Model Performance')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            # CV scores distribution
            plt.subplot(2, 2, 4)
            plt.hist(results['cv_scores'], bins=5, alpha=0.7, color='green')
            plt.axvline(results['cv_scores'].mean(), color='red', linestyle='--',
                       label=f'Mean: {results["cv_scores"].mean():.3f}')
            plt.xlabel('CV Accuracy')
            plt.ylabel('Frequency')
            plt.title('Cross-Validation Scores Distribution')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('optimized_emg_model_results.png', dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved: optimized_emg_model_results.png")
            
        except ImportError:
            print("⚠️  Matplotlib/Seaborn not available - skipping visualization")

def main():
    """Main training function"""
    trainer = OptimizedEMGTrainer()
    
    # Load real EMG dataset
    data = trainer.load_real_dataset()
    if data is None:
        return
    
    # Create optimized features
    X, y, feature_columns = trainer.create_optimized_features(data)
    
    # Train optimized model
    results = trainer.train_optimized_model(X, y, feature_columns)
    
    # Save model
    model_file = trainer.save_model(results)
    
    # Create visualization
    trainer.create_visualization(results)
    
    print(f"\n🎉 Optimized EMG Model Training Complete!")
    print(f"📊 Final Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
    print(f"📊 CV Accuracy: {results['cv_scores'].mean():.4f} ± {results['cv_scores'].std():.4f}")
    print(f"💾 Model saved as: {model_file}")
    print(f"🚀 Ready for real-time prediction!")

if __name__ == "__main__":
    main()
