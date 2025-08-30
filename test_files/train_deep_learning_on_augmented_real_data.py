#!/usr/bin/env python3
"""
Train Deep Learning Model on Augmented Real EMG Data
Use the augmented real dataset for training a high-performance deep learning model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import os

class AugmentedRealEMGTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.gesture_names = None
        self.history = None
        
        print("🧠 Deep Learning Trainer for Augmented Real EMG Data")
        print("📊 Training on your real EMG data with augmentation")
        print("=" * 70)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def load_augmented_dataset(self, filename='augmented_real_emg_dataset.csv'):
        """Load the augmented real EMG dataset"""
        print(f"📂 Loading augmented real EMG dataset...")

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
    
    def prepare_data(self, data):
        """Prepare data for deep learning with comprehensive features"""
        print(f"\n📈 Preparing data for deep learning...")
        
        # Use the clean EMG features that are already in the dataset
        base_features = ['emg1_clean', 'emg2_clean', 'emg3_clean']
        
        # Create comprehensive engineered features
        print(f"🔧 Creating comprehensive features...")
        
        # Statistical features (already computed in augmentation)
        if 'signal_strength' not in data.columns:
            data['signal_strength'] = data['emg1_clean'] + data['emg2_clean'] + data['emg3_clean']
            data['signal_balance'] = data[base_features].std(axis=1)
        
        # Additional statistical features
        data['emg_mean'] = data[base_features].mean(axis=1)
        data['emg_std'] = data[base_features].std(axis=1)
        data['emg_max'] = data[base_features].max(axis=1)
        data['emg_min'] = data[base_features].min(axis=1)
        data['emg_range'] = data['emg_max'] - data['emg_min']
        data['emg_median'] = data[base_features].median(axis=1)
        data['emg_skew'] = data[base_features].skew(axis=1)
        data['emg_kurt'] = data[base_features].kurtosis(axis=1)
        
        # Channel differences and relationships
        data['emg1_emg2_diff'] = np.abs(data['emg1_clean'] - data['emg2_clean'])
        data['emg1_emg3_diff'] = np.abs(data['emg1_clean'] - data['emg3_clean'])
        data['emg2_emg3_diff'] = np.abs(data['emg2_clean'] - data['emg3_clean'])
        
        # Channel ratios (with small epsilon to avoid division by zero)
        eps = 1e-6
        data['emg1_emg2_ratio'] = data['emg1_clean'] / (data['emg2_clean'] + eps)
        data['emg1_emg3_ratio'] = data['emg1_clean'] / (data['emg3_clean'] + eps)
        data['emg2_emg3_ratio'] = data['emg2_clean'] / (data['emg3_clean'] + eps)
        
        # Power and energy features
        data['emg1_squared'] = data['emg1_clean'] ** 2
        data['emg2_squared'] = data['emg2_clean'] ** 2
        data['emg3_squared'] = data['emg3_clean'] ** 2
        data['total_power'] = data['emg1_squared'] + data['emg2_squared'] + data['emg3_squared']
        data['power_ratio_1'] = data['emg1_squared'] / (data['total_power'] + eps)
        data['power_ratio_2'] = data['emg2_squared'] / (data['total_power'] + eps)
        data['power_ratio_3'] = data['emg3_squared'] / (data['total_power'] + eps)
        
        # Interaction terms
        data['emg1_emg2_product'] = data['emg1_clean'] * data['emg2_clean']
        data['emg1_emg3_product'] = data['emg1_clean'] * data['emg3_clean']
        data['emg2_emg3_product'] = data['emg2_clean'] * data['emg3_clean']
        
        # Dominant channel features
        if 'dominant_channel' not in data.columns:
            data['dominant_channel'] = data[base_features].idxmax(axis=1).map({
                'emg1_clean': 0, 'emg2_clean': 1, 'emg3_clean': 2
            })
        else:
            # Convert existing dominant_channel to numeric if it's string
            if data['dominant_channel'].dtype == 'object':
                data['dominant_channel'] = data['dominant_channel'].map({
                    'emg1_clean': 0, 'emg2_clean': 1, 'emg3_clean': 2
                }).fillna(0)
        
        data['dominant_value'] = data[base_features].max(axis=1)
        data['secondary_value'] = data[base_features].apply(lambda x: x.nlargest(2).iloc[1], axis=1)
        data['weakest_value'] = data[base_features].min(axis=1)
        
        # Normalized features (relative to signal strength)
        data['emg1_normalized'] = data['emg1_clean'] / (data['signal_strength'] + eps)
        data['emg2_normalized'] = data['emg2_clean'] / (data['signal_strength'] + eps)
        data['emg3_normalized'] = data['emg3_clean'] / (data['signal_strength'] + eps)
        
        # All feature columns for training
        feature_columns = [
            # Base features
            'emg1_clean', 'emg2_clean', 'emg3_clean',
            # Statistical features
            'emg_mean', 'emg_std', 'emg_max', 'emg_min', 'emg_range', 'emg_median', 'emg_skew', 'emg_kurt',
            # Signal characteristics
            'signal_strength', 'signal_balance',
            # Channel relationships
            'emg1_emg2_diff', 'emg1_emg3_diff', 'emg2_emg3_diff',
            'emg1_emg2_ratio', 'emg1_emg3_ratio', 'emg2_emg3_ratio',
            # Power features
            'emg1_squared', 'emg2_squared', 'emg3_squared', 'total_power',
            'power_ratio_1', 'power_ratio_2', 'power_ratio_3',
            # Interaction terms
            'emg1_emg2_product', 'emg1_emg3_product', 'emg2_emg3_product',
            # Dominant channel features
            'dominant_channel', 'dominant_value', 'secondary_value', 'weakest_value',
            # Normalized features
            'emg1_normalized', 'emg2_normalized', 'emg3_normalized'
        ]
        
        # Add quality metrics if available
        if 'noise_level' in data.columns:
            feature_columns.append('noise_level')
        
        print(f"✅ Created {len(feature_columns)} comprehensive features")
        
        # Prepare features and labels
        X = data[feature_columns].values
        y = data['gesture'].values
        
        # Handle any NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.gesture_names = list(self.label_encoder.classes_)
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"🎯 Number of classes: {len(self.gesture_names)}")
        print(f"🏷️ Gesture classes: {self.gesture_names}")
        
        return X, y_encoded, feature_columns
    
    def create_advanced_model(self, input_dim, num_classes):
        """Create advanced deep learning model optimized for real EMG data"""
        print(f"\n🏗️ Creating advanced deep learning model...")
        print(f"📊 Input features: {input_dim}")
        print(f"🎯 Output classes: {num_classes}")
        
        model = keras.Sequential([
            # Input layer with batch normalization
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Hidden layers with residual-like connections
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.15),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with optimized settings for real data
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Advanced model created")
        print(f"🏗️ Architecture: 512→256→128→64→32→{num_classes}")
        print(f"⚡ Optimizer: Adam with optimized settings")
        print(f"📊 Total parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, X, y, feature_columns):
        """Train the deep learning model with advanced techniques"""
        print(f"\n🚀 Training advanced deep learning model...")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        # Further split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        
        print(f"📊 Training samples: {len(X_train):,}")
        print(f"📊 Validation samples: {len(X_val):,}")
        print(f"📊 Test samples: {len(X_test):,}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create model
        self.model = self.create_advanced_model(X.shape[1], len(self.gesture_names))
        
        # Advanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_augmented_real_emg_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Calculate class weights for balanced training
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        
        print(f"🏃 Starting advanced training...")
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=150,
            batch_size=128,  # Larger batch size for stable training
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        # Predictions for detailed analysis
        y_pred = self.model.predict(X_test_scaled, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print(f"\n🎯 Advanced Model Results:")
        print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"📊 Test Loss: {test_loss:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_classes, 
                                  target_names=self.gesture_names))
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'feature_columns': feature_columns,
            'y_test': y_test,
            'y_pred': y_pred_classes,
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'class_weights': class_weight_dict
        }
    
    def save_model(self, results, filename='augmented_real_emg_deep_learning_model'):
        """Save the trained model"""
        print(f"\n💾 Saving augmented real EMG deep learning model...")
        
        # Save Keras model
        keras_filename = f"{filename}.h5"
        self.model.save(keras_filename)
        
        # Save preprocessing components
        preprocessing_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'gesture_names': self.gesture_names,
            'feature_columns': results['feature_columns'],
            'test_accuracy': results['test_accuracy'],
            'test_loss': results['test_loss'],
            'class_weights': results['class_weights'],
            'model_type': 'Advanced-Deep-Learning-Real-EMG',
            'architecture': '512-256-128-64-32',
            'total_params': self.model.count_params(),
            'input_features': len(results['feature_columns']),
            'training_data': 'augmented_real_emg_dataset.csv',
            'data_source': 'Real EMG Data with Augmentation'
        }
        
        preprocessing_filename = f"{filename}_preprocessing.pkl"
        joblib.dump(preprocessing_data, preprocessing_filename)
        
        print(f"✅ Keras model saved: {keras_filename}")
        print(f"✅ Preprocessing data saved: {preprocessing_filename}")
        print(f"🎯 Test accuracy: {results['test_accuracy']:.4f}")
        
        return keras_filename, preprocessing_filename

def main():
    """Main training function"""
    trainer = AugmentedRealEMGTrainer()
    
    # Load augmented dataset
    data = trainer.load_augmented_dataset()
    if data is None:
        return
    
    # Prepare data with comprehensive features
    X, y, feature_columns = trainer.prepare_data(data)
    
    # Train advanced model
    results = trainer.train_model(X, y, feature_columns)
    
    # Save model
    keras_file, preprocessing_file = trainer.save_model(results)
    
    print(f"\n🎉 Advanced Deep Learning Model Training Complete!")
    print(f"📊 Final Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
    print(f"💾 Model saved as: {keras_file}")
    print(f"🚀 Ready for real-time prediction with your real EMG data!")
    print(f"📈 This model is trained on YOUR augmented real data!")

if __name__ == "__main__":
    main()
