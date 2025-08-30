#!/usr/bin/env python3
"""
Train Deep Learning Model on Systematic EMG Dataset
Use TensorFlow/Keras to train neural network on 100K EMG samples
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os

class DeepLearningEMGTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.history = None
        self.gesture_names = None
        
        print("🧠 Deep Learning EMG Model Trainer")
        print("⚡ TensorFlow/Keras Neural Network")
        print("📊 Training on 100K Systematic EMG Dataset")
        print("=" * 70)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def load_dataset(self, filename='domain_adapted_emg_dataset.csv'):
        """Load the domain-adapted EMG dataset"""
        print(f"📂 Loading domain-adapted EMG dataset...")

        if not os.path.exists(filename):
            print(f"❌ Dataset not found: {filename}")
            print(f"🔧 Please run: python create_domain_adapted_dataset.py")
            return None
        
        data = pd.read_csv(filename)
        print(f"✅ Loaded dataset: {data.shape}")
        print(f"📊 Samples: {len(data):,}")
        print(f"📊 Features: {len(data.columns)}")
        print(f"📊 Gestures: {data['gesture'].nunique()}")
        
        return data
    
    def prepare_data(self, data):
        """Prepare data for deep learning"""
        print(f"\n📈 Preparing data for deep learning...")
        
        # Feature columns (EMG channels + engineered features)
        base_features = ['emg1_clean', 'emg2_clean', 'emg3_clean']
        
        # Create additional features for deep learning
        print(f"🔧 Creating engineered features...")
        
        # Ratios
        data['emg1_emg2_ratio'] = data['emg1_clean'] / (data['emg2_clean'] + 1e-6)
        data['emg1_emg3_ratio'] = data['emg1_clean'] / (data['emg3_clean'] + 1e-6)
        data['emg2_emg3_ratio'] = data['emg2_clean'] / (data['emg3_clean'] + 1e-6)
        
        # Statistical features
        data['emg_mean'] = data[['emg1_clean', 'emg2_clean', 'emg3_clean']].mean(axis=1)
        data['emg_std'] = data[['emg1_clean', 'emg2_clean', 'emg3_clean']].std(axis=1)
        data['emg_max'] = data[['emg1_clean', 'emg2_clean', 'emg3_clean']].max(axis=1)
        data['emg_min'] = data[['emg1_clean', 'emg2_clean', 'emg3_clean']].min(axis=1)
        data['emg_range'] = data['emg_max'] - data['emg_min']
        
        # Differences
        data['emg1_emg2_diff'] = np.abs(data['emg1_clean'] - data['emg2_clean'])
        data['emg1_emg3_diff'] = np.abs(data['emg1_clean'] - data['emg3_clean'])
        data['emg2_emg3_diff'] = np.abs(data['emg2_clean'] - data['emg3_clean'])
        
        # Products (interaction terms)
        data['emg1_emg2_product'] = data['emg1_clean'] * data['emg2_clean']
        data['emg1_emg3_product'] = data['emg1_clean'] * data['emg3_clean']
        data['emg2_emg3_product'] = data['emg2_clean'] * data['emg3_clean']
        
        # Power features
        data['emg1_squared'] = data['emg1_clean'] ** 2
        data['emg2_squared'] = data['emg2_clean'] ** 2
        data['emg3_squared'] = data['emg3_clean'] ** 2
        data['total_power'] = data['emg1_squared'] + data['emg2_squared'] + data['emg3_squared']
        
        # Check if subject and fatigue columns exist
        if 'subject_id' in data.columns:
            data['subject_encoded'] = data['subject_id'] / 4.0  # Normalize to 0-1
        else:
            data['subject_encoded'] = 0.5  # Default value

        if 'fatigue_factor' not in data.columns:
            data['fatigue_factor'] = 1.0  # Default value

        # All feature columns
        feature_columns = [
            'emg1_clean', 'emg2_clean', 'emg3_clean',
            'emg1_emg2_ratio', 'emg1_emg3_ratio', 'emg2_emg3_ratio',
            'emg_mean', 'emg_std', 'emg_max', 'emg_min', 'emg_range',
            'emg1_emg2_diff', 'emg1_emg3_diff', 'emg2_emg3_diff',
            'emg1_emg2_product', 'emg1_emg3_product', 'emg2_emg3_product',
            'emg1_squared', 'emg2_squared', 'emg3_squared', 'total_power',
            'subject_encoded', 'fatigue_factor'
        ]
        
        # Prepare features and labels
        X = data[feature_columns].values
        y = data['gesture'].values
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.gesture_names = list(self.label_encoder.classes_)
        
        print(f"✅ Feature engineering completed")
        print(f"📊 Total features: {len(feature_columns)}")
        print(f"📊 Samples: {X.shape[0]:,}")
        print(f"🎯 Classes: {len(self.gesture_names)}")
        
        return X, y_encoded, feature_columns
    
    def create_deep_learning_model(self, input_dim, num_classes):
        """Create deep neural network model"""
        print(f"\n🧠 Creating deep learning model...")
        print(f"📊 Input features: {input_dim}")
        print(f"🎯 Output classes: {num_classes}")
        
        model = keras.Sequential([
            # Input layer with dropout
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Deep learning model created")
        print(f"🏗️ Architecture: 512→256→128→64→32→{num_classes}")
        print(f"⚡ Optimizer: Adam (lr=0.001)")
        print(f"📊 Loss: Sparse Categorical Crossentropy")
        
        return model
    
    def train_model(self, X, y, feature_columns):
        """Train the deep learning model"""
        print(f"\n🚀 Training deep learning model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"📊 Data split:")
        print(f"   Training: {X_train.shape[0]:,} samples")
        print(f"   Validation: {X_val.shape[0]:,} samples")
        print(f"   Test: {X_test.shape[0]:,} samples")
        
        # Scale features
        print(f"⚖️ Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create model
        self.model = self.create_deep_learning_model(X_train.shape[1], len(self.gesture_names))
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        print(f"🏃 Starting training...")
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=128,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print(f"\n📊 Evaluating on test set...")
        test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        # Predictions for detailed analysis
        y_pred = self.model.predict(X_test_scaled, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print(f"\n✅ Training completed!")
        print(f"📊 Test Accuracy: {test_accuracy:.4f}")
        print(f"📊 Test Loss: {test_loss:.4f}")
        
        # Detailed classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.gesture_names))
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_classes': y_pred_classes,
            'feature_columns': feature_columns
        }
    
    def save_model(self, results, filename='deep_learning_emg_model'):
        """Save the trained model"""
        print(f"\n💾 Saving deep learning model...")
        
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
            'model_type': 'Deep-Learning-Neural-Network',
            'architecture': '512-256-128-64-32',
            'total_params': self.model.count_params(),
            'input_features': len(results['feature_columns'])
        }
        
        preprocessing_filename = f"{filename}_preprocessing.pkl"
        joblib.dump(preprocessing_data, preprocessing_filename)
        
        print(f"✅ Model saved:")
        print(f"   Keras model: {keras_filename}")
        print(f"   Preprocessing: {preprocessing_filename}")
        print(f"📊 Model info:")
        print(f"   Architecture: Deep Neural Network")
        print(f"   Parameters: {self.model.count_params():,}")
        print(f"   Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"   Input Features: {len(results['feature_columns'])}")
        
        return keras_filename, preprocessing_filename
    
    def create_training_visualization(self, results):
        """Create comprehensive training visualization"""
        print(f"\n📊 Creating training visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Deep Learning Model Training Results', fontsize=16, fontweight='bold')
        
        # 1. Training history
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Training loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Confusion matrix
        cm = confusion_matrix(results['y_test'], results['y_pred_classes'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=[g.split('-')[1] for g in self.gesture_names],
                   yticklabels=[g.split('-')[1] for g in self.gesture_names])
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title('Confusion Matrix', fontweight='bold')
        
        # 4. Per-class accuracy
        per_class_accuracy = []
        for i in range(len(self.gesture_names)):
            class_mask = results['y_test'] == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(results['y_test'][class_mask], 
                                         results['y_pred_classes'][class_mask])
                per_class_accuracy.append(class_acc)
            else:
                per_class_accuracy.append(0)
        
        gesture_names_short = [g.split('-')[1] for g in self.gesture_names]
        bars = ax4.bar(gesture_names_short, per_class_accuracy, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(per_class_accuracy))), alpha=0.8)
        ax4.set_xlabel('Gesture')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Per-Class Accuracy', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add accuracy labels on bars
        for bar, acc in zip(bars, per_class_accuracy):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('deep_learning_training_results.png', dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved: deep_learning_training_results.png")
        
        return fig

def main():
    """Main deep learning training function"""
    trainer = DeepLearningEMGTrainer()
    
    # Load dataset
    data = trainer.load_dataset()
    if data is None:
        return
    
    # Prepare data
    X, y, feature_columns = trainer.prepare_data(data)
    
    # Train model
    results = trainer.train_model(X, y, feature_columns)
    
    # Save model
    keras_file, preprocessing_file = trainer.save_model(results)
    
    # Create visualization
    try:
        trainer.create_training_visualization(results)
    except ImportError:
        print("⚠️  Matplotlib/Seaborn not available - skipping visualization")
    
    print(f"\n🎉 SUCCESS!")
    print(f"✅ Deep learning model trained successfully")
    print(f"📊 Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"🧠 Model: Deep Neural Network")
    print(f"📊 Parameters: {trainer.model.count_params():,}")
    print(f"💾 Saved as: {keras_file}")
    
    if results['test_accuracy'] > 0.9:
        print(f"🎯 EXCELLENT! >90% accuracy achieved!")
    elif results['test_accuracy'] > 0.8:
        print(f"🎯 VERY GOOD! >80% accuracy achieved!")
    else:
        print(f"🎯 GOOD! Model trained successfully!")
    
    print(f"\n🚀 Ready for real-time prediction!")
    print(f"⚡ Use this model for high-accuracy EMG classification")

if __name__ == "__main__":
    main()
