#!/usr/bin/env python3
"""
Pattern Analysis and ML Training for Combined EXG Data
Analyze raw data patterns for open (0) vs close (1) gestures
Train models based on discovered patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
import warnings
warnings.filterwarnings('ignore')

class EXGPatternAnalyzer:
    def __init__(self):
        self.data = None
        self.patterns = {}
        self.pattern_features = None
        self.models = {}
        
        print("🔍 EXG Pattern Analysis & ML Training")
        print("📊 Analyzing raw data patterns for open (0) vs close (1)")
        print("=" * 70)
    
    def load_data(self, filename='combined_exg_data.csv'):
        """Load the combined EXG dataset"""
        print(f"📂 Loading dataset: {filename}")
        
        if not os.path.exists(filename):
            print(f"❌ Dataset not found: {filename}")
            return False
        
        self.data = pd.read_csv(filename)
        print(f"✅ Loaded dataset: {self.data.shape}")
        print(f"📊 Total samples: {len(self.data):,}")
        
        # Show label distribution
        label_counts = self.data['label'].value_counts().sort_index()
        print(f"🏷️ Label distribution:")
        for label, count in label_counts.items():
            percentage = count / len(self.data) * 100
            print(f"   Label {label}: {count:,} samples ({percentage:.1f}%)")
        
        return True
    
    def analyze_raw_patterns(self):
        """Analyze patterns in raw EXG data for each label"""
        print(f"\n🔍 Analyzing Raw Data Patterns...")
        
        # Separate data by labels
        open_data = self.data[self.data['label'] == 0]  # Open gesture
        close_data = self.data[self.data['label'] == 1]  # Close gesture
        
        print(f"📊 Open (0) samples: {len(open_data):,}")
        print(f"📊 Close (1) samples: {len(close_data):,}")
        
        # Analyze raw EMG patterns
        patterns = {}
        
        for label, data_subset, name in [(0, open_data, 'Open'), (1, close_data, 'Close')]:
            print(f"\n🔍 Analyzing {name} (Label {label}) patterns:")
            
            # Raw EMG statistics
            raw_emg1_stats = {
                'mean': data_subset['raw_emg1'].mean(),
                'std': data_subset['raw_emg1'].std(),
                'min': data_subset['raw_emg1'].min(),
                'max': data_subset['raw_emg1'].max(),
                'median': data_subset['raw_emg1'].median(),
                'q25': data_subset['raw_emg1'].quantile(0.25),
                'q75': data_subset['raw_emg1'].quantile(0.75)
            }
            
            raw_emg2_stats = {
                'mean': data_subset['raw_emg2'].mean(),
                'std': data_subset['raw_emg2'].std(),
                'min': data_subset['raw_emg2'].min(),
                'max': data_subset['raw_emg2'].max(),
                'median': data_subset['raw_emg2'].median(),
                'q25': data_subset['raw_emg2'].quantile(0.25),
                'q75': data_subset['raw_emg2'].quantile(0.75)
            }
            
            # Normalized EMG patterns
            norm_emg1_stats = {
                'mean': data_subset['normalized_emg1'].mean(),
                'std': data_subset['normalized_emg1'].std(),
                'median': data_subset['normalized_emg1'].median()
            }
            
            norm_emg2_stats = {
                'mean': data_subset['normalized_emg2'].mean(),
                'std': data_subset['normalized_emg2'].std(),
                'median': data_subset['normalized_emg2'].median()
            }
            
            # Derivative patterns
            deriv_emg1_stats = {
                'mean': data_subset['derivative_emg1'].mean(),
                'std': data_subset['derivative_emg1'].std(),
                'positive_ratio': (data_subset['derivative_emg1'] > 0).mean(),
                'negative_ratio': (data_subset['derivative_emg1'] < 0).mean()
            }
            
            deriv_emg2_stats = {
                'mean': data_subset['derivative_emg2'].mean(),
                'std': data_subset['derivative_emg2'].std(),
                'positive_ratio': (data_subset['derivative_emg2'] > 0).mean(),
                'negative_ratio': (data_subset['derivative_emg2'] < 0).mean()
            }
            
            patterns[label] = {
                'raw_emg1': raw_emg1_stats,
                'raw_emg2': raw_emg2_stats,
                'normalized_emg1': norm_emg1_stats,
                'normalized_emg2': norm_emg2_stats,
                'derivative_emg1': deriv_emg1_stats,
                'derivative_emg2': deriv_emg2_stats
            }
            
            # Display key patterns
            print(f"   📊 Raw EMG1 - Mean: {raw_emg1_stats['mean']:.0f}, Range: {raw_emg1_stats['min']:.0f}-{raw_emg1_stats['max']:.0f}")
            print(f"   📊 Raw EMG2 - Mean: {raw_emg2_stats['mean']:.0f}, Range: {raw_emg2_stats['min']:.0f}-{raw_emg2_stats['max']:.0f}")
            print(f"   📊 Norm EMG1 - Mean: {norm_emg1_stats['mean']:.3f}, Std: {norm_emg1_stats['std']:.3f}")
            print(f"   📊 Norm EMG2 - Mean: {norm_emg2_stats['mean']:.3f}, Std: {norm_emg2_stats['std']:.3f}")
            print(f"   📊 Deriv EMG1 - Mean: {deriv_emg1_stats['mean']:.1f}, Pos%: {deriv_emg1_stats['positive_ratio']:.1%}")
            print(f"   📊 Deriv EMG2 - Mean: {deriv_emg2_stats['mean']:.1f}, Pos%: {deriv_emg2_stats['positive_ratio']:.1%}")
        
        self.patterns = patterns
        return patterns
    
    def identify_discriminative_patterns(self):
        """Identify the most discriminative patterns between open and close"""
        print(f"\n🎯 Identifying Discriminative Patterns...")
        
        open_patterns = self.patterns[0]
        close_patterns = self.patterns[1]
        
        discriminative_features = []
        
        print(f"🔍 Key Differences Between Open (0) and Close (1):")
        print("-" * 60)
        
        # Raw EMG differences
        emg1_mean_diff = abs(open_patterns['raw_emg1']['mean'] - close_patterns['raw_emg1']['mean'])
        emg2_mean_diff = abs(open_patterns['raw_emg2']['mean'] - close_patterns['raw_emg2']['mean'])
        
        print(f"📊 Raw EMG1 Mean Difference: {emg1_mean_diff:.0f}")
        print(f"   Open: {open_patterns['raw_emg1']['mean']:.0f}, Close: {close_patterns['raw_emg1']['mean']:.0f}")
        
        print(f"📊 Raw EMG2 Mean Difference: {emg2_mean_diff:.0f}")
        print(f"   Open: {open_patterns['raw_emg2']['mean']:.0f}, Close: {close_patterns['raw_emg2']['mean']:.0f}")
        
        # Normalized EMG differences
        norm1_diff = abs(open_patterns['normalized_emg1']['mean'] - close_patterns['normalized_emg1']['mean'])
        norm2_diff = abs(open_patterns['normalized_emg2']['mean'] - close_patterns['normalized_emg2']['mean'])
        
        print(f"📊 Normalized EMG1 Difference: {norm1_diff:.3f}")
        print(f"   Open: {open_patterns['normalized_emg1']['mean']:.3f}, Close: {close_patterns['normalized_emg1']['mean']:.3f}")
        
        print(f"📊 Normalized EMG2 Difference: {norm2_diff:.3f}")
        print(f"   Open: {open_patterns['normalized_emg2']['mean']:.3f}, Close: {close_patterns['normalized_emg2']['mean']:.3f}")
        
        # Derivative patterns
        deriv1_pos_diff = abs(open_patterns['derivative_emg1']['positive_ratio'] - close_patterns['derivative_emg1']['positive_ratio'])
        deriv2_pos_diff = abs(open_patterns['derivative_emg2']['positive_ratio'] - close_patterns['derivative_emg2']['positive_ratio'])
        
        print(f"📊 Derivative EMG1 Positive Ratio Difference: {deriv1_pos_diff:.3f}")
        print(f"   Open: {open_patterns['derivative_emg1']['positive_ratio']:.3f}, Close: {close_patterns['derivative_emg1']['positive_ratio']:.3f}")
        
        print(f"📊 Derivative EMG2 Positive Ratio Difference: {deriv2_pos_diff:.3f}")
        print(f"   Open: {open_patterns['derivative_emg2']['positive_ratio']:.3f}, Close: {close_patterns['derivative_emg2']['positive_ratio']:.3f}")
        
        # Identify most discriminative features
        feature_importance = {
            'raw_emg1_mean': emg1_mean_diff,
            'raw_emg2_mean': emg2_mean_diff,
            'normalized_emg1_mean': norm1_diff * 1000,  # Scale for comparison
            'normalized_emg2_mean': norm2_diff * 1000,
            'derivative_emg1_pos_ratio': deriv1_pos_diff * 1000,
            'derivative_emg2_pos_ratio': deriv2_pos_diff * 1000
        }
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 Most Discriminative Features (ranked by difference):")
        for i, (feature, importance) in enumerate(sorted_features, 1):
            print(f"   {i}. {feature}: {importance:.1f}")
        
        return sorted_features
    
    def create_pattern_based_features(self):
        """Create features based on discovered patterns"""
        print(f"\n🔧 Creating Pattern-Based Features...")
        
        # Get pattern thresholds from analysis
        open_patterns = self.patterns[0]
        close_patterns = self.patterns[1]
        
        # Create enhanced features based on patterns
        pattern_features = []
        
        for _, row in self.data.iterrows():
            features = []
            
            # Original features
            features.extend([
                row['raw_emg1'], row['raw_emg2'],
                row['normalized_emg1'], row['normalized_emg2'],
                row['derivative_emg1'], row['derivative_emg2']
            ])
            
            # Pattern-based features
            
            # 1. Distance from open/close centroids
            open_emg1_dist = abs(row['raw_emg1'] - open_patterns['raw_emg1']['mean'])
            close_emg1_dist = abs(row['raw_emg1'] - close_patterns['raw_emg1']['mean'])
            open_emg2_dist = abs(row['raw_emg2'] - open_patterns['raw_emg2']['mean'])
            close_emg2_dist = abs(row['raw_emg2'] - close_patterns['raw_emg2']['mean'])
            
            features.extend([open_emg1_dist, close_emg1_dist, open_emg2_dist, close_emg2_dist])
            
            # 2. Ratio features
            emg_ratio = row['raw_emg1'] / (row['raw_emg2'] + 1e-6)
            norm_ratio = row['normalized_emg1'] / (row['normalized_emg2'] + 1e-6)
            
            features.extend([emg_ratio, norm_ratio])
            
            # 3. Pattern similarity scores
            open_norm1_sim = 1 / (1 + abs(row['normalized_emg1'] - open_patterns['normalized_emg1']['mean']))
            close_norm1_sim = 1 / (1 + abs(row['normalized_emg1'] - close_patterns['normalized_emg1']['mean']))
            open_norm2_sim = 1 / (1 + abs(row['normalized_emg2'] - open_patterns['normalized_emg2']['mean']))
            close_norm2_sim = 1 / (1 + abs(row['normalized_emg2'] - close_patterns['normalized_emg2']['mean']))
            
            features.extend([open_norm1_sim, close_norm1_sim, open_norm2_sim, close_norm2_sim])
            
            # 4. Derivative pattern features
            deriv_sum = row['derivative_emg1'] + row['derivative_emg2']
            deriv_diff = abs(row['derivative_emg1'] - row['derivative_emg2'])
            deriv_sign_match = 1 if (row['derivative_emg1'] > 0) == (row['derivative_emg2'] > 0) else 0
            
            features.extend([deriv_sum, deriv_diff, deriv_sign_match])
            
            # 5. Amplitude features
            total_amplitude = row['raw_emg1'] + row['raw_emg2']
            amplitude_balance = abs(row['raw_emg1'] - row['raw_emg2']) / (total_amplitude + 1e-6)
            
            features.extend([total_amplitude, amplitude_balance])
            
            pattern_features.append(features)
        
        self.pattern_features = np.array(pattern_features)
        
        print(f"✅ Created pattern-based features: {self.pattern_features.shape}")
        print(f"📊 Features per sample: {self.pattern_features.shape[1]}")
        
        feature_names = [
            'raw_emg1', 'raw_emg2', 'normalized_emg1', 'normalized_emg2', 'derivative_emg1', 'derivative_emg2',
            'open_emg1_dist', 'close_emg1_dist', 'open_emg2_dist', 'close_emg2_dist',
            'emg_ratio', 'norm_ratio',
            'open_norm1_sim', 'close_norm1_sim', 'open_norm2_sim', 'close_norm2_sim',
            'deriv_sum', 'deriv_diff', 'deriv_sign_match',
            'total_amplitude', 'amplitude_balance'
        ]
        
        print(f"📋 Feature names: {feature_names}")
        return self.pattern_features, feature_names
    
    def train_pattern_based_models(self):
        """Train ML models using pattern-based features"""
        print(f"\n🚀 Training Pattern-Based ML Models...")
        
        if self.pattern_features is None:
            print("❌ Pattern features not created yet!")
            return None
        
        # Prepare data
        X = self.pattern_features
        y = self.data['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training samples: {len(X_train):,}")
        print(f"📊 Test samples: {len(X_test):,}")
        print(f"📊 Features: {X.shape[1]}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models optimized for pattern detection
        models = {
            'Pattern Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'Pattern Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, max_depth=8, learning_rate=0.1,
                random_state=42
            ),
            'Pattern Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50), activation='relu',
                random_state=42, max_iter=500
            ),
            'Pattern SVM': SVC(
                kernel='rbf', C=1.0, gamma='scale',
                random_state=42, probability=True
            ),
            'Pattern KNN': KNeighborsClassifier(
                n_neighbors=7, weights='distance'
            ),
            'Pattern Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, C=1.0
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Test
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                
                results[name] = {
                    'model': model,
                    'scaler': scaler,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                
                print(f"   ✅ {name}: {accuracy:.4f} (CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
        
        self.models = results
        return results
    
    def display_pattern_results(self):
        """Display comprehensive pattern-based results"""
        print(f"\n🏆 PATTERN-BASED MODEL RESULTS")
        print("=" * 70)
        
        # Sort by accuracy
        sorted_results = sorted(self.models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print(f"{'Rank':<4} {'Model':<30} {'Test Acc':<10} {'CV Mean':<10} {'CV Std':<10}")
        print("-" * 70)
        
        for i, (name, result) in enumerate(sorted_results, 1):
            print(f"{i:<4} {name:<30} {result['accuracy']:<10.4f} {result['cv_mean']:<10.4f} {result['cv_std']:<10.4f}")
        
        # Best model details
        best_name, best_result = sorted_results[0]
        print(f"\n🥇 BEST PATTERN-BASED MODEL: {best_name}")
        print(f"   🎯 Test Accuracy: {best_result['accuracy']:.4f}")
        print(f"   📊 Cross-Validation: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
        
        # Classification report for best model
        print(f"\n📊 Detailed Classification Report for {best_name}:")
        print(classification_report(best_result['y_test'], best_result['y_pred'], 
                                  target_names=['Open (0)', 'Close (1)']))
        
        return sorted_results
    
    def save_pattern_model(self, results):
        """Save the best pattern-based model"""
        best_name, best_result = max(results.items(), key=lambda x: x[1]['accuracy'])
        
        print(f"\n💾 Saving best pattern-based model...")
        
        model_data = {
            'model': best_result['model'],
            'scaler': best_result['scaler'],
            'patterns': self.patterns,
            'model_name': best_name,
            'test_accuracy': best_result['accuracy'],
            'cv_mean': best_result['cv_mean'],
            'cv_std': best_result['cv_std'],
            'training_data': 'combined_exg_data.csv',
            'model_type': 'Pattern-Based-EXG-Classification',
            'feature_count': self.pattern_features.shape[1]
        }
        
        filename = 'pattern_based_exg_model.pkl'
        joblib.dump(model_data, filename)
        
        print(f"✅ Pattern-based model saved: {filename}")
        print(f"🎯 Model: {best_name}")
        print(f"📊 Accuracy: {best_result['accuracy']:.4f}")
        print(f"🔧 Features: {self.pattern_features.shape[1]}")
        
        return filename

def main():
    """Main function"""
    analyzer = EXGPatternAnalyzer()
    
    # Load data
    if not analyzer.load_data():
        return
    
    # Analyze patterns
    patterns = analyzer.analyze_raw_patterns()
    
    # Identify discriminative patterns
    discriminative_features = analyzer.identify_discriminative_patterns()
    
    # Create pattern-based features
    pattern_features, feature_names = analyzer.create_pattern_based_features()
    
    # Train pattern-based models
    results = analyzer.train_pattern_based_models()
    
    if results:
        # Display results
        sorted_results = analyzer.display_pattern_results()
        
        # Save best model
        model_file = analyzer.save_pattern_model(results)
        
        print(f"\n🎉 Pattern Analysis & Training Complete!")
        print(f"🔍 Discovered key patterns between Open (0) and Close (1)")
        print(f"🏆 Best model: {sorted_results[0][0]}")
        print(f"🎯 Best accuracy: {sorted_results[0][1]['accuracy']:.4f}")
        print(f"💾 Saved as: {model_file}")

if __name__ == "__main__":
    main()
