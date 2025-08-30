#!/usr/bin/env python3
"""
Advanced EXG Accuracy Booster
Multiple sophisticated methods to maximize classification accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedEXGAccuracyBooster:
    def __init__(self):
        self.data = None
        self.results = {}
        self.best_features = None
        self.best_scaler = None
        
        print("🚀 Advanced EXG Accuracy Booster")
        print("📊 Multiple sophisticated methods for maximum accuracy")
        print("=" * 70)
    
    def rg
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    









    (self, filename='combined_exg_data.csv', sample_size=30000):
        """Load and perform deep analysis of EXG data"""
        print(f"📂 Loading and analyzing: {filename}")

        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            return False

        data_full = pd.read_csv(filename)
        print(f"✅ Loaded full dataset: {data_full.shape}")

        # Create balanced sample for faster processing
        if len(data_full) > sample_size:
            print(f"🔄 Creating balanced sample of {sample_size:,} for faster processing...")
            samples_per_label = sample_size // 2

            label_0 = data_full[data_full['label'] == 0].sample(n=samples_per_label, random_state=42)
            label_1 = data_full[data_full['label'] == 1].sample(n=samples_per_label, random_state=42)

            self.data = pd.concat([label_0, label_1]).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"✅ Using sample: {self.data.shape}")
        else:
            self.data = data_full
        
        # Deep statistical analysis
        print(f"\n🔍 Deep Statistical Analysis:")
        
        # Label distribution
        label_dist = self.data['label'].value_counts()
        print(f"📊 Label distribution: {dict(label_dist)}")
        
        # Feature correlations
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        print(f"🔗 High correlation pairs (>0.8): {len(high_corr_pairs)}")
        for pair in high_corr_pairs[:5]:  # Show top 5
            print(f"   {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")
        
        # Feature importance using Random Forest
        X_basic = self.data[['raw_emg1', 'raw_emg2', 'normalized_emg1', 'normalized_emg2', 'derivative_emg1', 'derivative_emg2']].values
        y = self.data['label'].values
        
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_temp.fit(X_basic, y)
        
        feature_names = ['raw_emg1', 'raw_emg2', 'normalized_emg1', 'normalized_emg2', 'derivative_emg1', 'derivative_emg2']
        feature_importance = list(zip(feature_names, rf_temp.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"🏆 Feature Importance (Random Forest):")
        for name, importance in feature_importance:
            print(f"   {name}: {importance:.4f}")
        
        return True
    
    def create_advanced_features(self):
        """Create advanced engineered features"""
        print(f"\n🔧 Creating Advanced Engineered Features...")
        
        advanced_features = []
        
        for _, row in self.data.iterrows():
            features = []
            
            # Original features
            raw1, raw2 = row['raw_emg1'], row['raw_emg2']
            norm1, norm2 = row['normalized_emg1'], row['normalized_emg2']
            deriv1, deriv2 = row['derivative_emg1'], row['derivative_emg2']
            
            features.extend([raw1, raw2, norm1, norm2, deriv1, deriv2])
            
            # Mathematical transformations
            features.extend([
                np.log1p(raw1), np.log1p(raw2),  # Log transforms
                np.sqrt(abs(raw1)), np.sqrt(abs(raw2)),  # Square root
                raw1**2, raw2**2,  # Squared
                raw1**0.5, raw2**0.5,  # Power 0.5
            ])
            
            # Statistical features
            raw_mean = (raw1 + raw2) / 2
            raw_std = np.std([raw1, raw2])
            raw_range = abs(raw1 - raw2)
            raw_max = max(raw1, raw2)
            raw_min = min(raw1, raw2)
            
            features.extend([raw_mean, raw_std, raw_range, raw_max, raw_min])
            
            # Ratio and interaction features
            features.extend([
                raw1 / (raw2 + 1e-6),  # Raw ratio
                norm1 / (norm2 + 1e-6),  # Norm ratio
                raw1 * raw2,  # Raw product
                norm1 * norm2,  # Norm product
                raw1 + raw2,  # Raw sum
                norm1 + norm2,  # Norm sum
            ])
            
            # Derivative features
            features.extend([
                deriv1 + deriv2,  # Derivative sum
                abs(deriv1 - deriv2),  # Derivative difference
                deriv1 * deriv2,  # Derivative product
                1 if deriv1 > 0 else 0,  # Deriv1 positive
                1 if deriv2 > 0 else 0,  # Deriv2 positive
                1 if (deriv1 > 0) == (deriv2 > 0) else 0,  # Same sign
            ])
            
            # Frequency-domain inspired features
            features.extend([
                abs(raw1 - raw_mean),  # Distance from mean
                abs(raw2 - raw_mean),
                (raw1 - raw_mean)**2,  # Squared distance
                (raw2 - raw_mean)**2,
            ])
            
            # Normalized features
            total_raw = raw1 + raw2 + 1e-6
            features.extend([
                raw1 / total_raw,  # Raw1 proportion
                raw2 / total_raw,  # Raw2 proportion
            ])
            
            # Advanced statistical measures
            features.extend([
                (raw1 - raw2) / (raw1 + raw2 + 1e-6),  # Normalized difference
                2 * raw1 * raw2 / (raw1 + raw2 + 1e-6),  # Harmonic mean
                np.sqrt(raw1 * raw2),  # Geometric mean
            ])
            
            advanced_features.append(features)
        
        self.advanced_features = np.array(advanced_features)
        print(f"✅ Created {self.advanced_features.shape[1]} advanced features")
        
        # Feature names for reference
        self.feature_names = [
            'raw_emg1', 'raw_emg2', 'normalized_emg1', 'normalized_emg2', 'derivative_emg1', 'derivative_emg2',
            'log_raw1', 'log_raw2', 'sqrt_raw1', 'sqrt_raw2', 'sq_raw1', 'sq_raw2', 'pow05_raw1', 'pow05_raw2',
            'raw_mean', 'raw_std', 'raw_range', 'raw_max', 'raw_min',
            'raw_ratio', 'norm_ratio', 'raw_product', 'norm_product', 'raw_sum', 'norm_sum',
            'deriv_sum', 'deriv_diff', 'deriv_product', 'deriv1_pos', 'deriv2_pos', 'deriv_same_sign',
            'dist_from_mean1', 'dist_from_mean2', 'sq_dist1', 'sq_dist2',
            'raw1_prop', 'raw2_prop', 'norm_diff', 'harmonic_mean', 'geometric_mean'
        ]
        
        return self.advanced_features
    
    def test_different_scalers(self, X, y):
        """Test different scaling methods"""
        print(f"\n⚖️ Testing Different Scaling Methods...")
        
        scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'PowerTransformer': PowerTransformer(method='yeo-johnson'),
            'No Scaling': None
        }
        
        scaler_results = {}
        
        # Use a simple model for quick testing
        test_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        for scaler_name, scaler in scalers.items():
            if scaler is None:
                X_scaled = X
            else:
                X_scaled = scaler.fit_transform(X)
            
            # Quick cross-validation
            cv_scores = cross_val_score(test_model, X_scaled, y, cv=3, scoring='accuracy')
            scaler_results[scaler_name] = cv_scores.mean()
            
            print(f"   {scaler_name}: {cv_scores.mean():.4f}")
        
        # Find best scaler
        best_scaler_name = max(scaler_results, key=scaler_results.get)
        self.best_scaler = scalers[best_scaler_name]
        
        print(f"🏆 Best scaler: {best_scaler_name} ({scaler_results[best_scaler_name]:.4f})")
        return self.best_scaler
    
    def feature_selection(self, X, y):
        """Advanced feature selection"""
        print(f"\n🎯 Advanced Feature Selection...")
        
        # Apply best scaler
        if self.best_scaler is not None:
            X_scaled = self.best_scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Method 1: Statistical selection
        selector_stats = SelectKBest(score_func=f_classif, k=20)
        X_stats = selector_stats.fit_transform(X_scaled, y)
        stats_scores = selector_stats.scores_
        
        # Method 2: Recursive Feature Elimination
        rf_selector = RandomForestClassifier(n_estimators=50, random_state=42)
        rfe = RFE(estimator=rf_selector, n_features_to_select=20)
        X_rfe = rfe.fit_transform(X_scaled, y)
        
        # Method 3: Feature importance from Random Forest
        rf_importance = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_importance.fit(X_scaled, y)
        importance_scores = rf_importance.feature_importances_
        
        # Get top features from each method
        top_stats_idx = np.argsort(stats_scores)[-20:]
        top_importance_idx = np.argsort(importance_scores)[-20:]
        rfe_selected_idx = np.where(rfe.support_)[0]
        
        # Combine and get unique features
        all_selected = np.unique(np.concatenate([top_stats_idx, top_importance_idx, rfe_selected_idx]))
        
        print(f"📊 Feature selection results:")
        print(f"   Statistical: {len(top_stats_idx)} features")
        print(f"   RFE: {len(rfe_selected_idx)} features")
        print(f"   Importance: {len(top_importance_idx)} features")
        print(f"   Combined unique: {len(all_selected)} features")
        
        self.best_features = all_selected
        return X_scaled[:, all_selected]
    
    def train_advanced_models(self, X, y):
        """Train advanced ML models with hyperparameter tuning"""
        print(f"\n🚀 Training Advanced Models with Hyperparameter Tuning...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training: {X_train.shape}, Test: {X_test.shape}")
        
        # Advanced models with hyperparameter grids
        models_and_params = {
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100]
                }
            },
            'Advanced Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'Extra Trees': {
                'model': ExtraTreesClassifier(random_state=42),
                'params': {
                    'n_estimators': [200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'Advanced Neural Network': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50), (200, 100), (100, 50, 25)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            },
            'SVM Advanced': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                }
            }
        }
        
        results = {}
        
        for name, config in models_and_params.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=3, 
                    scoring='accuracy',
                    n_jobs=-1 if name != 'SVM Advanced' else 1  # SVM can be memory intensive
                )
                
                grid_search.fit(X_train, y_train)
                
                # Best model predictions
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
                
                results[name] = {
                    'model': best_model,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'best_params': grid_search.best_params_,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                
                print(f"   ✅ {name}: {accuracy:.4f} (CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")
                print(f"   🔧 Best params: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
        
        return results
    
    def create_ensemble_models(self, individual_results, X_train, X_test, y_train, y_test):
        """Create ensemble models from best individual models"""
        print(f"\n🤝 Creating Ensemble Models...")
        
        # Get top 3 models
        sorted_models = sorted(individual_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        top_3_models = sorted_models[:3]
        
        print(f"🏆 Top 3 models for ensemble:")
        for i, (name, result) in enumerate(top_3_models, 1):
            print(f"   {i}. {name}: {result['accuracy']:.4f}")
        
        # Create voting classifier
        voting_models = [(name, result['model']) for name, result in top_3_models]
        
        # Hard voting
        hard_voting = VotingClassifier(estimators=voting_models, voting='hard')
        hard_voting.fit(X_train, y_train)
        hard_pred = hard_voting.predict(X_test)
        hard_accuracy = accuracy_score(y_test, hard_pred)
        
        # Soft voting
        soft_voting = VotingClassifier(estimators=voting_models, voting='soft')
        soft_voting.fit(X_train, y_train)
        soft_pred = soft_voting.predict(X_test)
        soft_accuracy = accuracy_score(y_test, soft_pred)
        
        ensemble_results = {
            'Hard Voting Ensemble': {
                'model': hard_voting,
                'accuracy': hard_accuracy,
                'y_pred': hard_pred
            },
            'Soft Voting Ensemble': {
                'model': soft_voting,
                'accuracy': soft_accuracy,
                'y_pred': soft_pred
            }
        }
        
        print(f"🗳️ Ensemble Results:")
        print(f"   Hard Voting: {hard_accuracy:.4f}")
        print(f"   Soft Voting: {soft_accuracy:.4f}")
        
        return ensemble_results
    
    def display_final_results(self, individual_results, ensemble_results):
        """Display comprehensive final results"""
        print(f"\n🏆 FINAL COMPREHENSIVE RESULTS")
        print("=" * 80)
        
        # Combine all results
        all_results = {**individual_results, **ensemble_results}
        
        # Sort by accuracy
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print(f"{'Rank':<4} {'Model':<25} {'Test Acc':<10} {'CV Mean':<10} {'CV Std':<10}")
        print("-" * 80)
        
        for i, (name, result) in enumerate(sorted_results, 1):
            cv_mean = result.get('cv_mean', 0)
            cv_std = result.get('cv_std', 0)
            print(f"{i:<4} {name:<25} {result['accuracy']:<10.4f} {cv_mean:<10.4f} {cv_std:<10.4f}")
        
        # Best model analysis
        best_name, best_result = sorted_results[0]
        print(f"\n🥇 BEST MODEL: {best_name}")
        print(f"   🎯 Test Accuracy: {best_result['accuracy']:.4f}")
        
        if 'cv_mean' in best_result:
            print(f"   📊 Cross-Validation: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
        
        if 'best_params' in best_result:
            print(f"   🔧 Best Parameters: {best_result['best_params']}")
        
        # Classification report
        if 'y_test' in best_result and 'y_pred' in best_result:
            print(f"\n📊 Detailed Classification Report:")
            print(classification_report(best_result['y_test'], best_result['y_pred'], 
                                      target_names=['Open (0)', 'Close (1)']))
        
        return sorted_results
    
    def save_best_model(self, sorted_results):
        """Save the best performing model"""
        best_name, best_result = sorted_results[0]
        
        model_data = {
            'model': best_result['model'],
            'scaler': self.best_scaler,
            'selected_features': self.best_features,
            'feature_names': [self.feature_names[i] for i in self.best_features] if self.best_features is not None else self.feature_names,
            'model_name': best_name,
            'test_accuracy': best_result['accuracy'],
            'cv_mean': best_result.get('cv_mean', 0),
            'cv_std': best_result.get('cv_std', 0),
            'best_params': best_result.get('best_params', {}),
            'training_data': 'combined_exg_data.csv',
            'model_type': 'Advanced-EXG-Classification'
        }
        
        filename = 'best_advanced_exg_model.pkl'
        joblib.dump(model_data, filename)
        
        print(f"\n💾 Best model saved: {filename}")
        print(f"🎯 Model: {best_name}")
        print(f"📊 Accuracy: {best_result['accuracy']:.4f}")
        
        return filename

def main():
    """Main function"""
    booster = AdvancedEXGAccuracyBooster()
    
    # Load and analyze data
    if not booster.load_and_analyze_data():
        return
    
    # Create advanced features
    X_advanced = booster.create_advanced_features()
    y = booster.data['label'].values
    
    # Test different scalers
    best_scaler = booster.test_different_scalers(X_advanced, y)
    
    # Feature selection
    X_selected = booster.feature_selection(X_advanced, y)
    
    # Train advanced models
    individual_results = booster.train_advanced_models(X_selected, y)
    
    if individual_results:
        # Split data for ensemble
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create ensemble models
        ensemble_results = booster.create_ensemble_models(
            individual_results, X_train, X_test, y_train, y_test
        )
        
        # Display final results
        sorted_results = booster.display_final_results(individual_results, ensemble_results)
        
        # Save best model
        model_file = booster.save_best_model(sorted_results)
        
        print(f"\n🎉 ADVANCED ACCURACY BOOSTING COMPLETE!")
        print(f"🏆 Best accuracy achieved: {sorted_results[0][1]['accuracy']:.4f}")
        print(f"💾 Best model saved as: {model_file}")

if __name__ == "__main__":
    main()
