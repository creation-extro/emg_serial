#!/usr/bin/env python3
"""
Fuzzy Logic EXG Classifier for Combined EXG Data
Implements fuzzy logic rules for Open (0) vs Close (1) gesture classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class FuzzyLogicEXGClassifier:
    def __init__(self):
        self.fuzzy_rules = []
        self.membership_functions = {}
        self.data_stats = {}
        
        print("🧠 Fuzzy Logic EXG Classifier")
        print("📊 Implementing fuzzy rules for Open/Close detection")
        print("=" * 60)
    
    def load_and_analyze_data(self, filename='combined_exg_data.csv'):
        """Load and analyze EXG data for fuzzy logic parameters"""
        print(f"📂 Loading EXG data: {filename}")
        
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            return None
        
        data = pd.read_csv(filename)
        print(f"✅ Loaded: {data.shape}")
        
        # Analyze data statistics for fuzzy membership functions
        self.data_stats = {
            'raw_emg1': {
                'min': data['raw_emg1'].min(),
                'max': data['raw_emg1'].max(),
                'mean': data['raw_emg1'].mean(),
                'std': data['raw_emg1'].std(),
                'open_mean': data[data['label'] == 0]['raw_emg1'].mean(),
                'close_mean': data[data['label'] == 1]['raw_emg1'].mean()
            },
            'raw_emg2': {
                'min': data['raw_emg2'].min(),
                'max': data['raw_emg2'].max(),
                'mean': data['raw_emg2'].mean(),
                'std': data['raw_emg2'].std(),
                'open_mean': data[data['label'] == 0]['raw_emg2'].mean(),
                'close_mean': data[data['label'] == 1]['raw_emg2'].mean()
            },
            'normalized_emg1': {
                'min': data['normalized_emg1'].min(),
                'max': data['normalized_emg1'].max(),
                'mean': data['normalized_emg1'].mean(),
                'open_mean': data[data['label'] == 0]['normalized_emg1'].mean(),
                'close_mean': data[data['label'] == 1]['normalized_emg1'].mean()
            },
            'normalized_emg2': {
                'min': data['normalized_emg2'].min(),
                'max': data['normalized_emg2'].max(),
                'mean': data['normalized_emg2'].mean(),
                'open_mean': data[data['label'] == 0]['normalized_emg2'].mean(),
                'close_mean': data[data['label'] == 1]['normalized_emg2'].mean()
            },
            'derivative_emg1': {
                'min': data['derivative_emg1'].min(),
                'max': data['derivative_emg1'].max(),
                'mean': data['derivative_emg1'].mean(),
                'open_mean': data[data['label'] == 0]['derivative_emg1'].mean(),
                'close_mean': data[data['label'] == 1]['derivative_emg1'].mean()
            },
            'derivative_emg2': {
                'min': data['derivative_emg2'].min(),
                'max': data['derivative_emg2'].max(),
                'mean': data['derivative_emg2'].mean(),
                'open_mean': data[data['label'] == 0]['derivative_emg2'].mean(),
                'close_mean': data[data['label'] == 1]['derivative_emg2'].mean()
            }
        }
        
        print(f"📊 Data Statistics Summary:")
        print(f"   Raw EMG1 - Open: {self.data_stats['raw_emg1']['open_mean']:.0f}, Close: {self.data_stats['raw_emg1']['close_mean']:.0f}")
        print(f"   Raw EMG2 - Open: {self.data_stats['raw_emg2']['open_mean']:.0f}, Close: {self.data_stats['raw_emg2']['close_mean']:.0f}")
        print(f"   Norm EMG1 - Open: {self.data_stats['normalized_emg1']['open_mean']:.3f}, Close: {self.data_stats['normalized_emg1']['close_mean']:.3f}")
        print(f"   Norm EMG2 - Open: {self.data_stats['normalized_emg2']['open_mean']:.3f}, Close: {self.data_stats['normalized_emg2']['close_mean']:.3f}")
        
        return data
    
    def create_membership_functions(self):
        """Create fuzzy membership functions based on data analysis"""
        print(f"\n🔧 Creating Fuzzy Membership Functions...")
        
        # Raw EMG1 membership functions
        self.membership_functions['raw_emg1'] = {
            'low': {'center': self.data_stats['raw_emg1']['min'], 'width': 5000},
            'medium': {'center': self.data_stats['raw_emg1']['mean'], 'width': 3000},
            'high': {'center': self.data_stats['raw_emg1']['max'], 'width': 5000},
            'open_typical': {'center': self.data_stats['raw_emg1']['open_mean'], 'width': 2000},
            'close_typical': {'center': self.data_stats['raw_emg1']['close_mean'], 'width': 2000}
        }
        
        # Raw EMG2 membership functions (most discriminative)
        self.membership_functions['raw_emg2'] = {
            'low': {'center': self.data_stats['raw_emg2']['min'], 'width': 5000},
            'medium': {'center': self.data_stats['raw_emg2']['mean'], 'width': 3000},
            'high': {'center': self.data_stats['raw_emg2']['max'], 'width': 5000},
            'open_typical': {'center': self.data_stats['raw_emg2']['open_mean'], 'width': 2000},
            'close_typical': {'center': self.data_stats['raw_emg2']['close_mean'], 'width': 2000}
        }
        
        # Normalized EMG membership functions
        self.membership_functions['normalized_emg1'] = {
            'low': {'center': 0.2, 'width': 0.15},
            'medium': {'center': 0.5, 'width': 0.2},
            'high': {'center': 0.8, 'width': 0.15},
            'open_typical': {'center': self.data_stats['normalized_emg1']['open_mean'], 'width': 0.1},
            'close_typical': {'center': self.data_stats['normalized_emg1']['close_mean'], 'width': 0.1}
        }
        
        self.membership_functions['normalized_emg2'] = {
            'low': {'center': 0.2, 'width': 0.15},
            'medium': {'center': 0.5, 'width': 0.2},
            'high': {'center': 0.8, 'width': 0.15},
            'open_typical': {'center': self.data_stats['normalized_emg2']['open_mean'], 'width': 0.1},
            'close_typical': {'center': self.data_stats['normalized_emg2']['close_mean'], 'width': 0.1}
        }
        
        # Derivative membership functions
        self.membership_functions['derivative_emg1'] = {
            'negative': {'center': -500, 'width': 300},
            'zero': {'center': 0, 'width': 100},
            'positive': {'center': 500, 'width': 300},
            'open_typical': {'center': self.data_stats['derivative_emg1']['open_mean'], 'width': 200},
            'close_typical': {'center': self.data_stats['derivative_emg1']['close_mean'], 'width': 200}
        }
        
        self.membership_functions['derivative_emg2'] = {
            'negative': {'center': -500, 'width': 300},
            'zero': {'center': 0, 'width': 100},
            'positive': {'center': 500, 'width': 300},
            'open_typical': {'center': self.data_stats['derivative_emg2']['open_mean'], 'width': 200},
            'close_typical': {'center': self.data_stats['derivative_emg2']['close_mean'], 'width': 200}
        }
        
        print(f"✅ Created membership functions for 6 input variables")
        print(f"📊 Total membership functions: {sum(len(mf) for mf in self.membership_functions.values())}")
    
    def gaussian_membership(self, x, center, width):
        """Calculate Gaussian membership function"""
        return np.exp(-0.5 * ((x - center) / width) ** 2)
    
    def triangular_membership(self, x, left, center, right):
        """Calculate triangular membership function"""
        if x <= left or x >= right:
            return 0.0
        elif x <= center:
            return (x - left) / (center - left)
        else:
            return (right - x) / (right - center)
    
    def calculate_membership_degrees(self, raw_emg1, raw_emg2, norm_emg1, norm_emg2, deriv_emg1, deriv_emg2):
        """Calculate membership degrees for all input variables"""
        memberships = {}
        
        # Raw EMG1 memberships
        memberships['raw_emg1'] = {}
        for label, params in self.membership_functions['raw_emg1'].items():
            memberships['raw_emg1'][label] = self.gaussian_membership(raw_emg1, params['center'], params['width'])
        
        # Raw EMG2 memberships
        memberships['raw_emg2'] = {}
        for label, params in self.membership_functions['raw_emg2'].items():
            memberships['raw_emg2'][label] = self.gaussian_membership(raw_emg2, params['center'], params['width'])
        
        # Normalized EMG memberships
        memberships['normalized_emg1'] = {}
        for label, params in self.membership_functions['normalized_emg1'].items():
            memberships['normalized_emg1'][label] = self.gaussian_membership(norm_emg1, params['center'], params['width'])
        
        memberships['normalized_emg2'] = {}
        for label, params in self.membership_functions['normalized_emg2'].items():
            memberships['normalized_emg2'][label] = self.gaussian_membership(norm_emg2, params['center'], params['width'])
        
        # Derivative memberships
        memberships['derivative_emg1'] = {}
        for label, params in self.membership_functions['derivative_emg1'].items():
            memberships['derivative_emg1'][label] = self.gaussian_membership(deriv_emg1, params['center'], params['width'])
        
        memberships['derivative_emg2'] = {}
        for label, params in self.membership_functions['derivative_emg2'].items():
            memberships['derivative_emg2'][label] = self.gaussian_membership(deriv_emg2, params['center'], params['width'])
        
        return memberships
    
    def create_fuzzy_rules(self):
        """Create fuzzy logic rules for Open/Close classification"""
        print(f"\n📋 Creating Fuzzy Logic Rules...")
        
        # Rule 1: Strong Open indicators
        self.fuzzy_rules.append({
            'name': 'Strong Open Rule',
            'conditions': [
                ('raw_emg2', 'open_typical', 0.8),
                ('normalized_emg2', 'open_typical', 0.7)
            ],
            'conclusion': 'open',
            'weight': 1.0
        })
        
        # Rule 2: Strong Close indicators
        self.fuzzy_rules.append({
            'name': 'Strong Close Rule',
            'conditions': [
                ('raw_emg2', 'close_typical', 0.8),
                ('normalized_emg2', 'close_typical', 0.7)
            ],
            'conclusion': 'close',
            'weight': 1.0
        })
        
        # Rule 3: High EMG2 suggests Open
        self.fuzzy_rules.append({
            'name': 'High EMG2 Open Rule',
            'conditions': [
                ('raw_emg2', 'high', 0.6),
                ('normalized_emg2', 'high', 0.5)
            ],
            'conclusion': 'open',
            'weight': 0.8
        })
        
        # Rule 4: Low EMG2 suggests Close
        self.fuzzy_rules.append({
            'name': 'Low EMG2 Close Rule',
            'conditions': [
                ('raw_emg2', 'low', 0.6),
                ('normalized_emg2', 'low', 0.5)
            ],
            'conclusion': 'close',
            'weight': 0.8
        })
        
        # Rule 5: Combined EMG1 and EMG2 for Open
        self.fuzzy_rules.append({
            'name': 'Combined Open Rule',
            'conditions': [
                ('raw_emg1', 'open_typical', 0.6),
                ('raw_emg2', 'open_typical', 0.6),
                ('normalized_emg1', 'open_typical', 0.5)
            ],
            'conclusion': 'open',
            'weight': 0.9
        })
        
        # Rule 6: Combined EMG1 and EMG2 for Close
        self.fuzzy_rules.append({
            'name': 'Combined Close Rule',
            'conditions': [
                ('raw_emg1', 'close_typical', 0.6),
                ('raw_emg2', 'close_typical', 0.6),
                ('normalized_emg1', 'close_typical', 0.5)
            ],
            'conclusion': 'close',
            'weight': 0.9
        })
        
        # Rule 7: Derivative-based Open rule
        self.fuzzy_rules.append({
            'name': 'Derivative Open Rule',
            'conditions': [
                ('derivative_emg1', 'positive', 0.7),
                ('raw_emg2', 'open_typical', 0.6)
            ],
            'conclusion': 'open',
            'weight': 0.7
        })
        
        # Rule 8: Derivative-based Close rule
        self.fuzzy_rules.append({
            'name': 'Derivative Close Rule',
            'conditions': [
                ('derivative_emg1', 'negative', 0.7),
                ('raw_emg2', 'close_typical', 0.6)
            ],
            'conclusion': 'close',
            'weight': 0.7
        })
        
        # Rule 9: Medium activity default to most discriminative feature
        self.fuzzy_rules.append({
            'name': 'Medium Activity Rule',
            'conditions': [
                ('raw_emg1', 'medium', 0.8),
                ('raw_emg2', 'medium', 0.8)
            ],
            'conclusion': 'open' if self.data_stats['raw_emg2']['open_mean'] > self.data_stats['raw_emg2']['close_mean'] else 'close',
            'weight': 0.5
        })
        
        print(f"✅ Created {len(self.fuzzy_rules)} fuzzy rules")
        for i, rule in enumerate(self.fuzzy_rules, 1):
            print(f"   {i}. {rule['name']} → {rule['conclusion']} (weight: {rule['weight']})")
    
    def evaluate_fuzzy_rules(self, memberships):
        """Evaluate all fuzzy rules and return aggregated result"""
        open_strength = 0.0
        close_strength = 0.0
        
        for rule in self.fuzzy_rules:
            # Calculate rule activation strength (minimum of all conditions)
            rule_strength = 1.0
            
            for variable, membership_label, threshold in rule['conditions']:
                membership_value = memberships[variable][membership_label]
                if membership_value >= threshold:
                    rule_strength = min(rule_strength, membership_value)
                else:
                    rule_strength = 0.0  # Rule not activated
                    break
            
            # Apply rule weight
            weighted_strength = rule_strength * rule['weight']
            
            # Accumulate strength for conclusion
            if rule['conclusion'] == 'open':
                open_strength += weighted_strength
            else:
                close_strength += weighted_strength
        
        return open_strength, close_strength
    
    def fuzzy_classify(self, raw_emg1, raw_emg2, norm_emg1, norm_emg2, deriv_emg1, deriv_emg2):
        """Classify a single EXG sample using fuzzy logic"""
        
        # Calculate membership degrees
        memberships = self.calculate_membership_degrees(
            raw_emg1, raw_emg2, norm_emg1, norm_emg2, deriv_emg1, deriv_emg2
        )
        
        # Evaluate fuzzy rules
        open_strength, close_strength = self.evaluate_fuzzy_rules(memberships)
        
        # Make final decision
        total_strength = open_strength + close_strength
        
        if total_strength == 0:
            # Default decision based on most discriminative feature (raw_emg2)
            prediction = 0 if raw_emg2 > self.data_stats['raw_emg2']['mean'] else 1
            confidence = 0.5
        else:
            open_confidence = open_strength / total_strength
            close_confidence = close_strength / total_strength
            
            if open_confidence > close_confidence:
                prediction = 0  # Open
                confidence = open_confidence
            else:
                prediction = 1  # Close
                confidence = close_confidence
        
        return {
            'prediction': prediction,
            'gesture': 'Open' if prediction == 0 else 'Close',
            'confidence': confidence,
            'open_strength': open_strength,
            'close_strength': close_strength,
            'memberships': memberships
        }
    
    def test_fuzzy_classifier(self, data, test_size=0.2):
        """Test the fuzzy classifier on the dataset"""
        print(f"\n🧪 Testing Fuzzy Logic Classifier...")
        
        # Prepare data
        X = data[['raw_emg1', 'raw_emg2', 'normalized_emg1', 'normalized_emg2', 'derivative_emg1', 'derivative_emg2']].values
        y = data['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        print(f"📊 Test samples: {len(X_test):,}")
        
        # Make predictions
        predictions = []
        confidences = []
        
        for i, sample in enumerate(X_test):
            if i % 1000 == 0:
                print(f"   Processing sample {i+1}/{len(X_test)}...")
            
            result = self.fuzzy_classify(*sample)
            predictions.append(result['prediction'])
            confidences.append(result['confidence'])
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        avg_confidence = np.mean(confidences)
        
        print(f"\n🎯 Fuzzy Logic Classifier Results:")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   Average Confidence: {avg_confidence:.4f}")
        
        # Detailed classification report
        print(f"\n📊 Detailed Classification Report:")
        print(classification_report(y_test, predictions, target_names=['Open (0)', 'Close (1)']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        print(f"\n📊 Confusion Matrix:")
        print(f"   Predicted:  Open  Close")
        print(f"   Open:      {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"   Close:     {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        return accuracy, predictions, confidences
    
    def save_fuzzy_classifier(self, filename='fuzzy_logic_exg_classifier.pkl'):
        """Save the fuzzy classifier"""
        classifier_data = {
            'membership_functions': self.membership_functions,
            'fuzzy_rules': self.fuzzy_rules,
            'data_stats': self.data_stats,
            'classifier_type': 'Fuzzy Logic EXG Classifier'
        }
        
        joblib.dump(classifier_data, filename)
        print(f"\n💾 Fuzzy classifier saved: {filename}")
        return filename

def main():
    """Main function"""
    # Initialize fuzzy classifier
    fuzzy_classifier = FuzzyLogicEXGClassifier()
    
    # Load and analyze data
    data = fuzzy_classifier.load_and_analyze_data()
    if data is None:
        return
    
    # Create membership functions
    fuzzy_classifier.create_membership_functions()
    
    # Create fuzzy rules
    fuzzy_classifier.create_fuzzy_rules()
    
    # Test the classifier
    accuracy, predictions, confidences = fuzzy_classifier.test_fuzzy_classifier(data)
    
    # Save the classifier
    model_file = fuzzy_classifier.save_fuzzy_classifier()
    
    print(f"\n🎉 Fuzzy Logic EXG Classification Complete!")
    print(f"🎯 Final Accuracy: {accuracy:.4f}")
    print(f"💾 Model saved as: {model_file}")
    print(f"🧠 Ready for real-time fuzzy logic classification!")

if __name__ == "__main__":
    main()
