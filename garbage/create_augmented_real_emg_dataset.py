#!/usr/bin/env python3
"""
Create Augmented Dataset from Real EMG Data (No Relax)
Apply various augmentation techniques to increase dataset size and diversity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm

class EMGDataAugmenter:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        print("🔧 EMG Data Augmentation Pipeline")
        print("📊 Creating diverse training data from real EMG samples")
        print("=" * 60)
    
    def load_real_dataset(self, filename='data/emg_data_no_relax.csv'):
        """Load the real EMG dataset"""
        print(f"📂 Loading real EMG dataset: {filename}")
        
        if not os.path.exists(filename):
            print(f"❌ Dataset not found: {filename}")
            return None
        
        data = pd.read_csv(filename)
        print(f"✅ Loaded dataset: {data.shape}")
        print(f"📊 Total samples: {len(data):,}")
        print(f"📊 Gestures: {data['gesture'].nunique()}")
        
        # Show gesture distribution
        print(f"\n🏷️ Original Gesture Distribution:")
        gesture_counts = data['gesture'].value_counts().sort_index()
        for gesture, count in gesture_counts.items():
            print(f"   {gesture:15s}: {count:,} samples")
        
        return data
    
    def add_gaussian_noise(self, emg_values, noise_level=0.05):
        """Add Gaussian noise to EMG signals"""
        noise = np.random.normal(0, noise_level, emg_values.shape)
        return emg_values + noise * np.std(emg_values, axis=0)
    
    def add_amplitude_scaling(self, emg_values, scale_range=(0.8, 1.2)):
        """Scale EMG amplitude randomly"""
        scale_factors = np.random.uniform(scale_range[0], scale_range[1], emg_values.shape[1])
        return emg_values * scale_factors
    
    def add_baseline_shift(self, emg_values, shift_range=(-0.1, 0.1)):
        """Add random baseline shift"""
        shifts = np.random.uniform(shift_range[0], shift_range[1], emg_values.shape[1])
        return emg_values + shifts
    
    def add_channel_dropout(self, emg_values, dropout_prob=0.1):
        """Randomly set one channel to zero (simulate electrode disconnection)"""
        augmented = emg_values.copy()
        if np.random.random() < dropout_prob:
            channel_to_drop = np.random.randint(0, emg_values.shape[1])
            augmented[:, channel_to_drop] = 0
        return augmented
    
    def add_time_warping(self, emg_values, warp_factor=0.1):
        """Simulate slight timing variations"""
        # For single samples, we'll add small random variations to simulate timing effects
        time_noise = np.random.normal(0, warp_factor, emg_values.shape)
        return emg_values + time_noise * 0.1  # Small timing-related variations
    
    def add_electrode_shift(self, emg_values, shift_prob=0.2):
        """Simulate electrode position variations"""
        augmented = emg_values.copy()
        if np.random.random() < shift_prob:
            # Swap two channels occasionally
            if np.random.random() < 0.5:
                ch1, ch2 = np.random.choice(3, 2, replace=False)
                augmented[:, [ch1, ch2]] = augmented[:, [ch2, ch1]]
            else:
                # Add cross-channel interference
                interference = np.random.normal(0, 0.02, emg_values.shape)
                augmented += interference
        return augmented
    
    def augment_single_sample(self, row, augmentation_type):
        """Apply specific augmentation to a single sample"""
        # Extract clean EMG values
        emg_values = np.array([[row['emg1_clean'], row['emg2_clean'], row['emg3_clean']]])
        
        # Apply augmentation based on type
        if augmentation_type == 'noise':
            augmented_emg = self.add_gaussian_noise(emg_values, noise_level=0.03)
        elif augmentation_type == 'scale':
            augmented_emg = self.add_amplitude_scaling(emg_values, scale_range=(0.85, 1.15))
        elif augmentation_type == 'shift':
            augmented_emg = self.add_baseline_shift(emg_values, shift_range=(-0.05, 0.05))
        elif augmentation_type == 'dropout':
            augmented_emg = self.add_channel_dropout(emg_values, dropout_prob=0.15)
        elif augmentation_type == 'warp':
            augmented_emg = self.add_time_warping(emg_values, warp_factor=0.05)
        elif augmentation_type == 'electrode':
            augmented_emg = self.add_electrode_shift(emg_values, shift_prob=0.3)
        elif augmentation_type == 'combined':
            # Apply multiple augmentations
            augmented_emg = self.add_gaussian_noise(emg_values, noise_level=0.02)
            augmented_emg = self.add_amplitude_scaling(augmented_emg, scale_range=(0.9, 1.1))
            augmented_emg = self.add_baseline_shift(augmented_emg, shift_range=(-0.03, 0.03))
        else:
            augmented_emg = emg_values
        
        # Ensure values stay in reasonable range [0, 1]
        augmented_emg = np.clip(augmented_emg, 0, 1)
        
        # Create new row
        new_row = row.copy()
        new_row['emg1_clean'] = augmented_emg[0, 0]
        new_row['emg2_clean'] = augmented_emg[0, 1]
        new_row['emg3_clean'] = augmented_emg[0, 2]
        
        # Update raw values proportionally (approximate)
        scale1 = augmented_emg[0, 0] / (row['emg1_clean'] + 1e-6)
        scale2 = augmented_emg[0, 1] / (row['emg2_clean'] + 1e-6)
        scale3 = augmented_emg[0, 2] / (row['emg3_clean'] + 1e-6)
        
        new_row['ch1'] = int(np.clip(row['ch1'] * scale1, 1000, 15000))
        new_row['ch2'] = int(np.clip(row['ch2'] * scale2, 15000, 65000))
        new_row['ch3'] = int(np.clip(row['ch3'] * scale3, 15000, 65000))
        
        # Update timestamp to make it unique
        new_row['timestamp'] = row['timestamp'] + np.random.randint(1, 1000)
        
        return new_row
    
    def create_augmented_dataset(self, data, augmentation_factor=5):
        """Create augmented dataset with specified multiplication factor"""
        print(f"\n🔧 Creating augmented dataset...")
        print(f"📊 Original samples: {len(data):,}")
        print(f"📈 Augmentation factor: {augmentation_factor}x")
        print(f"🎯 Target samples: {len(data) * augmentation_factor:,}")
        
        augmentation_types = ['noise', 'scale', 'shift', 'dropout', 'warp', 'electrode', 'combined']
        
        augmented_data = []
        
        # Keep original data
        augmented_data.extend(data.to_dict('records'))
        
        # Create augmented samples
        for aug_round in range(augmentation_factor - 1):
            print(f"\n🔄 Augmentation round {aug_round + 1}/{augmentation_factor - 1}")
            
            for _, row in tqdm(data.iterrows(), total=len(data), desc="Augmenting samples"):
                # Choose random augmentation type
                aug_type = np.random.choice(augmentation_types)
                
                # Create augmented sample
                augmented_row = self.augment_single_sample(row, aug_type)
                augmented_data.append(augmented_row)
        
        # Convert to DataFrame
        augmented_df = pd.DataFrame(augmented_data)
        
        print(f"\n✅ Augmentation completed!")
        print(f"📊 Final dataset size: {len(augmented_df):,} samples")
        print(f"📈 Increase: {len(augmented_df) / len(data):.1f}x")
        
        # Show new gesture distribution
        print(f"\n🏷️ Augmented Gesture Distribution:")
        gesture_counts = augmented_df['gesture'].value_counts().sort_index()
        for gesture, count in gesture_counts.items():
            print(f"   {gesture:15s}: {count:,} samples")
        
        return augmented_df
    
    def add_quality_metrics(self, data):
        """Add data quality and diversity metrics"""
        print(f"\n📊 Adding quality metrics...")
        
        # Signal quality metrics
        data['signal_strength'] = data['emg1_clean'] + data['emg2_clean'] + data['emg3_clean']
        data['signal_balance'] = data[['emg1_clean', 'emg2_clean', 'emg3_clean']].std(axis=1)
        data['dominant_channel'] = data[['emg1_clean', 'emg2_clean', 'emg3_clean']].idxmax(axis=1)
        
        # Noise estimation (difference between clean and raw normalized)
        data['ch1_norm'] = (data['ch1'] - 1000) / (15000 - 1000)
        data['ch2_norm'] = (data['ch2'] - 15000) / (65000 - 15000)
        data['ch3_norm'] = (data['ch3'] - 15000) / (65000 - 15000)
        
        data['noise_level'] = (
            np.abs(data['emg1_clean'] - data['ch1_norm']) +
            np.abs(data['emg2_clean'] - data['ch2_norm']) +
            np.abs(data['emg3_clean'] - data['ch3_norm'])
        ) / 3
        
        # Remove temporary columns
        data = data.drop(['ch1_norm', 'ch2_norm', 'ch3_norm'], axis=1)
        
        print(f"✅ Quality metrics added")
        return data
    
    def save_augmented_dataset(self, data, filename='augmented_real_emg_dataset.csv'):
        """Save the augmented dataset"""
        print(f"\n💾 Saving augmented dataset...")
        
        # Add quality metrics
        data = self.add_quality_metrics(data)
        
        # Save to CSV
        data.to_csv(filename, index=False)
        
        print(f"✅ Dataset saved: {filename}")
        print(f"📊 Final shape: {data.shape}")
        print(f"📁 File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
        
        return filename
    
    def create_visualization(self, original_data, augmented_data):
        """Create visualization comparing original and augmented data"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Gesture distribution comparison
            plt.subplot(2, 3, 1)
            orig_counts = original_data['gesture'].value_counts().sort_index()
            aug_counts = augmented_data['gesture'].value_counts().sort_index()
            
            x = range(len(orig_counts))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], orig_counts.values, width, label='Original', alpha=0.7)
            plt.bar([i + width/2 for i in x], aug_counts.values, width, label='Augmented', alpha=0.7)
            
            plt.xlabel('Gesture')
            plt.ylabel('Sample Count')
            plt.title('Gesture Distribution: Original vs Augmented')
            plt.xticks(x, orig_counts.index, rotation=45)
            plt.legend()
            
            # EMG signal distribution
            plt.subplot(2, 3, 2)
            plt.hist(original_data['emg1_clean'], bins=50, alpha=0.5, label='Original EMG1', density=True)
            plt.hist(augmented_data['emg1_clean'], bins=50, alpha=0.5, label='Augmented EMG1', density=True)
            plt.xlabel('EMG1 Clean Value')
            plt.ylabel('Density')
            plt.title('EMG1 Signal Distribution')
            plt.legend()
            
            plt.subplot(2, 3, 3)
            plt.hist(original_data['emg2_clean'], bins=50, alpha=0.5, label='Original EMG2', density=True)
            plt.hist(augmented_data['emg2_clean'], bins=50, alpha=0.5, label='Augmented EMG2', density=True)
            plt.xlabel('EMG2 Clean Value')
            plt.ylabel('Density')
            plt.title('EMG2 Signal Distribution')
            plt.legend()
            
            plt.subplot(2, 3, 4)
            plt.hist(original_data['emg3_clean'], bins=50, alpha=0.5, label='Original EMG3', density=True)
            plt.hist(augmented_data['emg3_clean'], bins=50, alpha=0.5, label='Augmented EMG3', density=True)
            plt.xlabel('EMG3 Clean Value')
            plt.ylabel('Density')
            plt.title('EMG3 Signal Distribution')
            plt.legend()
            
            # Signal strength comparison
            plt.subplot(2, 3, 5)
            orig_strength = original_data['emg1_clean'] + original_data['emg2_clean'] + original_data['emg3_clean']
            aug_strength = augmented_data['signal_strength']
            
            plt.hist(orig_strength, bins=50, alpha=0.5, label='Original', density=True)
            plt.hist(aug_strength, bins=50, alpha=0.5, label='Augmented', density=True)
            plt.xlabel('Total Signal Strength')
            plt.ylabel('Density')
            plt.title('Signal Strength Distribution')
            plt.legend()
            
            # Data size comparison
            plt.subplot(2, 3, 6)
            sizes = [len(original_data), len(augmented_data)]
            labels = ['Original', 'Augmented']
            colors = ['skyblue', 'lightcoral']
            
            bars = plt.bar(labels, sizes, color=colors)
            plt.ylabel('Number of Samples')
            plt.title('Dataset Size Comparison')
            
            # Add value labels on bars
            for bar, size in zip(bars, sizes):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                        f'{size:,}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('augmented_dataset_analysis.png', dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved: augmented_dataset_analysis.png")
            
        except ImportError:
            print("⚠️  Matplotlib not available - skipping visualization")

def main():
    """Main augmentation function"""
    augmenter = EMGDataAugmenter()
    
    # Load real dataset
    data = augmenter.load_real_dataset()
    if data is None:
        return
    
    # Get augmentation factor from user
    print(f"\n📋 Augmentation Options:")
    print(f"1. 3x augmentation (~78k samples)")
    print(f"2. 5x augmentation (~130k samples)")
    print(f"3. 10x augmentation (~260k samples)")
    print(f"4. Custom factor")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        aug_factor = 3
    elif choice == '2':
        aug_factor = 5
    elif choice == '3':
        aug_factor = 10
    elif choice == '4':
        try:
            aug_factor = int(input("Enter augmentation factor: "))
        except ValueError:
            print("❌ Invalid number, using 5x")
            aug_factor = 5
    else:
        print("❌ Invalid choice, using 5x")
        aug_factor = 5
    
    # Create augmented dataset
    augmented_data = augmenter.create_augmented_dataset(data, aug_factor)
    
    # Save dataset
    filename = augmenter.save_augmented_dataset(augmented_data)
    
    # Create visualization
    augmenter.create_visualization(data, augmented_data)
    
    print(f"\n🎉 Augmented Dataset Creation Complete!")
    print(f"📊 Original: {len(data):,} samples")
    print(f"📊 Augmented: {len(augmented_data):,} samples")
    print(f"📈 Increase: {len(augmented_data) / len(data):.1f}x")
    print(f"💾 Saved as: {filename}")
    print(f"🚀 Ready for deep learning training!")

if __name__ == "__main__":
    main()
