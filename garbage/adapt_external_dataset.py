#!/usr/bin/env python3
"""
Adapt External EMG Datasets to Your 3-Channel MyoBand Setup
Convert 8-channel datasets to 3-channel format compatible with your equipment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

class EMGDatasetAdapter:
    def __init__(self):
        self.myoband_channels = 3  # Your MyoBand has 3 channels
        
        print("🔄 EMG Dataset Adapter")
        print("📊 Convert External Datasets → Your 3-Channel MyoBand")
        print("🎯 Compatible with Ninapro, UCI, CapgMyo datasets")
        print("=" * 70)
    
    def load_external_dataset(self, file_path, dataset_type='ninapro'):
        """Load external EMG dataset"""
        print(f"📂 Loading external dataset: {dataset_type}")
        
        try:
            if dataset_type.lower() == 'ninapro':
                # Ninapro format: usually has emg1, emg2, ..., emg8, stimulus, repetition
                data = pd.read_csv(file_path)
                print(f"✅ Loaded Ninapro dataset: {data.shape}")
                
            elif dataset_type.lower() == 'uci':
                # UCI format: different column names
                data = pd.read_csv(file_path)
                print(f"✅ Loaded UCI dataset: {data.shape}")
                
            elif dataset_type.lower() == 'capgmyo':
                # CapgMyo format: high-density EMG
                data = pd.read_csv(file_path)
                print(f"✅ Loaded CapgMyo dataset: {data.shape}")
                
            else:
                # Generic CSV format
                data = pd.read_csv(file_path)
                print(f"✅ Loaded generic dataset: {data.shape}")
            
            print(f"📊 Columns: {list(data.columns)}")
            return data
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def identify_emg_channels(self, data):
        """Automatically identify EMG channels in the dataset"""
        print(f"\n🔍 Identifying EMG channels...")
        
        emg_columns = []
        
        # Common EMG column patterns
        patterns = ['emg', 'EMG', 'channel', 'ch', 'sensor', 'electrode']
        
        for col in data.columns:
            col_lower = col.lower()
            if any(pattern.lower() in col_lower for pattern in patterns):
                # Check if it's numeric data
                if data[col].dtype in ['int64', 'float64']:
                    emg_columns.append(col)
        
        # If no pattern match, look for numeric columns (excluding obvious non-EMG)
        if not emg_columns:
            exclude_patterns = ['time', 'label', 'class', 'gesture', 'stimulus', 'repetition', 'subject']
            for col in data.columns:
                if data[col].dtype in ['int64', 'float64']:
                    if not any(pattern.lower() in col.lower() for pattern in exclude_patterns):
                        emg_columns.append(col)
        
        print(f"✅ Found {len(emg_columns)} EMG channels: {emg_columns}")
        return emg_columns
    
    def identify_label_column(self, data):
        """Identify the gesture/label column"""
        print(f"\n🎯 Identifying label column...")
        
        label_patterns = ['label', 'class', 'gesture', 'stimulus', 'movement', 'action']
        
        for col in data.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in label_patterns):
                print(f"✅ Found label column: {col}")
                return col
        
        # If no pattern match, assume last column or ask user
        print(f"⚠️  No obvious label column found")
        print(f"📊 Available columns: {list(data.columns)}")
        return None
    
    def reduce_channels_pca(self, emg_data, n_components=3):
        """Reduce EMG channels using PCA to match your 3-channel setup"""
        print(f"\n📉 Reducing {emg_data.shape[1]} channels → {n_components} channels using PCA...")
        
        # Standardize the data
        scaler = StandardScaler()
        emg_scaled = scaler.fit_transform(emg_data)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        emg_reduced = pca.fit_transform(emg_scaled)
        
        # Show explained variance
        explained_variance = pca.explained_variance_ratio_
        total_variance = np.sum(explained_variance)
        
        print(f"✅ PCA completed:")
        print(f"   📊 Explained variance per component: {explained_variance}")
        print(f"   📊 Total variance retained: {total_variance:.3f} ({total_variance*100:.1f}%)")
        
        return emg_reduced, pca, scaler
    
    def reduce_channels_selection(self, emg_data, n_components=3):
        """Select best EMG channels based on variance and correlation"""
        print(f"\n🎯 Selecting best {n_components} channels from {emg_data.shape[1]} channels...")
        
        # Calculate variance for each channel
        variances = np.var(emg_data, axis=0)
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(emg_data.T)
        
        # Select channels with high variance and low correlation
        selected_indices = []
        remaining_indices = list(range(emg_data.shape[1]))
        
        # First, select channel with highest variance
        first_idx = np.argmax(variances)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining channels with low correlation to already selected
        for _ in range(n_components - 1):
            best_idx = None
            best_score = -1
            
            for idx in remaining_indices:
                # Calculate average correlation with already selected channels
                avg_corr = np.mean([abs(corr_matrix[idx, sel_idx]) for sel_idx in selected_indices])
                # Score = variance / (1 + correlation)
                score = variances[idx] / (1 + avg_corr)
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        selected_data = emg_data[:, selected_indices]
        
        print(f"✅ Selected channels: {selected_indices}")
        print(f"   📊 Variances: {variances[selected_indices]}")
        
        return selected_data, selected_indices
    
    def create_myoband_format(self, emg_reduced, labels, method='pca'):
        """Create dataset in your MyoBand format"""
        print(f"\n🔄 Creating MyoBand-compatible format...")
        
        # Create timestamp (simulate your format)
        n_samples = emg_reduced.shape[0]
        timestamps = np.arange(1752250000, 1752250000 + n_samples)
        
        # Create raw EMG values (simulate 16-bit ADC like your MyoBand)
        # Normalize to 0-1 first, then scale to 16-bit range
        emg_normalized = (emg_reduced - emg_reduced.min(axis=0)) / (emg_reduced.max(axis=0) - emg_reduced.min(axis=0))
        raw_emg = (emg_normalized * 65535).astype(int)
        
        # Add some realistic baseline offset
        baseline_offset = np.random.randint(1000, 3000, (n_samples, 3))
        raw_emg = raw_emg + baseline_offset
        
        # Create clean EMG (normalized 0-1)
        clean_emg = emg_normalized
        
        # Create DataFrame in your format
        adapted_data = pd.DataFrame({
            'timestamp': timestamps,
            'ch1': raw_emg[:, 0],
            'ch2': raw_emg[:, 1],
            'ch3': raw_emg[:, 2],
            'gesture': labels,
            'label': pd.Categorical(labels).codes,
            'emg1_clean': clean_emg[:, 0],
            'emg2_clean': clean_emg[:, 1],
            'emg3_clean': clean_emg[:, 2]
        })
        
        print(f"✅ Created MyoBand-compatible dataset:")
        print(f"   📊 Shape: {adapted_data.shape}")
        print(f"   📊 Columns: {list(adapted_data.columns)}")
        print(f"   🎯 Gestures: {adapted_data['gesture'].nunique()}")
        
        return adapted_data
    
    def save_adapted_dataset(self, adapted_data, filename='adapted_emg_dataset.csv'):
        """Save the adapted dataset"""
        adapted_data.to_csv(filename, index=False)
        print(f"\n💾 Saved adapted dataset: {filename}")
        
        # Show statistics
        print(f"\n📊 Adapted Dataset Statistics:")
        print(f"   Total samples: {len(adapted_data):,}")
        print(f"   Gestures: {adapted_data['gesture'].nunique()}")
        
        print(f"\n🎯 Gesture Distribution:")
        gesture_counts = adapted_data['gesture'].value_counts()
        for gesture, count in gesture_counts.items():
            print(f"   {gesture}: {count:,} samples")
        
        print(f"\n📈 EMG Channel Ranges:")
        for channel in ['ch1', 'ch2', 'ch3']:
            stats = adapted_data[channel].describe()
            print(f"   {channel}: {stats['min']:.0f} - {stats['max']:.0f} (mean: {stats['mean']:.0f})")
        
        return filename
    
    def visualize_adaptation(self, original_emg, adapted_data):
        """Visualize the adaptation process"""
        print(f"\n📊 Creating adaptation visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('EMG Dataset Adaptation: External → MyoBand Format', fontsize=16)
        
        # Plot 1: Original channels
        ax1 = axes[0, 0]
        n_show = min(5, original_emg.shape[1])
        for i in range(n_show):
            ax1.plot(original_emg[:1000, i], label=f'Original Ch{i+1}', alpha=0.7)
        ax1.set_title(f'Original EMG Channels ({original_emg.shape[1]} channels)')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('EMG Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Adapted channels
        ax2 = axes[0, 1]
        ax2.plot(adapted_data['emg1_clean'][:1000], label='Adapted Ch1', alpha=0.8)
        ax2.plot(adapted_data['emg2_clean'][:1000], label='Adapted Ch2', alpha=0.8)
        ax2.plot(adapted_data['emg3_clean'][:1000], label='Adapted Ch3', alpha=0.8)
        ax2.set_title('Adapted EMG Channels (3 channels)')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Clean EMG (0-1)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Gesture distribution
        ax3 = axes[1, 0]
        gesture_counts = adapted_data['gesture'].value_counts()
        ax3.bar(range(len(gesture_counts)), gesture_counts.values)
        ax3.set_title('Gesture Distribution')
        ax3.set_xlabel('Gesture')
        ax3.set_ylabel('Sample Count')
        ax3.set_xticks(range(len(gesture_counts)))
        ax3.set_xticklabels(gesture_counts.index, rotation=45)
        ax3.grid(True)
        
        # Plot 4: Channel correlation
        ax4 = axes[1, 1]
        corr_matrix = adapted_data[['emg1_clean', 'emg2_clean', 'emg3_clean']].corr()
        im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        ax4.set_title('Adapted Channel Correlations')
        ax4.set_xticks([0, 1, 2])
        ax4.set_yticks([0, 1, 2])
        ax4.set_xticklabels(['Ch1', 'Ch2', 'Ch3'])
        ax4.set_yticklabels(['Ch1', 'Ch2', 'Ch3'])
        
        # Add correlation values
        for i in range(3):
            for j in range(3):
                ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='black')
        
        plt.colorbar(im, ax=ax4)
        plt.tight_layout()
        plt.savefig('dataset_adaptation_visualization.png', dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved: dataset_adaptation_visualization.png")

def main():
    """Main adaptation function"""
    adapter = EMGDatasetAdapter()
    
    print(f"\n📋 Dataset Adaptation Options:")
    print(f"1. 📂 Load and adapt external dataset")
    print(f"2. 🧪 Create sample adapted dataset")
    print(f"3. 🚪 Exit")
    
    choice = input(f"\nChoose option (1-3): ").strip()
    
    if choice == '1':
        file_path = input("Enter path to external EMG dataset CSV: ").strip()
        dataset_type = input("Dataset type (ninapro/uci/capgmyo/generic): ").strip() or 'generic'
        
        # Load external dataset
        data = adapter.load_external_dataset(file_path, dataset_type)
        if data is None:
            return
        
        # Identify EMG channels and labels
        emg_columns = adapter.identify_emg_channels(data)
        label_column = adapter.identify_label_column(data)
        
        if not emg_columns or label_column is None:
            print("❌ Could not identify EMG channels or labels")
            return
        
        # Extract EMG data and labels
        emg_data = data[emg_columns].values
        labels = data[label_column].values
        
        # Choose reduction method
        method = input("Reduction method (pca/selection): ").strip() or 'pca'
        
        if method.lower() == 'pca':
            emg_reduced, pca, scaler = adapter.reduce_channels_pca(emg_data, 3)
        else:
            emg_reduced, selected_indices = adapter.reduce_channels_selection(emg_data, 3)
        
        # Create MyoBand format
        adapted_data = adapter.create_myoband_format(emg_reduced, labels, method)
        
        # Save adapted dataset
        filename = adapter.save_adapted_dataset(adapted_data)
        
        # Create visualization
        try:
            adapter.visualize_adaptation(emg_data, adapted_data)
        except ImportError:
            print("⚠️  Matplotlib not available - skipping visualization")
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ External dataset adapted to MyoBand format")
        print(f"💾 Saved as: {filename}")
        print(f"🚀 Ready for training with your 3-channel setup!")
        
    elif choice == '2':
        print(f"\n🧪 Creating sample adapted dataset...")
        
        # Create sample data (simulating 8-channel external dataset)
        n_samples = 5000
        n_gestures = 5
        
        # Simulate 8-channel EMG data
        original_emg = np.random.randn(n_samples, 8) * 100 + 500
        labels = np.random.choice(['gesture_1', 'gesture_2', 'gesture_3', 'gesture_4', 'gesture_5'], n_samples)
        
        # Reduce to 3 channels using PCA
        emg_reduced, pca, scaler = adapter.reduce_channels_pca(original_emg, 3)
        
        # Create MyoBand format
        adapted_data = adapter.create_myoband_format(emg_reduced, labels)
        
        # Save sample dataset
        filename = adapter.save_adapted_dataset(adapted_data, 'sample_adapted_dataset.csv')
        
        print(f"\n🎉 Sample adapted dataset created!")
        print(f"💾 Saved as: {filename}")
        
    else:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
