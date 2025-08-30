#!/usr/bin/env python3
"""
Create Synthetic EMG Dataset
Generate realistic EMG patterns for gesture recognition
Based on typical EMG signal characteristics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SyntheticEMGGenerator:
    def __init__(self):
        self.gestures = [
            '0-OPEN', '1-CLOSE', '2-PINCH', '3-POINT', '4-FOUR', '5-FIVE',
            '6-PEACE', '7-THUMBS_UP', '8-HOOK_GRIP', '9-FLAT_PALM', '10-OK_SIGN'
        ]
        
        # Typical EMG patterns for each gesture (normalized 0-1)
        self.gesture_patterns = {
            '0-OPEN': [0.2, 0.3, 0.25],      # Low activity
            '1-CLOSE': [0.8, 0.9, 0.85],     # High activity all channels
            '2-PINCH': [0.7, 0.4, 0.3],      # High ch1, moderate others
            '3-POINT': [0.3, 0.6, 0.5],      # Moderate activity
            '4-FOUR': [0.5, 0.7, 0.6],       # Moderate-high activity
            '5-FIVE': [0.4, 0.5, 0.45],      # Balanced moderate activity
            '6-PEACE': [0.6, 0.8, 0.7],      # High activity
            '7-THUMBS_UP': [0.9, 0.3, 0.4],  # Very high ch1, low others
            '8-HOOK_GRIP': [0.5, 0.9, 0.6],  # Very high ch2
            '9-FLAT_PALM': [0.3, 0.4, 0.35], # Low-moderate activity
            '10-OK_SIGN': [0.3, 0.6, 0.55]   # Moderate activity
        }
        
        print("🧪 Synthetic EMG Dataset Generator")
        print("📊 Creating Realistic EMG Patterns")
        print("=" * 60)
    
    def generate_emg_sample(self, gesture, noise_level=0.05):
        """Generate a single EMG sample for a gesture"""
        base_pattern = self.gesture_patterns[gesture]
        
        # Add realistic noise and variation
        noise = np.random.normal(0, noise_level, 3)
        muscle_fatigue = np.random.uniform(0.95, 1.05)  # Slight fatigue variation
        individual_variation = np.random.normal(1.0, 0.1, 3)  # Individual differences
        
        # Generate clean EMG (0-1 normalized)
        clean_emg = []
        for i, base_val in enumerate(base_pattern):
            clean_val = base_val * muscle_fatigue * individual_variation[i] + noise[i]
            clean_val = max(0.0, min(1.0, clean_val))  # Clamp to 0-1
            clean_emg.append(clean_val)
        
        # Convert to raw EMG (simulate 16-bit ADC)
        raw_emg = [int(val * 65535) for val in clean_emg]
        
        # Add some baseline offset
        baseline_offset = np.random.randint(500, 2000, 3)
        raw_emg = [raw + offset for raw, offset in zip(raw_emg, baseline_offset)]
        
        return raw_emg, clean_emg
    
    def generate_dataset(self, samples_per_gesture=1000):
        """Generate complete synthetic EMG dataset"""
        print(f"🧪 Generating {samples_per_gesture} samples per gesture...")
        
        data = []
        timestamp_base = 1752250000
        
        for gesture_idx, gesture in enumerate(self.gestures):
            print(f"   Generating {gesture}...")
            
            for sample_idx in range(samples_per_gesture):
                # Generate EMG sample
                raw_emg, clean_emg = self.generate_emg_sample(gesture)
                
                # Create timestamp
                timestamp = timestamp_base + gesture_idx * 10000 + sample_idx
                
                # Create data row
                row = {
                    'timestamp': timestamp,
                    'ch1': raw_emg[0],
                    'ch2': raw_emg[1], 
                    'ch3': raw_emg[2],
                    'gesture': gesture,
                    'label': gesture_idx,
                    'emg1_clean': clean_emg[0],
                    'emg2_clean': clean_emg[1],
                    'emg3_clean': clean_emg[2]
                }
                
                data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        print(f"✅ Generated {len(df):,} synthetic EMG samples")
        print(f"📊 Gestures: {len(self.gestures)}")
        print(f"📊 Samples per gesture: {samples_per_gesture}")
        
        return df
    
    def save_dataset(self, df, filename='synthetic_emg_dataset.csv'):
        """Save synthetic dataset"""
        df.to_csv(filename, index=False)
        print(f"💾 Saved synthetic dataset: {filename}")
        
        # Show statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(df):,}")
        print(f"   Gestures: {df['gesture'].nunique()}")
        print(f"   Features: {len(df.columns)}")
        
        print(f"\n🎯 Gesture Distribution:")
        for gesture, count in df['gesture'].value_counts().sort_index().items():
            print(f"   {gesture}: {count:,} samples")
        
        print(f"\n📈 EMG Channel Ranges:")
        for channel in ['ch1', 'ch2', 'ch3']:
            stats = df[channel].describe()
            print(f"   {channel}: {stats['min']:.0f} - {stats['max']:.0f} (mean: {stats['mean']:.0f})")
        
        return filename
    
    def visualize_patterns(self, df):
        """Visualize EMG patterns"""
        print(f"\n📊 Creating visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Synthetic EMG Dataset Patterns')
        
        # Plot 1: Clean EMG patterns by gesture
        ax1 = axes[0, 0]
        for gesture in self.gestures[:5]:  # Show first 5 gestures
            gesture_data = df[df['gesture'] == gesture]
            sample_data = gesture_data.head(100)  # First 100 samples
            
            ax1.plot(sample_data['emg1_clean'], label=f'{gesture} Ch1', alpha=0.7)
        
        ax1.set_title('Clean EMG Patterns (Channel 1)')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Clean EMG (0-1)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Raw EMG distribution
        ax2 = axes[0, 1]
        ax2.hist(df['ch1'], bins=50, alpha=0.7, label='Channel 1')
        ax2.hist(df['ch2'], bins=50, alpha=0.7, label='Channel 2')
        ax2.hist(df['ch3'], bins=50, alpha=0.7, label='Channel 3')
        ax2.set_title('Raw EMG Distribution')
        ax2.set_xlabel('Raw EMG Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Gesture patterns comparison
        ax3 = axes[1, 0]
        gesture_means = df.groupby('gesture')[['emg1_clean', 'emg2_clean', 'emg3_clean']].mean()
        
        x = np.arange(len(self.gestures))
        width = 0.25
        
        ax3.bar(x - width, gesture_means['emg1_clean'], width, label='Channel 1', alpha=0.8)
        ax3.bar(x, gesture_means['emg2_clean'], width, label='Channel 2', alpha=0.8)
        ax3.bar(x + width, gesture_means['emg3_clean'], width, label='Channel 3', alpha=0.8)
        
        ax3.set_title('Average EMG Patterns by Gesture')
        ax3.set_xlabel('Gesture')
        ax3.set_ylabel('Average Clean EMG')
        ax3.set_xticks(x)
        ax3.set_xticklabels([g.split('-')[1] for g in self.gestures], rotation=45)
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Channel correlation
        ax4 = axes[1, 1]
        correlation_matrix = df[['emg1_clean', 'emg2_clean', 'emg3_clean']].corr()
        
        im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        ax4.set_title('EMG Channel Correlations')
        ax4.set_xticks([0, 1, 2])
        ax4.set_yticks([0, 1, 2])
        ax4.set_xticklabels(['Ch1', 'Ch2', 'Ch3'])
        ax4.set_yticklabels(['Ch1', 'Ch2', 'Ch3'])
        
        # Add correlation values
        for i in range(3):
            for j in range(3):
                ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='black')
        
        plt.colorbar(im, ax=ax4)
        plt.tight_layout()
        plt.savefig('synthetic_emg_patterns.png', dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved: synthetic_emg_patterns.png")
        
        return fig

def main():
    """Main function"""
    generator = SyntheticEMGGenerator()
    
    # Generate dataset
    samples_per_gesture = int(input("Enter samples per gesture (default 1000): ") or "1000")
    df = generator.generate_dataset(samples_per_gesture)
    
    # Save dataset
    filename = generator.save_dataset(df)
    
    # Create visualization
    try:
        generator.visualize_patterns(df)
    except ImportError:
        print("⚠️  Matplotlib not available - skipping visualization")
    
    print(f"\n🎉 SUCCESS!")
    print(f"✅ Synthetic EMG dataset created: {filename}")
    print(f"📊 Total samples: {len(df):,}")
    print(f"🎯 Ready for training!")
    
    print(f"\n🚀 Next Steps:")
    print(f"1. Use this dataset to train your model")
    print(f"2. Test with: python train_simple_working_model.py")
    print(f"3. Replace data path in training script")
    print(f"4. Should achieve high accuracy!")

if __name__ == "__main__":
    main()
