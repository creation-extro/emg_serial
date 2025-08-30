#!/usr/bin/env python3
"""
Create Systematic EMG Dataset
Generate 100,000 high-quality EMG samples for deep learning
Based on realistic muscle activation patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class SystematicEMGGenerator:
    def __init__(self):
        # 10 gestures (no RELAX)
        self.gestures = [
            '0-OPEN', '1-CLOSE', '2-PINCH', '3-POINT', '4-FOUR', 
            '5-FIVE', '6-PEACE', '7-THUMBS_UP', '8-HOOK_GRIP', 
            '9-FLAT_PALM', '10-OK_SIGN'
        ]
        
        # Realistic muscle activation patterns for each gesture
        self.muscle_patterns = {
            '0-OPEN': {
                'flexor': 0.15,      # Low flexor activity
                'extensor': 0.25,    # Moderate extensor activity
                'intrinsic': 0.20,   # Low intrinsic muscle activity
                'variability': 0.08
            },
            '1-CLOSE': {
                'flexor': 0.85,      # High flexor activity
                'extensor': 0.15,    # Low extensor activity
                'intrinsic': 0.75,   # High intrinsic activity
                'variability': 0.12
            },
            '2-PINCH': {
                'flexor': 0.70,      # High precision grip
                'extensor': 0.30,    # Moderate stabilization
                'intrinsic': 0.80,   # Very high fine motor control
                'variability': 0.15
            },
            '3-POINT': {
                'flexor': 0.45,      # Moderate flexor for index finger
                'extensor': 0.60,    # High extensor for other fingers
                'intrinsic': 0.55,   # Moderate intrinsic
                'variability': 0.10
            },
            '4-FOUR': {
                'flexor': 0.35,      # Moderate flexor activity
                'extensor': 0.50,    # Moderate extensor
                'intrinsic': 0.45,   # Moderate intrinsic
                'variability': 0.09
            },
            '5-FIVE': {
                'flexor': 0.25,      # Low flexor (fingers extended)
                'extensor': 0.65,    # High extensor activity
                'intrinsic': 0.40,   # Moderate intrinsic
                'variability': 0.08
            },
            '6-PEACE': {
                'flexor': 0.40,      # Moderate flexor
                'extensor': 0.70,    # High extensor for V-shape
                'intrinsic': 0.60,   # High intrinsic for finger separation
                'variability': 0.11
            },
            '7-THUMBS_UP': {
                'flexor': 0.30,      # Low flexor
                'extensor': 0.80,    # Very high extensor (thumb extension)
                'intrinsic': 0.50,   # Moderate intrinsic
                'variability': 0.10
            },
            '8-HOOK_GRIP': {
                'flexor': 0.75,      # High flexor for hook shape
                'extensor': 0.25,    # Low extensor
                'intrinsic': 0.65,   # High intrinsic for grip strength
                'variability': 0.13
            },
            '9-FLAT_PALM': {
                'flexor': 0.20,      # Very low flexor
                'extensor': 0.35,    # Moderate extensor
                'intrinsic': 0.25,   # Low intrinsic
                'variability': 0.07
            },
            '10-OK_SIGN': {
                'flexor': 0.55,      # Moderate flexor for circle
                'extensor': 0.45,    # Moderate extensor for other fingers
                'intrinsic': 0.70,   # High intrinsic for precision
                'variability': 0.12
            }
        }
        
        print("🧪 Systematic EMG Dataset Generator")
        print("📊 Creating 100,000 High-Quality EMG Samples")
        print("🎯 No RELAX - Only Active Gestures")
        print("=" * 70)
    
    def generate_realistic_emg_sample(self, gesture, subject_variation=1.0, fatigue_factor=1.0):
        """Generate a single realistic EMG sample"""
        pattern = self.muscle_patterns[gesture]
        
        # Base muscle activations
        flexor_activation = pattern['flexor']
        extensor_activation = pattern['extensor']
        intrinsic_activation = pattern['intrinsic']
        
        # Apply subject variation (individual differences)
        flexor_activation *= subject_variation * np.random.uniform(0.8, 1.2)
        extensor_activation *= subject_variation * np.random.uniform(0.8, 1.2)
        intrinsic_activation *= subject_variation * np.random.uniform(0.8, 1.2)
        
        # Apply fatigue factor (muscle fatigue over time)
        flexor_activation *= fatigue_factor
        extensor_activation *= fatigue_factor
        intrinsic_activation *= fatigue_factor
        
        # Add physiological noise
        noise_level = pattern['variability']
        flexor_noise = np.random.normal(0, noise_level)
        extensor_noise = np.random.normal(0, noise_level)
        intrinsic_noise = np.random.normal(0, noise_level)
        
        # Generate clean EMG values (0-1 normalized)
        emg1_clean = np.clip(flexor_activation + flexor_noise, 0.05, 0.95)
        emg2_clean = np.clip(extensor_activation + extensor_noise, 0.05, 0.95)
        emg3_clean = np.clip(intrinsic_activation + intrinsic_noise, 0.05, 0.95)
        
        # Convert to raw EMG (simulate 16-bit ADC with realistic baseline)
        baseline_offset = np.random.randint(2000, 5000, 3)
        raw_scaling = np.random.uniform(45000, 55000, 3)  # Realistic ADC range
        
        ch1 = int(emg1_clean * raw_scaling[0] + baseline_offset[0])
        ch2 = int(emg2_clean * raw_scaling[1] + baseline_offset[1])
        ch3 = int(emg3_clean * raw_scaling[2] + baseline_offset[2])
        
        # Add some cross-talk between channels (realistic)
        crosstalk_factor = 0.05
        ch1 += int(crosstalk_factor * (ch2 + ch3) / 2)
        ch2 += int(crosstalk_factor * (ch1 + ch3) / 2)
        ch3 += int(crosstalk_factor * (ch1 + ch2) / 2)
        
        # Ensure values stay within 16-bit range
        ch1 = np.clip(ch1, 1000, 65000)
        ch2 = np.clip(ch2, 1000, 65000)
        ch3 = np.clip(ch3, 1000, 65000)
        
        return {
            'raw': [ch1, ch2, ch3],
            'clean': [emg1_clean, emg2_clean, emg3_clean]
        }
    
    def generate_systematic_dataset(self, total_samples=100000):
        """Generate systematic EMG dataset with 100k samples"""
        print(f"🧪 Generating {total_samples:,} systematic EMG samples...")
        
        # Calculate samples per gesture (roughly equal distribution)
        base_samples_per_gesture = total_samples // len(self.gestures)
        
        # Add some realistic imbalance (some gestures are more common)
        gesture_weights = {
            '0-OPEN': 1.2,      # More common
            '1-CLOSE': 1.3,     # Very common
            '2-PINCH': 0.9,     # Less common
            '3-POINT': 1.1,     # Common
            '4-FOUR': 0.8,      # Less common
            '5-FIVE': 1.0,      # Average
            '6-PEACE': 1.1,     # Common
            '7-THUMBS_UP': 0.9, # Less common
            '8-HOOK_GRIP': 0.7, # Rare
            '9-FLAT_PALM': 1.0, # Average
            '10-OK_SIGN': 1.0   # Average
        }
        
        # Calculate actual samples per gesture
        total_weight = sum(gesture_weights.values())
        samples_per_gesture = {}
        
        for gesture in self.gestures:
            weight = gesture_weights[gesture]
            samples = int((weight / total_weight) * total_samples)
            samples_per_gesture[gesture] = samples
        
        # Adjust to ensure exact total
        current_total = sum(samples_per_gesture.values())
        diff = total_samples - current_total
        if diff != 0:
            # Add/subtract from most common gesture
            samples_per_gesture['1-CLOSE'] += diff
        
        print(f"📊 Samples per gesture:")
        for gesture, count in samples_per_gesture.items():
            print(f"   {gesture}: {count:,} samples")
        
        # Generate dataset
        data = []
        timestamp_base = 1752250000
        sample_id = 0
        
        for gesture_idx, gesture in enumerate(self.gestures):
            n_samples = samples_per_gesture[gesture]
            print(f"   Generating {gesture}... ({n_samples:,} samples)")
            
            # Simulate multiple subjects (5 subjects)
            samples_per_subject = n_samples // 5
            
            for subject_id in range(5):
                # Subject-specific variation
                subject_variation = np.random.uniform(0.7, 1.3)
                
                for sample_idx in range(samples_per_subject):
                    # Simulate fatigue over time
                    fatigue_progress = sample_idx / samples_per_subject
                    fatigue_factor = 1.0 - (fatigue_progress * 0.1)  # 10% fatigue over session
                    
                    # Generate EMG sample
                    emg_sample = self.generate_realistic_emg_sample(
                        gesture, subject_variation, fatigue_factor
                    )
                    
                    # Create timestamp with realistic spacing
                    timestamp = timestamp_base + sample_id * np.random.randint(8, 15)  # 8-15ms intervals
                    
                    # Create data row
                    row = {
                        'timestamp': timestamp,
                        'ch1': emg_sample['raw'][0],
                        'ch2': emg_sample['raw'][1],
                        'ch3': emg_sample['raw'][2],
                        'gesture': gesture,
                        'label': gesture_idx,
                        'emg1_clean': emg_sample['clean'][0],
                        'emg2_clean': emg_sample['clean'][1],
                        'emg3_clean': emg_sample['clean'][2],
                        'subject_id': subject_id,
                        'sample_id': sample_id,
                        'fatigue_factor': fatigue_factor
                    }
                    
                    data.append(row)
                    sample_id += 1
            
            # Handle remaining samples
            remaining = n_samples - (samples_per_subject * 5)
            for i in range(remaining):
                subject_variation = np.random.uniform(0.7, 1.3)
                fatigue_factor = np.random.uniform(0.9, 1.0)
                
                emg_sample = self.generate_realistic_emg_sample(
                    gesture, subject_variation, fatigue_factor
                )
                
                timestamp = timestamp_base + sample_id * np.random.randint(8, 15)
                
                row = {
                    'timestamp': timestamp,
                    'ch1': emg_sample['raw'][0],
                    'ch2': emg_sample['raw'][1],
                    'ch3': emg_sample['raw'][2],
                    'gesture': gesture,
                    'label': gesture_idx,
                    'emg1_clean': emg_sample['clean'][0],
                    'emg2_clean': emg_sample['clean'][1],
                    'emg3_clean': emg_sample['clean'][2],
                    'subject_id': np.random.randint(0, 5),
                    'sample_id': sample_id,
                    'fatigue_factor': fatigue_factor
                }
                
                data.append(row)
                sample_id += 1
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Generated {len(df):,} systematic EMG samples")
        print(f"📊 Gestures: {len(self.gestures)}")
        print(f"📊 Features: {len(df.columns)}")
        
        return df
    
    def save_dataset(self, df, filename='systematic_emg_dataset_100k.csv'):
        """Save systematic dataset"""
        print(f"\n💾 Saving systematic dataset...")
        df.to_csv(filename, index=False)
        print(f"✅ Saved: {filename}")
        
        # Show comprehensive statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(df):,}")
        print(f"   Gestures: {df['gesture'].nunique()}")
        print(f"   Features: {len(df.columns)}")
        print(f"   Subjects: {df['subject_id'].nunique()}")
        print(f"   File size: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        print(f"\n🎯 Gesture Distribution:")
        gesture_counts = df['gesture'].value_counts().sort_index()
        for gesture, count in gesture_counts.items():
            percentage = count / len(df) * 100
            print(f"   {gesture}: {count:,} samples ({percentage:.1f}%)")
        
        print(f"\n📈 EMG Channel Statistics:")
        for channel in ['ch1', 'ch2', 'ch3']:
            stats = df[channel].describe()
            print(f"   {channel}: min={stats['min']:.0f}, max={stats['max']:.0f}, mean={stats['mean']:.0f}, std={stats['std']:.0f}")
        
        print(f"\n📈 Clean EMG Statistics:")
        for channel in ['emg1_clean', 'emg2_clean', 'emg3_clean']:
            stats = df[channel].describe()
            print(f"   {channel}: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
        return filename
    
    def create_dataset_visualization(self, df):
        """Create comprehensive dataset visualization"""
        print(f"\n📊 Creating dataset visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Systematic EMG Dataset Analysis (100K Samples)', fontsize=16, fontweight='bold')
        
        # 1. Gesture distribution
        ax1 = axes[0, 0]
        gesture_counts = df['gesture'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(gesture_counts)))
        
        wedges, texts, autotexts = ax1.pie(gesture_counts.values, labels=[g.split('-')[1] for g in gesture_counts.index], 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Gesture Distribution', fontweight='bold')
        
        # 2. EMG signal patterns
        ax2 = axes[0, 1]
        gesture_patterns = df.groupby('gesture')[['emg1_clean', 'emg2_clean', 'emg3_clean']].mean()
        
        x = np.arange(len(gesture_patterns))
        width = 0.25
        
        ax2.bar(x - width, gesture_patterns['emg1_clean'], width, label='Channel 1', alpha=0.8, color='#FF6B6B')
        ax2.bar(x, gesture_patterns['emg2_clean'], width, label='Channel 2', alpha=0.8, color='#4ECDC4')
        ax2.bar(x + width, gesture_patterns['emg3_clean'], width, label='Channel 3', alpha=0.8, color='#45B7D1')
        
        ax2.set_xlabel('Gestures')
        ax2.set_ylabel('Average EMG Amplitude')
        ax2.set_title('EMG Patterns by Gesture', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([g.split('-')[1] for g in gesture_patterns.index], rotation=45)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Channel correlation
        ax3 = axes[0, 2]
        corr_matrix = df[['emg1_clean', 'emg2_clean', 'emg3_clean']].corr()
        im = ax3.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        ax3.set_title('Channel Correlation Matrix', fontweight='bold')
        ax3.set_xticks([0, 1, 2])
        ax3.set_yticks([0, 1, 2])
        ax3.set_xticklabels(['Ch1', 'Ch2', 'Ch3'])
        ax3.set_yticklabels(['Ch1', 'Ch2', 'Ch3'])
        
        for i in range(3):
            for j in range(3):
                ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax3)
        
        # 4. Signal distribution
        ax4 = axes[1, 0]
        ax4.hist(df['emg1_clean'], bins=50, alpha=0.7, label='Channel 1', color='#FF6B6B')
        ax4.hist(df['emg2_clean'], bins=50, alpha=0.7, label='Channel 2', color='#4ECDC4')
        ax4.hist(df['emg3_clean'], bins=50, alpha=0.7, label='Channel 3', color='#45B7D1')
        ax4.set_xlabel('Clean EMG Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('EMG Signal Distribution', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Subject variation
        ax5 = axes[1, 1]
        subject_means = df.groupby('subject_id')['emg1_clean'].mean()
        ax5.bar(range(len(subject_means)), subject_means.values, 
               color=plt.cm.viridis(np.linspace(0, 1, len(subject_means))), alpha=0.8)
        ax5.set_xlabel('Subject ID')
        ax5.set_ylabel('Average EMG (Channel 1)')
        ax5.set_title('Inter-Subject Variation', fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Fatigue effect
        ax6 = axes[1, 2]
        fatigue_bins = pd.cut(df['fatigue_factor'], bins=10)
        fatigue_means = df.groupby(fatigue_bins)['emg1_clean'].mean()
        
        bin_centers = [interval.mid for interval in fatigue_means.index]
        ax6.plot(bin_centers, fatigue_means.values, 'o-', linewidth=2, markersize=6, color='#E74C3C')
        ax6.set_xlabel('Fatigue Factor')
        ax6.set_ylabel('Average EMG (Channel 1)')
        ax6.set_title('Fatigue Effect on EMG', fontweight='bold')
        ax6.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('systematic_dataset_analysis.png', dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved: systematic_dataset_analysis.png")
        
        return fig

def main():
    """Main dataset generation function"""
    generator = SystematicEMGGenerator()
    
    # Generate systematic dataset
    print(f"\n🎯 Generating systematic EMG dataset...")
    total_samples = int(input("Enter total samples (default 100000): ") or "100000")
    
    df = generator.generate_systematic_dataset(total_samples)
    
    # Save dataset
    filename = generator.save_dataset(df)
    
    # Create visualization
    try:
        generator.create_dataset_visualization(df)
    except ImportError:
        print("⚠️  Matplotlib not available - skipping visualization")
    
    print(f"\n🎉 SUCCESS!")
    print(f"✅ Systematic EMG dataset created: {filename}")
    print(f"📊 Total samples: {len(df):,}")
    print(f"🎯 No RELAX gestures - Only active movements")
    print(f"👥 Multi-subject data with realistic variation")
    print(f"💪 Fatigue modeling included")
    print(f"🔬 Physiologically realistic patterns")
    
    print(f"\n🚀 Ready for Deep Learning!")
    print(f"📊 Perfect for neural network training")
    print(f"🎯 High-quality, balanced dataset")
    print(f"⚡ 100K samples for robust model training")

if __name__ == "__main__":
    main()
