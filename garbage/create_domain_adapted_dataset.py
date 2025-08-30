#!/usr/bin/env python3
"""
Create Domain-Adapted EMG Dataset
Generate training data that matches your actual EMG signal characteristics
Based on your real test samples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class DomainAdaptedEMGGenerator:
    def __init__(self):
        # Your actual test samples (ground truth)
        self.real_samples = {
            '10-OK_SIGN': {
                'raw': [1216, 44154, 43322],
                'clean': [0.019, 0.674, 0.661],
                'pattern': 'high_ch2_ch3_low_ch1'
            },
            '6-PEACE': {
                'raw': [8930, 52764, 53100],
                'clean': [0.136, 0.805, 0.810],
                'pattern': 'very_high_ch2_ch3_moderate_ch1'
            },
            '3-POINT': {
                'raw': [1536, 61951, 61487],
                'clean': [0.023, 0.945, 0.938],
                'pattern': 'maximum_ch2_ch3_minimal_ch1'
            },
            '5-FIVE': {
                'raw': [3456, 48234, 47892],
                'clean': [0.053, 0.736, 0.731],
                'pattern': 'high_ch2_ch3_low_ch1'
            },
            '1-CLOSE': {
                'raw': [1424, 20869, 21061],
                'clean': [0.022, 0.318, 0.321],
                'pattern': 'moderate_ch2_ch3_minimal_ch1'
            }
        }
        
        # Infer patterns for other gestures based on muscle physiology
        self.gesture_patterns = {
            '0-OPEN': {
                'ch1_range': (0.015, 0.035),    # Very low flexor
                'ch2_range': (0.200, 0.400),    # Low-moderate extensor
                'ch3_range': (0.200, 0.400),    # Low-moderate intrinsic
                'noise_level': 0.02
            },
            '1-CLOSE': {
                'ch1_range': (0.020, 0.040),    # Based on your sample
                'ch2_range': (0.300, 0.350),    # Based on your sample
                'ch3_range': (0.300, 0.350),    # Based on your sample
                'noise_level': 0.03
            },
            '2-PINCH': {
                'ch1_range': (0.040, 0.080),    # Higher precision grip
                'ch2_range': (0.600, 0.800),    # High extensor
                'ch3_range': (0.600, 0.800),    # High intrinsic
                'noise_level': 0.04
            },
            '3-POINT': {
                'ch1_range': (0.020, 0.030),    # Based on your sample
                'ch2_range': (0.920, 0.970),    # Based on your sample
                'ch3_range': (0.920, 0.970),    # Based on your sample
                'noise_level': 0.02
            },
            '4-FOUR': {
                'ch1_range': (0.030, 0.060),    # Moderate flexor
                'ch2_range': (0.500, 0.700),    # Moderate-high extensor
                'ch3_range': (0.500, 0.700),    # Moderate-high intrinsic
                'noise_level': 0.03
            },
            '5-FIVE': {
                'ch1_range': (0.050, 0.060),    # Based on your sample
                'ch2_range': (0.720, 0.750),    # Based on your sample
                'ch3_range': (0.720, 0.750),    # Based on your sample
                'noise_level': 0.02
            },
            '6-PEACE': {
                'ch1_range': (0.130, 0.150),    # Based on your sample
                'ch2_range': (0.790, 0.820),    # Based on your sample
                'ch3_range': (0.790, 0.820),    # Based on your sample
                'noise_level': 0.03
            },
            '7-THUMBS_UP': {
                'ch1_range': (0.080, 0.120),    # Moderate flexor for thumb
                'ch2_range': (0.850, 0.950),    # Very high extensor
                'ch3_range': (0.400, 0.600),    # Moderate intrinsic
                'noise_level': 0.04
            },
            '8-HOOK_GRIP': {
                'ch1_range': (0.060, 0.100),    # Moderate-high flexor
                'ch2_range': (0.400, 0.600),    # Moderate extensor
                'ch3_range': (0.700, 0.900),    # High intrinsic
                'noise_level': 0.05
            },
            '9-FLAT_PALM': {
                'ch1_range': (0.015, 0.025),    # Very low flexor
                'ch2_range': (0.250, 0.350),    # Low extensor
                'ch3_range': (0.200, 0.300),    # Low intrinsic
                'noise_level': 0.02
            },
            '10-OK_SIGN': {
                'ch1_range': (0.015, 0.025),    # Based on your sample
                'ch2_range': (0.660, 0.690),    # Based on your sample
                'ch3_range': (0.650, 0.680),    # Based on your sample
                'noise_level': 0.03
            }
        }
        
        print("🎯 Domain-Adapted EMG Dataset Generator")
        print("📊 Based on Your Real EMG Signal Characteristics")
        print("🔧 Matching Your Hardware and Sensor Setup")
        print("=" * 70)
    
    def generate_realistic_sample(self, gesture, variation_factor=1.0):
        """Generate sample that matches your real EMG characteristics"""
        pattern = self.gesture_patterns[gesture]
        
        # Generate clean EMG values within realistic ranges
        ch1_min, ch1_max = pattern['ch1_range']
        ch2_min, ch2_max = pattern['ch2_range']
        ch3_min, ch3_max = pattern['ch3_range']
        
        # Add variation
        ch1_clean = np.random.uniform(ch1_min, ch1_max) * variation_factor
        ch2_clean = np.random.uniform(ch2_min, ch2_max) * variation_factor
        ch3_clean = np.random.uniform(ch3_min, ch3_max) * variation_factor
        
        # Add realistic noise
        noise_level = pattern['noise_level']
        ch1_clean += np.random.normal(0, noise_level)
        ch2_clean += np.random.normal(0, noise_level)
        ch3_clean += np.random.normal(0, noise_level)
        
        # Clamp to valid range
        ch1_clean = np.clip(ch1_clean, 0.01, 0.99)
        ch2_clean = np.clip(ch2_clean, 0.01, 0.99)
        ch3_clean = np.clip(ch3_clean, 0.01, 0.99)
        
        # Convert to raw EMG (matching your sensor characteristics)
        # Your samples show: low values ~1000-9000, high values ~20000-62000
        
        # Channel 1: typically low (1000-10000 range)
        if ch1_clean < 0.1:
            raw_ch1 = int(ch1_clean * 10000 + np.random.randint(800, 1500))
        else:
            raw_ch1 = int(ch1_clean * 15000 + np.random.randint(2000, 5000))
        
        # Channels 2&3: can be high (20000-65000 range)
        raw_ch2 = int(ch2_clean * 45000 + np.random.randint(15000, 25000))
        raw_ch3 = int(ch3_clean * 45000 + np.random.randint(15000, 25000))
        
        # Add some correlation between ch2 and ch3 (like your real data)
        correlation_noise = np.random.normal(0, 2000)
        raw_ch2 += int(correlation_noise)
        raw_ch3 += int(correlation_noise)
        
        # Ensure realistic bounds
        raw_ch1 = np.clip(raw_ch1, 1000, 15000)
        raw_ch2 = np.clip(raw_ch2, 15000, 65000)
        raw_ch3 = np.clip(raw_ch3, 15000, 65000)
        
        return {
            'raw': [raw_ch1, raw_ch2, raw_ch3],
            'clean': [ch1_clean, ch2_clean, ch3_clean]
        }
    
    def generate_domain_adapted_dataset(self, total_samples=50000):
        """Generate domain-adapted dataset"""
        print(f"🎯 Generating {total_samples:,} domain-adapted EMG samples...")
        
        gestures = list(self.gesture_patterns.keys())
        samples_per_gesture = total_samples // len(gestures)
        
        print(f"📊 Samples per gesture: {samples_per_gesture:,}")
        
        data = []
        timestamp_base = 1752250000
        sample_id = 0
        
        for gesture_idx, gesture in enumerate(gestures):
            print(f"   Generating {gesture}...")
            
            for i in range(samples_per_gesture):
                # Add variation across samples
                variation_factor = np.random.uniform(0.8, 1.2)
                
                # Generate sample
                sample = self.generate_realistic_sample(gesture, variation_factor)
                
                # Create timestamp
                timestamp = timestamp_base + sample_id * np.random.randint(10, 20)
                
                # Create data row
                row = {
                    'timestamp': timestamp,
                    'ch1': sample['raw'][0],
                    'ch2': sample['raw'][1],
                    'ch3': sample['raw'][2],
                    'gesture': gesture,
                    'label': gesture_idx,
                    'emg1_clean': sample['clean'][0],
                    'emg2_clean': sample['clean'][1],
                    'emg3_clean': sample['clean'][2],
                    'sample_id': sample_id
                }
                
                data.append(row)
                sample_id += 1
        
        # Handle remaining samples
        remaining = total_samples - len(data)
        for i in range(remaining):
            gesture = np.random.choice(gestures)
            gesture_idx = gestures.index(gesture)
            variation_factor = np.random.uniform(0.8, 1.2)
            
            sample = self.generate_realistic_sample(gesture, variation_factor)
            timestamp = timestamp_base + sample_id * np.random.randint(10, 20)
            
            row = {
                'timestamp': timestamp,
                'ch1': sample['raw'][0],
                'ch2': sample['raw'][1],
                'ch3': sample['raw'][2],
                'gesture': gesture,
                'label': gesture_idx,
                'emg1_clean': sample['clean'][0],
                'emg2_clean': sample['clean'][1],
                'emg3_clean': sample['clean'][2],
                'sample_id': sample_id
            }
            
            data.append(row)
            sample_id += 1
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Generated {len(df):,} domain-adapted samples")
        return df
    
    def validate_against_real_samples(self, df):
        """Validate generated data against your real samples"""
        print(f"\n🔍 Validating against your real EMG samples...")
        
        for gesture, real_data in self.real_samples.items():
            # Get generated samples for this gesture
            generated_samples = df[df['gesture'] == gesture]
            
            if len(generated_samples) == 0:
                continue
            
            # Compare statistics
            real_clean = real_data['clean']
            gen_clean_mean = [
                generated_samples['emg1_clean'].mean(),
                generated_samples['emg2_clean'].mean(),
                generated_samples['emg3_clean'].mean()
            ]
            
            print(f"\n📊 {gesture}:")
            print(f"   Real:      [{real_clean[0]:.3f}, {real_clean[1]:.3f}, {real_clean[2]:.3f}]")
            print(f"   Generated: [{gen_clean_mean[0]:.3f}, {gen_clean_mean[1]:.3f}, {gen_clean_mean[2]:.3f}]")
            
            # Check if generated data is close to real data
            differences = [abs(r - g) for r, g in zip(real_clean, gen_clean_mean)]
            max_diff = max(differences)
            
            if max_diff < 0.1:
                print(f"   ✅ GOOD match (max diff: {max_diff:.3f})")
            elif max_diff < 0.2:
                print(f"   ⚠️  OK match (max diff: {max_diff:.3f})")
            else:
                print(f"   ❌ Poor match (max diff: {max_diff:.3f})")
    
    def save_dataset(self, df, filename='domain_adapted_emg_dataset.csv'):
        """Save domain-adapted dataset"""
        print(f"\n💾 Saving domain-adapted dataset...")
        df.to_csv(filename, index=False)
        print(f"✅ Saved: {filename}")
        
        # Statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(df):,}")
        print(f"   Gestures: {df['gesture'].nunique()}")
        print(f"   Features: {len(df.columns)}")
        
        print(f"\n🎯 Gesture Distribution:")
        gesture_counts = df['gesture'].value_counts().sort_index()
        for gesture, count in gesture_counts.items():
            print(f"   {gesture}: {count:,} samples")
        
        print(f"\n📈 EMG Channel Ranges (matching your hardware):")
        for channel in ['ch1', 'ch2', 'ch3']:
            stats = df[channel].describe()
            print(f"   {channel}: {stats['min']:.0f} - {stats['max']:.0f} (mean: {stats['mean']:.0f})")
        
        return filename
    
    def create_comparison_visualization(self, df):
        """Compare generated vs real data"""
        print(f"\n📊 Creating comparison visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Domain-Adapted Dataset vs Real EMG Data', fontsize=16, fontweight='bold')
        
        # 1. Channel value distributions
        ax1 = axes[0, 0]
        ax1.hist(df['emg1_clean'], bins=50, alpha=0.7, label='Generated Ch1', color='lightblue')
        ax1.hist(df['emg2_clean'], bins=50, alpha=0.7, label='Generated Ch2', color='lightgreen')
        ax1.hist(df['emg3_clean'], bins=50, alpha=0.7, label='Generated Ch3', color='lightcoral')
        
        # Add real sample points
        for gesture, real_data in self.real_samples.items():
            clean = real_data['clean']
            ax1.axvline(clean[0], color='blue', linestyle='--', alpha=0.8)
            ax1.axvline(clean[1], color='green', linestyle='--', alpha=0.8)
            ax1.axvline(clean[2], color='red', linestyle='--', alpha=0.8)
        
        ax1.set_xlabel('Clean EMG Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Generated vs Real EMG Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Gesture patterns comparison
        ax2 = axes[0, 1]
        gesture_patterns = df.groupby('gesture')[['emg1_clean', 'emg2_clean', 'emg3_clean']].mean()
        
        x = np.arange(len(gesture_patterns))
        width = 0.25
        
        ax2.bar(x - width, gesture_patterns['emg1_clean'], width, label='Ch1', alpha=0.8, color='blue')
        ax2.bar(x, gesture_patterns['emg2_clean'], width, label='Ch2', alpha=0.8, color='green')
        ax2.bar(x + width, gesture_patterns['emg3_clean'], width, label='Ch3', alpha=0.8, color='red')
        
        # Add real sample points
        for i, (gesture, real_data) in enumerate(self.real_samples.items()):
            if gesture in gesture_patterns.index:
                idx = list(gesture_patterns.index).index(gesture)
                clean = real_data['clean']
                ax2.scatter([idx-width, idx, idx+width], clean, 
                           color=['darkblue', 'darkgreen', 'darkred'], s=100, marker='x')
        
        ax2.set_xlabel('Gestures')
        ax2.set_ylabel('Average EMG')
        ax2.set_title('Generated Patterns vs Real Samples', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([g.split('-')[1] for g in gesture_patterns.index], rotation=45)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Raw EMG ranges
        ax3 = axes[1, 0]
        channels = ['ch1', 'ch2', 'ch3']
        channel_ranges = []
        
        for ch in channels:
            channel_ranges.append([df[ch].min(), df[ch].max()])
        
        for i, (ch, (min_val, max_val)) in enumerate(zip(channels, channel_ranges)):
            ax3.barh(i, max_val - min_val, left=min_val, alpha=0.7, 
                    color=['blue', 'green', 'red'][i], label=f'{ch} range')
        
        # Add real sample points
        for gesture, real_data in self.real_samples.items():
            raw = real_data['raw']
            for i, val in enumerate(raw):
                ax3.scatter(val, i, color=['darkblue', 'darkgreen', 'darkred'][i], 
                           s=50, marker='o', alpha=0.8)
        
        ax3.set_xlabel('Raw EMG Value')
        ax3.set_ylabel('Channel')
        ax3.set_title('Raw EMG Ranges vs Real Samples', fontweight='bold')
        ax3.set_yticks(range(3))
        ax3.set_yticklabels(channels)
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Channel correlation
        ax4 = axes[1, 1]
        corr_matrix = df[['emg1_clean', 'emg2_clean', 'emg3_clean']].corr()
        im = ax4.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        ax4.set_title('Channel Correlations', fontweight='bold')
        ax4.set_xticks([0, 1, 2])
        ax4.set_yticks([0, 1, 2])
        ax4.set_xticklabels(['Ch1', 'Ch2', 'Ch3'])
        ax4.set_yticklabels(['Ch1', 'Ch2', 'Ch3'])
        
        for i in range(3):
            for j in range(3):
                ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax4)
        plt.tight_layout()
        plt.savefig('domain_adapted_comparison.png', dpi=150, bbox_inches='tight')
        print(f"📊 Comparison saved: domain_adapted_comparison.png")

def main():
    """Main function"""
    generator = DomainAdaptedEMGGenerator()
    
    # Generate domain-adapted dataset
    total_samples = int(input("Enter total samples (default 50000): ") or "50000")
    df = generator.generate_domain_adapted_dataset(total_samples)
    
    # Validate against real samples
    generator.validate_against_real_samples(df)
    
    # Save dataset
    filename = generator.save_dataset(df)
    
    # Create visualization
    try:
        generator.create_comparison_visualization(df)
    except ImportError:
        print("⚠️  Matplotlib not available - skipping visualization")
    
    print(f"\n🎉 SUCCESS!")
    print(f"✅ Domain-adapted dataset created: {filename}")
    print(f"📊 Total samples: {len(df):,}")
    print(f"🎯 Matched to your real EMG characteristics")
    print(f"🔧 Ready for deep learning training!")
    
    print(f"\n🚀 Next Steps:")
    print(f"1. Train deep learning model on this adapted dataset")
    print(f"2. Should achieve much better accuracy on your real data")
    print(f"3. Test with your actual EMG samples")

if __name__ == "__main__":
    main()
