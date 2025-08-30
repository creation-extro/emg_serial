#!/usr/bin/env python3
"""
Convert exg_data_tej4.py to CSV and combine all exg_data_tej files
"""

import pandas as pd
import os
import shutil

def convert_and_combine_exg_data():
    """Convert exg_data_tej4.py to CSV and combine all exg_data_tej files"""
    
    print("🔄 Converting and Combining EXG Data Files")
    print("=" * 50)
    
    # Step 1: Convert exg_data_tej4.py to exg_data_tej4.csv
    py_file = 'data/exg_data_tej4.py'
    csv_file = 'data/exg_data_tej4.csv'
    
    if os.path.exists(py_file):
        print(f"📂 Converting {py_file} to {csv_file}")
        shutil.copy2(py_file, csv_file)
        print(f"✅ Converted: {py_file} → {csv_file}")
    else:
        print(f"❌ File not found: {py_file}")
        return
    
    # Step 2: Find all exg_data_tej files
    data_dir = 'data'
    exg_files = []
    
    for filename in os.listdir(data_dir):
        if filename.startswith('exg_data_tej') and filename.endswith('.csv'):
            exg_files.append(os.path.join(data_dir, filename))
    
    exg_files.sort()  # Sort for consistent order
    
    print(f"\n📋 Found EXG data files:")
    for i, file in enumerate(exg_files, 1):
        print(f"   {i}. {file}")
    
    if not exg_files:
        print("❌ No exg_data_tej*.csv files found!")
        return
    
    # Step 3: Load and combine all files
    print(f"\n🔄 Loading and combining {len(exg_files)} files...")
    
    combined_data = []
    total_samples = 0
    
    for i, file_path in enumerate(exg_files, 1):
        print(f"   📂 Loading {file_path}...")
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Add source file information
            df['source_file'] = os.path.basename(file_path)
            df['file_id'] = i
            
            # Add to combined data
            combined_data.append(df)
            
            print(f"   ✅ Loaded {len(df):,} samples from {os.path.basename(file_path)}")
            total_samples += len(df)
            
        except Exception as e:
            print(f"   ❌ Error loading {file_path}: {e}")
            continue
    
    if not combined_data:
        print("❌ No data loaded successfully!")
        return
    
    # Step 4: Combine all dataframes
    print(f"\n🔗 Combining all data...")
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    print(f"✅ Combined dataset created:")
    print(f"   📊 Total samples: {len(combined_df):,}")
    print(f"   📊 Total columns: {len(combined_df.columns)}")
    print(f"   📊 Source files: {len(exg_files)}")
    
    # Step 5: Show data summary
    print(f"\n📊 Data Summary:")
    print(f"   Columns: {list(combined_df.columns)}")
    
    # Show label distribution
    if 'label' in combined_df.columns:
        print(f"\n🏷️ Label Distribution:")
        label_counts = combined_df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            percentage = count / len(combined_df) * 100
            print(f"   Label {label}: {count:,} samples ({percentage:.1f}%)")
    
    # Show source file distribution
    if 'source_file' in combined_df.columns:
        print(f"\n📁 Source File Distribution:")
        file_counts = combined_df['source_file'].value_counts()
        for file, count in file_counts.items():
            percentage = count / len(combined_df) * 100
            print(f"   {file}: {count:,} samples ({percentage:.1f}%)")
    
    # Step 6: Save combined dataset
    output_file = 'combined_exg_data.csv'
    print(f"\n💾 Saving combined dataset to {output_file}...")
    
    try:
        combined_df.to_csv(output_file, index=False)
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
        
        print(f"✅ Combined dataset saved successfully!")
        print(f"   📁 File: {output_file}")
        print(f"   📊 Size: {file_size:.1f} MB")
        print(f"   📊 Samples: {len(combined_df):,}")
        print(f"   📊 Features: {len(combined_df.columns)}")
        
        # Show first few rows
        print(f"\n👀 First 5 rows of combined dataset:")
        print(combined_df.head().to_string(index=False))
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error saving combined dataset: {e}")
        return None

def analyze_combined_data(filename='combined_exg_data.csv'):
    """Analyze the combined dataset"""
    
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return
    
    print(f"\n🔍 Analyzing Combined Dataset: {filename}")
    print("=" * 50)
    
    try:
        df = pd.read_csv(filename)
        
        print(f"📊 Dataset Overview:")
        print(f"   Shape: {df.shape}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        print(f"\n📈 Statistical Summary:")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())
        
        print(f"\n🔍 Data Quality Check:")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        print(f"   Duplicate rows: {df.duplicated().sum()}")
        
        # Check data ranges
        if 'raw_emg1' in df.columns:
            print(f"\n📊 EMG Data Ranges:")
            print(f"   raw_emg1: {df['raw_emg1'].min():,} - {df['raw_emg1'].max():,}")
            print(f"   raw_emg2: {df['raw_emg2'].min():,} - {df['raw_emg2'].max():,}")
            print(f"   normalized_emg1: {df['normalized_emg1'].min():.3f} - {df['normalized_emg1'].max():.3f}")
            print(f"   normalized_emg2: {df['normalized_emg2'].min():.3f} - {df['normalized_emg2'].max():.3f}")
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")

def main():
    """Main function"""
    
    # Convert and combine files
    output_file = convert_and_combine_exg_data()
    
    if output_file:
        # Analyze the combined dataset
        analyze_combined_data(output_file)
        
        print(f"\n🎉 Process completed successfully!")
        print(f"📁 Combined dataset: {output_file}")
        print(f"🚀 Ready for analysis and model training!")
    else:
        print(f"\n❌ Process failed!")

if __name__ == "__main__":
    main()
