# EMG Gesture Recognition System

This project creates a machine learning model to predict hand gestures from EMG (Electromyography) signals using your combined dataset.

## Dataset Overview
- **Total samples**: 52,322 EMG recordings
- **Channels**: 3 EMG sensors (ch1, ch2, ch3)
- **Gestures**: 12 different hand gestures:
  - RELAX (26,138 samples)
  - 0-OPEN (2,394 samples)
  - 1-CLOSE (2,394 samples)
  - 2-PINCH (2,392 samples)
  - 3-POINT (2,388 samples)
  - 4-FOUR (2,381 samples)
  - 5-FIVE (2,376 samples)
  - 6-PEACE (2,370 samples)
  - 7-THUMBS_UP (2,364 samples)
  - 8-HOOK_GRIP (2,366 samples)
  - 9-FLAT_PALM (2,374 samples)
  - 10-OK_SIGN (2,385 samples)

## Files Description

### 1. `EMG_Gesture_Recognition.ipynb`
**Main Jupyter notebook** containing the complete machine learning pipeline:
- Data loading and exploration
- Feature extraction (time and frequency domain)
- Model training and comparison
- Performance evaluation
- Model saving

### 2. `real_time_prediction.py`
**Real-time prediction script** for using the trained model:
- Load trained model
- Predict gestures from serial EMG data
- Test predictions on CSV data
- Real-time gesture recognition

### 3. `data/combined_emg_data (1).csv`
Your EMG dataset with 3 channels and gesture labels.

## Getting Started

### Step 1: Install Dependencies
Make sure you're in the virtual environment and install required packages:
```bash
# Activate virtual environment (if not already active)
venv\Scripts\activate

# Install packages (already done)
pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib jupyter notebook ipykernel
```

### Step 2: Train the Model
Open and run the Jupyter notebook:
```bash
jupyter notebook EMG_Gesture_Recognition.ipynb
```

**Or start Jupyter Lab:**
```bash
jupyter lab
```

Then:
1. Open `EMG_Gesture_Recognition.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Load your data
   - Extract features from sliding windows
   - Train multiple models (Random Forest, SVM, Neural Network, etc.)
   - Compare performance and select the best model
   - Save the best model as a `.pkl` file

### Step 3: Use the Trained Model
After training, you can use the model for real-time prediction:

```bash
python real_time_prediction.py
```

Choose from:
1. **Real-time from serial port**: Connect to your EMG device
2. **Test on CSV data**: Test predictions on existing data
3. **Exit**

## Model Features

### Feature Extraction
The model extracts comprehensive features from EMG signals:

**Time Domain Features (per channel):**
- Mean, Standard deviation, Variance
- RMS (Root Mean Square)
- Min, Max, Range
- Skewness, Kurtosis
- Mean Absolute Deviation
- Percentiles (25th, 75th, IQR)
- Zero Crossing Rate
- Slope Sign Changes
- Waveform Length
- Average Amplitude Change

**Frequency Domain Features (per channel):**
- Mean Frequency
- Median Frequency
- Band Power (Low: 20-80Hz, Mid: 80-150Hz, High: 150-250Hz)
- Peak Frequency

**Cross-Channel Features:**
- Correlation between channels (ch1-ch2, ch1-ch3, ch2-ch3)

### Model Architecture
- **Window Size**: 150 samples
- **Overlap**: 70%
- **Total Features**: ~75 features per window
- **Models Tested**: Random Forest, Gradient Boosting, SVM, Neural Network

## Expected Performance
Based on the comprehensive feature set and dataset size, you can expect:
- **Accuracy**: 85-95% (depending on gesture complexity)
- **Best Model**: Likely Random Forest or Gradient Boosting
- **Real-time Capability**: Yes, with proper feature extraction

## Usage Examples

### Training a Model
```python
# In Jupyter notebook
classifier = EMGGestureClassifier()
data = classifier.load_data('data/combined_emg_data (1).csv')
features, labels = classifier.preprocess_data(data)
classifier.train_model(features, labels, model_type='random_forest')
classifier.save_model('my_emg_model.pkl')
```

### Real-time Prediction
```python
# Using the real-time script
predictor = RealTimeGesturePredictor('best_emg_model_random_forest.pkl')

# For serial data
predictor.predict_from_serial(com_port='COM4')

# For CSV testing
predictor.predict_from_csv('data/combined_emg_data (1).csv')
```

## Troubleshooting

### Common Issues:
1. **Import errors**: Make sure all packages are installed in the virtual environment
2. **Serial port issues**: Check COM port number and baud rate
3. **Memory issues**: Reduce window size or overlap if needed
4. **Low accuracy**: Try different models or adjust feature extraction parameters

### Tips for Better Performance:
1. **Data Quality**: Ensure EMG signals are properly filtered
2. **Consistent Gestures**: Make sure gesture labels are accurate
3. **Feature Selection**: Use feature importance to select best features
4. **Hyperparameter Tuning**: Adjust model parameters for better performance

## Next Steps

1. **Run the Jupyter notebook** to train your first model
2. **Experiment with different models** and parameters
3. **Test real-time prediction** with your EMG device
4. **Fine-tune features** based on model performance
5. **Deploy the model** for your specific application

## Support

If you encounter any issues:
1. Check that all dependencies are installed
2. Verify your data file path and format
3. Ensure your virtual environment is activated
4. Review the error messages in the notebook output

Good luck with your EMG gesture recognition project! 🚀
