# EMG Serial Communication Setup Guide

This guide helps you set up serial communication between your microcontroller (sending EMG data) and your PC (receiving data for gesture recognition).

## 📁 Files Overview

### PC Side (Python):
- **`serial_send.py`** - Enhanced receiver with real-time gesture prediction
- **`test_serial_communication.py`** - Test and debug serial communication
- **`real_time_prediction.py`** - Real-time gesture prediction module

### Microcontroller Side:
- **`emg_sender_micropython.py`** - MicroPython code for ESP32/PyBoard
- **`emg_sender_arduino.ino`** - Arduino/ESP32 C++ code

## 🔧 Hardware Setup

### Required Components:
1. **Microcontroller**: ESP32, Arduino Uno, or similar
2. **EMG Sensors**: 3 EMG sensors (e.g., MyoWare, Grove EMG)
3. **Connections**:
   - EMG Sensor 1 → A0 (Arduino) or GPIO 32 (ESP32)
   - EMG Sensor 2 → A1 (Arduino) or GPIO 33 (ESP32)
   - EMG Sensor 3 → A2 (Arduino) or GPIO 34 (ESP32)
   - Power and ground connections

### Wiring Diagram:
```
EMG Sensor 1:
  VCC → 3.3V/5V
  GND → GND
  OUT → A0/GPIO32

EMG Sensor 2:
  VCC → 3.3V/5V
  GND → GND
  OUT → A1/GPIO33

EMG Sensor 3:
  VCC → 3.3V/5V
  GND → GND
  OUT → A2/GPIO34
```

## 🚀 Quick Start

### Step 1: Choose Your Platform

#### Option A: MicroPython (ESP32/PyBoard)
1. Install MicroPython on your ESP32
2. Open Thonny IDE
3. Copy `emg_sender_micropython.py` to your device
4. Run the script

#### Option B: Arduino IDE (Arduino/ESP32)
1. Open Arduino IDE
2. Load `emg_sender_arduino.ino`
3. Select your board and COM port
4. Upload the code

### Step 2: Test Serial Communication
```bash
# Test if communication works
python test_serial_communication.py
```

### Step 3: Start Data Collection
```bash
# Collect data and perform real-time prediction
python serial_send.py
```

## 📊 Data Format

### Microcontroller → PC:
```
1234 2345 3456
1245 2356 3467
1256 2367 3478
...
```
Format: `EMG1 EMG2 EMG3` (space-separated integers)

### Alternative CSV Format:
```
1234,2345,3456
1245,2356,3467
1256,2367,3478
...
```

## ⚙️ Configuration

### Microcontroller Settings:
```python
# MicroPython
ADC_PINS = [32, 33, 34]  # GPIO pins for EMG sensors
UART_ID = 0              # 0 for USB serial
BAUD_RATE = 9600         # Serial baud rate
SAMPLE_RATE = 100        # Hz
```

```cpp
// Arduino
const int emgPins[3] = {A0, A1, A2};  // Analog pins
#define BAUD_RATE 9600
#define SAMPLE_RATE 100
```

### PC Settings:
```python
# serial_send.py
COM_PORT = 'COM4'        # Adjust for your system
BAUD_RATE = 9600
BUFFER_SIZE = 150        # Window size for prediction
```

## 🔍 Testing & Debugging

### 1. Test Serial Connection
```bash
python test_serial_communication.py
```
This will:
- List available COM ports
- Test data reception
- Validate data format
- Show success rate

### 2. Check Data Quality
Look for:
- ✅ Consistent data rate (~100 Hz)
- ✅ Valid 3-channel format
- ✅ Reasonable EMG values (500-4000 range)
- ✅ No communication errors

### 3. Common Issues & Solutions

#### "Serial port not found"
- Check COM port number in Device Manager (Windows)
- Try different USB cable
- Restart microcontroller

#### "No data received"
- Verify baud rate matches (9600)
- Check if another program is using the port
- Test with Arduino Serial Monitor first

#### "Malformed data"
- Check data format (space or comma separated)
- Verify 3 values per line
- Look for extra characters or noise

#### "Low prediction accuracy"
- Ensure model is trained first
- Check EMG sensor placement
- Calibrate sensors properly

## 📈 Usage Examples

### Basic Data Collection:
```bash
python serial_send.py
# Enter COM port: COM4
# Duration: 60 seconds
# Enable prediction: y
```

### Real-time Gesture Recognition:
```python
from real_time_prediction import RealTimeGesturePredictor

predictor = RealTimeGesturePredictor('best_emg_model.pkl')
predictor.predict_from_serial(com_port='COM4')
```

### Simulation Mode (No Hardware):
```bash
python test_serial_communication.py
# Choose option 3: Run EMG data simulator
```

## 🎯 Performance Optimization

### For Better Accuracy:
1. **Proper Sensor Placement**: Place EMG sensors on target muscles
2. **Calibration**: Run calibration before each session
3. **Consistent Gestures**: Perform gestures consistently during training
4. **Clean Signals**: Minimize electrical noise and movement artifacts

### For Higher Speed:
1. **Increase Sample Rate**: Up to 200-500 Hz if needed
2. **Reduce Buffer Size**: Smaller windows for faster response
3. **Optimize Features**: Use only most important features

## 🔧 Advanced Configuration

### Custom Data Processing:
```python
# In serial_send.py, modify parse_emg_data()
def parse_emg_data(self, line):
    # Add custom filtering, scaling, etc.
    values = line.split()
    emg1, emg2, emg3 = map(int, values)
    
    # Apply custom processing
    emg1 = self.apply_filter(emg1)
    emg2 = self.apply_filter(emg2)
    emg3 = self.apply_filter(emg3)
    
    return emg1, emg2, emg3
```

### Multiple Gesture Sessions:
```python
# Collect data for specific gestures
receiver = EMGSerialReceiver()
receiver.collect_gesture_data(gesture_name="PINCH", duration=30)
```

## 📋 Troubleshooting Checklist

- [ ] Hardware connections are secure
- [ ] Correct COM port selected
- [ ] Matching baud rates (9600)
- [ ] EMG sensors are powered
- [ ] No other programs using serial port
- [ ] Python packages installed (serial, numpy, etc.)
- [ ] Model file exists for prediction mode

## 🎉 Success Indicators

When everything works correctly, you should see:
```
✓ Connected to COM4 at 9600 baud
✓ CSV logging to: emg_data_1234567890.csv
✓ Model loaded: best_emg_model_random_forest.pkl

EMG DATA COLLECTION & GESTURE RECOGNITION
========================================
🎯 Prediction #1: OPEN (Confidence: 0.892)
📊 Samples: 150, Rate: 98.5 Hz, Buffer: 150
🎯 Prediction #2: CLOSE (Confidence: 0.945)
```

## 📞 Support

If you encounter issues:
1. Check this troubleshooting guide
2. Test with the simulation mode first
3. Verify hardware connections
4. Check serial monitor output from microcontroller

Good luck with your EMG gesture recognition system! 🚀
