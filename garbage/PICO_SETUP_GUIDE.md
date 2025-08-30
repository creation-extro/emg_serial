# Raspberry Pi Pico EMG Setup Guide

## 🥧 **Raspberry Pi Pico Specifications**

### **ADC Pins Available:**
- **GPIO 26** (ADC0) - EMG Sensor 1
- **GPIO 27** (ADC1) - EMG Sensor 2  
- **GPIO 28** (ADC2) - EMG Sensor 3
- **GPIO 29** (ADC3) - Available for 4th sensor
- **GPIO 25** - Onboard LED (status indicator)

### **Technical Specs:**
- **ADC Resolution**: 16-bit (0-65535)
- **Reference Voltage**: 3.3V
- **Max Sample Rate**: ~500 kHz (we'll use 100 Hz)
- **USB Serial**: Built-in via micro USB

## 🔌 **Hardware Connections**

```
Raspberry Pi Pico Pinout:
┌─────────────────────────┐
│  ┌─┐                    │
│  │U│  Raspberry Pi Pico │
│  │S│                    │
│  │B│                    │
│  └─┘                    │
│                         │
│ GPIO 26 (ADC0) ●────────┼── EMG Sensor 1 (OUT)
│ GPIO 27 (ADC1) ●────────┼── EMG Sensor 2 (OUT)  
│ GPIO 28 (ADC2) ●────────┼── EMG Sensor 3 (OUT)
│                         │
│ 3V3(OUT)       ●────────┼── EMG Sensors (VCC)
│ GND            ●────────┼── EMG Sensors (GND)
│                         │
│ GPIO 25        ●────────┼── Onboard LED
└─────────────────────────┘
```

### **EMG Sensor Connections:**
```
EMG Sensor 1:
  VCC → 3V3(OUT) [Pin 36]
  GND → GND [Pin 38] 
  OUT → GPIO 26 [Pin 31]

EMG Sensor 2:
  VCC → 3V3(OUT) [Pin 36]
  GND → GND [Pin 33]
  OUT → GPIO 27 [Pin 32]

EMG Sensor 3:
  VCC → 3V3(OUT) [Pin 36]
  GND → GND [Pin 28]
  OUT → GPIO 28 [Pin 34]
```

## 🚀 **Quick Setup Steps**

### **Step 1: Install MicroPython on Pico**
1. Download MicroPython UF2 file for Pico
2. Hold BOOTSEL button while connecting USB
3. Drag UF2 file to RPI-RP2 drive
4. Pico will reboot with MicroPython

### **Step 2: Setup Thonny IDE**
1. Install Thonny IDE
2. Go to **Tools → Options → Interpreter**
3. Select **MicroPython (Raspberry Pi Pico)**
4. Choose correct COM port

### **Step 3: Upload Pico Code**
1. Open `emg_sender_pico.py` in Thonny
2. Click **Run** or press F5
3. Code will execute on Pico

### **Step 4: Test on PC**
```bash
python test_serial_communication.py
```

## 📊 **Expected Output**

### **In Thonny Console (Pico):**
```
============================================================
EMG DATA SENDER - RASPBERRY PI PICO
============================================================
REAL SENSOR MODE SELECTED
Make sure EMG sensors are connected to GPIO 26, 27, 28

EMG Sender initialized successfully!
ADC pins: [26, 27, 28]
Sample rate: 100 Hz
ADC resolution: 16-bit (0-65535)

=== SENSOR CALIBRATION ===
Calibrating for 5 seconds...
IMPORTANT: Keep muscles completely RELAXED!
Starting in 3...
Starting in 2...
Starting in 1...
Calibrating now...
Calibration progress: 20.0%
Calibration progress: 40.0%
Calibration progress: 60.0%
Calibration progress: 80.0%
=== CALIBRATION COMPLETE ===
Baseline values: [32456, 33123, 31987]

==================================================
STARTING EMG DATA TRANSMISSION
==================================================
1234 2345 3456
1245 2356 3467
1256 2367 3478
# Status: 100 samples sent, Rate: 99.8 Hz
1267 2378 3489
...
```

### **In PC Serial Receiver:**
```bash
python serial_send.py

✓ Connected to COM4 at 9600 baud
✓ Model loaded: best_emg_model_random_forest.pkl
✓ CSV logging to: emg_data_1234567890.csv

EMG DATA COLLECTION & GESTURE RECOGNITION
========================================
Received: 1234 2345 3456
  ✓ Parsed EMG: CH1=1234, CH2=2345, CH3=3456
🎯 Prediction #1: OPEN (Confidence: 0.892)
📊 Samples: 150, Rate: 98.5 Hz, Buffer: 150
🎯 Prediction #2: CLOSE (Confidence: 0.945)
```

## 🎯 **About the Model**

### **What is the Model?**
The model is a **trained machine learning algorithm** that recognizes gestures:

```
Your Data → Training → Model → Predictions
52,322 samples → ML Algorithm → .pkl file → "OPEN" gesture
```

### **Model Creation Process:**
1. **Run Jupyter Notebook**: `EMG_Gesture_Recognition.ipynb`
2. **Training**: Uses your `combined_emg_data (1).csv`
3. **Feature Extraction**: 75+ features per window
4. **Algorithm Selection**: Tests Random Forest, SVM, etc.
5. **Model Saving**: Creates `best_emg_model.pkl`

### **Model Performance:**
- **Accuracy**: 85-95% (how often it's correct)
- **Confidence**: 0.0-1.0 (how sure it is)
- **Real-time**: Predicts every 150 samples (~1.5 seconds)

## 🔧 **Pico-Specific Features**

### **Code Highlights:**
```python
# Pico ADC reading (16-bit)
raw_value = adc.read_u16()  # Returns 0-65535

# LED status indication
self.led = Pin(25, Pin.OUT)  # Onboard LED
self.led.on()  # Solid = ready
self.led.toggle()  # Blink = active
```

### **Calibration Process:**
1. **Relax muscles** for 5 seconds
2. **Pico measures baseline** EMG levels
3. **LED blinks** during calibration
4. **LED solid** when ready

### **Data Processing:**
```python
# Scale 16-bit ADC to manageable range
processed = int(abs(amplified) / 16) + 1000
processed = max(500, min(4000, processed))
```

## 🧪 **Testing Modes**

### **1. Simulation Mode** (No Hardware Needed)
```python
# In emg_sender_pico.py, change:
USE_REAL_SENSORS = False
```
Generates fake EMG data for testing.

### **2. Sensor Test Mode**
```python
emg_sender.test_sensors(duration=10)
```
Shows raw and processed values in real-time.

### **3. Real Sensor Mode**
```python
USE_REAL_SENSORS = True
```
Uses actual EMG sensors connected to Pico.

## 🔍 **Troubleshooting**

### **Common Issues:**

#### **"No ADC channels initialized"**
- Check GPIO pins 26, 27, 28 are connected
- Verify EMG sensors have power (3.3V)

#### **"Thonny can't connect to Pico"**
- Press BOOTSEL + Reset to enter bootloader
- Re-flash MicroPython UF2 file
- Check USB cable (data, not just power)

#### **"No data on PC"**
- Verify COM port in Device Manager
- Check baud rate (9600)
- Make sure Pico code is running

#### **"Erratic EMG readings"**
- Check sensor connections
- Ensure good skin contact
- Run calibration again

### **LED Status Indicators:**
- **Off**: Not initialized
- **Blinking**: Calibrating or transmitting
- **Solid**: Ready/calibrated
- **Fast blink**: Error or high activity

## 📈 **Performance Optimization**

### **For Pico:**
```python
# Increase sample rate
self.sample_rate = 200  # Up to 500 Hz possible

# Reduce processing
processed = raw_val // 16 + 1000  # Faster than float math
```

### **For Better Accuracy:**
1. **Good sensor placement** on target muscles
2. **Stable connections** (solder if possible)
3. **Proper calibration** before each session
4. **Consistent gestures** during training

## 🎉 **Success Checklist**

- [ ] Pico has MicroPython installed
- [ ] EMG sensors connected to GPIO 26, 27, 28
- [ ] Thonny can connect to Pico
- [ ] Calibration completes successfully
- [ ] Data appears in Thonny console
- [ ] PC receives data via serial
- [ ] Model makes predictions (if trained)

## 📞 **Next Steps**

1. **Test the Pico code** in simulation mode first
2. **Connect real EMG sensors** to GPIO pins
3. **Run calibration** and data transmission
4. **Train your model** using the Jupyter notebook
5. **Start real-time gesture recognition**!

Your Raspberry Pi Pico is now ready for EMG gesture recognition! 🚀
