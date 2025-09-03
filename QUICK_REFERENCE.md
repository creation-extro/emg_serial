# Motion AI - Quick Reference Guide

## 🚀 Quick Start Commands

### Installation
```bash
git clone https://github.com/creation-extro/emg_serial.git
cd emg_serial
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Run Demo
```bash
# Basic demo
python demo/run_motion_ai.py --csv demo/sample.csv

# Interactive demo with dashboard
python demo/final_integrated_demo.py

# API server
uvicorn motion_ai.api.router:app --reload
```

## 📊 Models Summary

| Model Type | File Location | Purpose | Key Features |
|------------|--------------|---------|--------------|
| **MLP Classifier** | `motion_ai/classifiers/mlp_light.py` | Real-time gesture recognition | Single hidden layer, probability output |
| **SVM Baseline** | `motion_ai/classifiers/svm_baseline.py` | Baseline comparison | RBF kernel, feature scaling |
| **Deep Learning** | Training scripts in `test_files/` | Advanced recognition | CNN/RNN architectures |
| **Random Forest** | `test_files/train_ar_random_forest.py` | Autoregressive features | Ensemble method |

## 🔌 API Endpoints Summary

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|--------|--------|
| `/health` | GET | Health check | None | `{"status": "ok"}` |
| `/v1/classify` | POST | Gesture recognition | `SignalFrame` | `IntentFrame` |
| `/v1/policy` | POST | Command generation | `IntentFrame` | `MotorCmd[]` |
| `/v1/hybrid` | POST | End-to-end processing | `SignalFrame` | `MotorCmd[]` |

## 🛡️ Safety Features

| Feature | Purpose | Configuration |
|---------|---------|---------------|
| **Dead Zone** | Filter small movements | `dead_zone_angle_deg: 1.5` |
| **Rate Limiting** | Control movement speed | `max_angle_rate_deg_s: 90.0` |
| **Hysteresis** | Prevent oscillation | `hysteresis_high_deg: 2.0` |
| **Confidence Threshold** | Reject uncertain predictions | `confidence_threshold: 0.6` |
| **Drift Detection** | Monitor signal changes | `z_thresh: 2.0, req_consec: 5` |

## 📈 Data Flow

```
EMG Sensors → Preprocessing → Feature Extraction → Classification → Safety Layer → Actuator Commands
     ↓              ↓               ↓                ↓              ↓
   Raw Signals → Filtered → Feature Vector → Gesture+Confidence → Safe Commands
```

## 🎯 Gesture Classes

| Gesture | Description | Typical Use |
|---------|-------------|-------------|
| `rest` | Relaxed hand | Default/safe state |
| `open` | Open hand | Release/extend |
| `fist` | Closed fist | Grip/contract |
| `pinch` | Thumb-finger pinch | Precision grip |
| `point` | Index finger extended | Selection/pointing |
| `four` | Four fingers extended | Multi-finger actions |
| `five` | All fingers extended | Full hand open |
| `peace` | Peace sign | Specific gestures |

## 🔧 Configuration Examples

### Basic Model Training
```python
from motion_ai.classifiers.mlp_light import train_mlp

model = train_mlp(
    csv_path="data/emg_data.csv",
    window_ms=200,
    hop_ms=100,
    fs=1000.0,
    hidden_size=100,
    test_size=0.2
)
```

### Safety Configuration
```python
from motion_ai.control.safety_layer import SafetyConfig

config = SafetyConfig(
    max_angle_rate_deg_s=45.0,    # Slower movement
    dead_zone_angle_deg=2.0,      # Larger dead zone
    hysteresis_high_deg=3.0,      # Higher hysteresis
    confidence_threshold=0.8       # Higher confidence required
)
```

### API Usage
```python
import requests

# Classify gesture
response = requests.post("http://localhost:8000/v1/classify", json={
    "timestamp": 1732310400.0,
    "channels": [0.1, 0.2, 0.3],
    "metadata": {"model_path": "models/emg_model.pkl"}
})

gesture_result = response.json()
print(f"Gesture: {gesture_result['gesture']}")
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `demo/run_motion_ai.py` | Main demo script |
| `demo/final_integrated_demo.py` | Interactive dashboard demo |
| `motion_ai/api/router.py` | FastAPI endpoints |
| `motion_ai/control/safety_layer.py` | Safety mechanisms |
| `motion_ai/features/extractors.py` | Feature extraction |
| `motion_ai/preprocess/adaptation.py` | Signal adaptation |
| `motion_ai/CONTRACT.md` | API contracts |
| `requirements.txt` | Python dependencies |

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Model file not found | Check path in metadata, ensure model exists |
| Low accuracy | Increase training data, adjust features |
| High latency | Optimize feature extraction, reduce model complexity |
| Safety interventions | Adjust safety thresholds, check signal quality |
| Drift detection triggers | Recalibrate baseline, check sensor placement |

## 📊 Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| **Accuracy** | >85% | 87-92% |
| **Latency** | <20ms | 10-15ms |
| **Confidence** | >0.6 | 0.7-0.9 |
| **Safety Rate** | <5% | 2-3% |

## 🔄 Development Workflow

1. **Data Collection**: Gather EMG data with gesture labels
2. **Preprocessing**: Apply filters and normalization
3. **Feature Engineering**: Extract relevant features
4. **Model Training**: Train and validate classifier
5. **Safety Testing**: Validate safety mechanisms
6. **Integration**: Deploy through API endpoints
7. **Monitoring**: Track performance metrics

## 📝 Notes

- Always test with fault injection before deployment
- Monitor adaptation events for signal drift
- Regularly update safety thresholds based on usage
- Keep training data quality high for best performance
- Use live plotting for real-time debugging