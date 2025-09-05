# Motion AI: EMG Gesture Recognition System

Motion AI is a complete machine learning system for recognizing hand gestures from EMG (Electromyography) signals. It includes both a Python backend with FastAPI and a modern React frontend dashboard.

## 🌟 Complete Full-Stack Application

- **🖥️ Backend**: Python FastAPI with ML models, safety systems, and real-time processing
- **🌐 Frontend**: React TypeScript dashboard with real-time visualization
- **🔄 Real-time**: Live EMG signal processing and gesture recognition
- **🛡️ Safety**: 5-layer safety system with monitoring and alerts
- **📊 Analytics**: Performance metrics and interactive charts

## Features

- **Real-time EMG Processing**: Process EMG signals from multiple channels
- **Adaptive Thresholding**: Automatically adjust to changing baseline signals
- **Drift Detection**: Identify when signal characteristics change significantly
- **Fault Resilience**: Handle noise, dropouts, and signal degradation
- **Safety Layer**: Prevent unsafe commands with deadzone, hysteresis, and rate limiting
- **Metrics Tracking**: Monitor accuracy, latency, and safety interventions

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/emg_serial.git
cd emg_serial
```

2. Create and activate a virtual environment (recommended):

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the demo script with a sample CSV file:

```bash
python demo/run_motion_ai.py --csv demo/sample.csv
```

This will:
1. Load the sample EMG data
2. Process it through the Motion AI pipeline
3. Generate predictions, safety metrics, and adaptation logs
4. Create visualizations and a metrics card

## Advanced Usage

### Command-line Options

```bash
python demo/run_motion_ai.py --csv demo/sample.csv --model best_model/ar_random_forest_model.pkl --window_ms 200 --hop_ms 100 --confidence_threshold 0.6 --live_plot
```

Options:
- `--csv`: Path to CSV file(s) with EMG data
- `--model`: Path to trained model bundle
- `--window_ms`: Window size in milliseconds
- `--hop_ms`: Hop size in milliseconds
- `--confidence_threshold`: Minimum confidence for gesture recognition
- `--live_plot`: Enable real-time visualization
- `--inject_faults`: Enable fault injection for resilience testing

### Testing Fault Resilience

```bash
python demo/run_motion_ai.py --csv demo/sample.csv --inject_faults
```

## Data Schema

### Input CSV Format

The system expects CSV files with the following columns:

- `timestamp`: Time in milliseconds
- `ch1`, `ch2`, `ch3`: Raw EMG channel values
- `gesture` (optional): Ground truth labels for evaluation

Example:
```
timestamp,ch1,ch2,ch3,gesture
1000,2048,2048,2048,RELAX
1001,2052,2045,2050,RELAX
...
```

### API Schemas

#### SignalFrame
```python
class SignalFrame:
    timestamp: float  # Unix epoch seconds
    channels: List[float]  # EMG readings in fixed channel order
    metadata: Dict[str, Any]  # Free-form metadata
```

#### IntentFrame
```python
class IntentFrame:
    gesture: str  # e.g., "open_hand", "pinch", "unknown"
    confidence: float  # Value between 0 and 1
    features: Dict[str, Any]  # Transparent features for debugging
    design_candidates: Optional[List[Dict[str, Any]]]  # UI-only candidates
```

#### MotorCmd
```python
class MotorCmd:
    actuator_id: str  # Identifier for the actuator
    angle: Optional[float]  # Target angle in degrees
    force: Optional[float]  # Target force
    safety_flags: Dict[str, Any]  # Safety annotations
```

## Safety Notes

Motion AI implements several safety mechanisms to ensure reliable operation:

1. **Confidence Thresholding**: Gestures with low confidence are mapped to "rest"
2. **Deadzone**: Small movements are filtered out to prevent jitter
3. **Hysteresis**: Prevents rapid oscillation between states
4. **Rate Limiting**: Restricts how quickly actuators can move
5. **Drift Detection**: Identifies when signal characteristics change significantly
6. **Dropout Handling**: Forces safe state when signal quality degrades
7. **Adaptation**: Adjusts to changing baseline signals over time

### Safety Recommendations

- Always monitor the system during initial deployment
- Start with conservative safety settings and adjust as needed
- Regularly check adaptation logs for signal drift
- Implement emergency stop mechanisms in any physical system
- Test with fault injection to ensure resilience

## Metrics and Evaluation

The system generates a metrics card with key performance indicators:

- **Classification Accuracy**: How often the correct gesture is predicted
- **Latency**: Processing time per window (mean, median, p95, max)
- **Safety Coverage**: Count of safety interventions by type
- **Adaptation Events**: Baseline updates and drift detection events

## Project Structure

```
motion_ai/
  api/            # API endpoints and router
  classifiers/    # ML models for gesture recognition
  control/        # Safety layer and policy mapping
  eval/           # Offline evaluation tools
  features/       # Feature extraction from EMG signals
  preprocess/     # Signal preprocessing, adaptation, fault injection
  CONTRACT.md     # API contract and schema definitions
demo/
  run_motion_ai.py  # One-command demo script
  sample.csv        # Sample EMG data for testing
best_model/       # Trained model bundles
data/             # EMG datasets
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.