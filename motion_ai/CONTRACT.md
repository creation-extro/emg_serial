# motion_ai Contracts — Day 1 (Contracts & Scaffolding)

This document defines the shared schemas and Day 1 API stubs to integrate preprocessing, feature extraction, classification, policy generation, and evaluation. The immediate goal is to publish a stable contract for integration with:
- Akash: backend router and JSON alignment
- Rutuja: dashboard/agent UI (DesignResponse candidates display only)

Repo skeleton (present):

```
motion_ai/
  api/
    router.py
  CONTRACT.md
  preprocess/           # to be implemented Day 2+
  features/             # to be implemented Day 2+
  classifiers/          # to be implemented Day 2+
  policy/               # to be implemented Day 2+
  eval/                 # to be implemented Day 2+
```

Note: `motion_ai/api/router.py` exposes runnable endpoints and an importable router.

Schemas (Pydantic, defined in motion_ai/api/router.py):

- SignalFrame
  - timestamp: float (Unix epoch seconds)
  - channels: number[] (EMG readings in fixed channel order from preprocessing)
  - metadata: object (free-form; examples: device_id, window_size_ms, subject_id, include_design_candidates, model_id, preprocessing_version)

- IntentFrame
  - gesture: string (e.g., "open_hand", "pinch", "unknown")
  - confidence: number in [0,1]
  - features: object (transparent features for debugging/explainability)
  - design_candidates?: object[] (UI-only; optional; not used in control loop)

- MotorCmd
  - actuator_id: string
  - angle?: number
  - force?: number
  - safety_flags: object

Day 1 Stubs (live):

- POST /v1/classify -> IntentFrame
  - Always returns:
    {
      "gesture": "unknown",
      "confidence": 0.0,
      "features": {"channels_len": <int>, "timestamp": <float>},
      "design_candidates": [] | null
    }
  - If `metadata.include_design_candidates == true`, returns empty array for `design_candidates` (UI plumbing for Rutuja). Otherwise null.

- POST /v1/policy -> MotorCmd[]
  - Always returns a single noop command:
    [
      {
        "actuator_id": "noop",
        "angle": null,
        "force": null,
        "safety_flags": {"reason": "stub", "from_gesture": "<gesture>"}
      }
    ]

Running the stub service:

- Install:
  pip install fastapi uvicorn pydantic

- Run:
  uvicorn motion_ai.api.router:app --reload

- Health:
  curl http://127.0.0.1:8000/health
  -> {"status":"ok"}

- Classify example:
  curl -X POST http://127.0.0.1:8000/v1/classify \
    -H "Content-Type: application/json" \
    -d '{
      "timestamp": 1732310400.0,
      "channels": [0.1, 0.2, 0.3],
      "metadata": {"include_design_candidates": true, "model_id": "deep_learning_emg_model.h5", "preproc": "deep_learning_emg_model_preprocessing.pkl"}
    }'

- Policy example:
  curl -X POST http://127.0.0.1:8000/v1/policy \
    -H "Content-Type: application/json" \
    -d '{
      "gesture": "unknown",
      "confidence": 0.0,
      "features": {"channels_len": 3, "timestamp": 1732310400.0}
    }'

Acceptance tests (Day 1):

1. Health returns valid JSON:
   GET /health -> {"status": "ok"}

2. Classifier returns contract-shaped IntentFrame:
   - gesture == "unknown"
   - confidence == 0.0
   - features has keys ["channels_len", "timestamp"]
   - design_candidates is [] if include_design_candidates==true; otherwise null/omitted

3. Policy returns noop MotorCmd list with 1 element:
   - actuator_id == "noop"
   - angle == null; force == null
   - safety_flags contains keys ["reason", "from_gesture"]

Handoffs:

- Akash
  - Confirm the above JSON payloads match the backend expectations.
  - Import path for router: motion_ai.api.router:router (for inclusion into a larger FastAPI app) or use the app at motion_ai.api.router:app directly.

- Rutuja
  - `IntentFrame.design_candidates` is optional and for dashboard display only. Not part of the control loop. You can render these candidates in the same agent dashboard used for the unified demo.

Notes for Day 2+ (non-blocking for Day 1):

- Preprocessing: ensure the channel order and normalization used during training (deep_learning_emg_model.h5 + deep_learning_emg_model_preprocessing.pkl) are reflected in `SignalFrame.channels` construction.
- Features: optionally cache feature values in `IntentFrame.features` for traceability.
- Classifier/Policy: swap stubs with real implementations behind the same endpoints to preserve contracts.
