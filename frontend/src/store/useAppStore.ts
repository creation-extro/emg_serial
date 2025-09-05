import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

export interface EMGData {
  timestamp: number;
  channels: number[];
  gesture?: string;
  confidence?: number;
  features?: { [key: string]: number };
}

export interface SystemMetrics {
  accuracy: number;
  latency: number;
  safetyEvents: number;
  totalPredictions: number;
  processedSamples: number;
  averageConfidence: number;
  gestureDistribution: { [gesture: string]: number };
  featureStats: { [feature: string]: { mean: number; std: number; min: number; max: number } };
}

export interface TrainingSession {
  id: string;
  name: string;
  startTime: Date;
  endTime?: Date;
  gestureLabel: string;
  samplesCollected: number;
  status: 'active' | 'paused' | 'completed';
}

export interface ModelInfo {
  name: string;
  type: 'svm' | 'mlp' | 'random_forest' | 'deep_learning';
  accuracy: number;
  isActive: boolean;
  lastTrained: Date;
  parameters: { [key: string]: any };
}

export interface AppState {
  // Connection & Status
  isConnected: boolean;
  isProcessing: boolean;
  currentView: 'dashboard' | 'training' | 'analytics' | 'settings' | '3d-viz';
  lastUpdate: Date;
  errorMessage?: string;

  // EMG Data
  emgData: EMGData[];
  currentGesture: string;
  gestureConfidence: number;
  realtimeFeatures: { [key: string]: number };

  // System Metrics
  metrics: SystemMetrics;
  
  // Models
  availableModels: ModelInfo[];
  activeModel: string;

  // Training
  trainingSession?: TrainingSession;
  trainingData: EMGData[];
  
  // Settings
  settings: {
    samplingRate: number;
    windowSize: number;
    hopSize: number;
    confidenceThreshold: number;
    maxDataPoints: number;
    enableFeatureViz: boolean;
    enable3DViz: boolean;
    autoSave: boolean;
  };

  // WebSocket
  wsConnected: boolean;
  wsLatency: number;
}

export interface AppActions {
  // Connection actions
  setConnectionStatus: (connected: boolean) => void;
  setProcessingStatus: (processing: boolean) => void;
  setView: (view: AppState['currentView']) => void;
  setError: (error?: string) => void;

  // EMG Data actions
  addEMGData: (data: EMGData) => void;
  clearEMGData: () => void;
  setCurrentGesture: (gesture: string, confidence: number) => void;
  updateRealtimeFeatures: (features: { [key: string]: number }) => void;

  // Metrics actions
  updateMetrics: (updates: Partial<SystemMetrics>) => void;
  incrementPredictions: () => void;
  recordLatency: (latency: number) => void;
  recordSafetyEvent: () => void;

  // Model actions
  setActiveModel: (modelName: string) => void;
  updateModelInfo: (modelName: string, updates: Partial<ModelInfo>) => void;
  addModel: (model: ModelInfo) => void;

  // Training actions
  startTrainingSession: (gestureLabel: string, sessionName?: string) => void;
  pauseTrainingSession: () => void;
  resumeTrainingSession: () => void;
  endTrainingSession: () => void;
  addTrainingData: (data: EMGData) => void;
  clearTrainingData: () => void;

  // Settings actions
  updateSettings: (updates: Partial<AppState['settings']>) => void;

  // WebSocket actions
  setWebSocketStatus: (connected: boolean, latency?: number) => void;
}

export type AppStore = AppState & AppActions;

const initialState: AppState = {
  isConnected: false,
  isProcessing: false,
  currentView: 'dashboard',
  lastUpdate: new Date(),
  
  emgData: [],
  currentGesture: 'rest',
  gestureConfidence: 0,
  realtimeFeatures: {},
  
  metrics: {
    accuracy: 0,
    latency: 0,
    safetyEvents: 0,
    totalPredictions: 0,
    processedSamples: 0,
    averageConfidence: 0,
    gestureDistribution: {},
    featureStats: {},
  },
  
  availableModels: [
    {
      name: 'SVM Baseline',
      type: 'svm',
      accuracy: 0.85,
      isActive: true,
      lastTrained: new Date(),
      parameters: { kernel: 'rbf', C: 1.0 }
    },
    {
      name: 'MLP Light',
      type: 'mlp',
      accuracy: 0.78,
      isActive: false,
      lastTrained: new Date(),
      parameters: { hidden_layers: [64, 32], activation: 'relu' }
    },
    {
      name: 'Random Forest',
      type: 'random_forest',
      accuracy: 0.82,
      isActive: false,
      lastTrained: new Date(),
      parameters: { n_estimators: 100, max_depth: 10 }
    }
  ],
  activeModel: 'SVM Baseline',
  
  trainingData: [],
  
  settings: {
    samplingRate: 1000,
    windowSize: 200,
    hopSize: 100,
    confidenceThreshold: 0.6,
    maxDataPoints: 1000,
    enableFeatureViz: true,
    enable3DViz: true,
    autoSave: true,
  },
  
  wsConnected: false,
  wsLatency: 0,
};

export const useAppStore = create<AppStore>()(
  devtools(
    (set, get) => ({
      ...initialState,

      // Connection actions
      setConnectionStatus: (connected) => set({ isConnected: connected, lastUpdate: new Date() }),
      setProcessingStatus: (processing) => set({ isProcessing: processing, lastUpdate: new Date() }),
      setView: (view) => set({ currentView: view }),
      setError: (error) => set({ errorMessage: error }),

      // EMG Data actions
      addEMGData: (data) => set((state) => {
        const newData = [...state.emgData, data];
        const maxPoints = state.settings.maxDataPoints;
        return {
          emgData: newData.slice(-maxPoints),
          lastUpdate: new Date(),
        };
      }),
      
      clearEMGData: () => set({ emgData: [] }),
      
      setCurrentGesture: (gesture, confidence) => set({
        currentGesture: gesture,
        gestureConfidence: confidence,
        lastUpdate: new Date(),
      }),
      
      updateRealtimeFeatures: (features) => set({ realtimeFeatures: features }),

      // Metrics actions
      updateMetrics: (updates) => set((state) => ({
        metrics: { ...state.metrics, ...updates },
        lastUpdate: new Date(),
      })),
      
      incrementPredictions: () => set((state) => ({
        metrics: {
          ...state.metrics,
          totalPredictions: state.metrics.totalPredictions + 1,
          processedSamples: state.metrics.processedSamples + 1,
        },
      })),
      
      recordLatency: (latency) => set((state) => ({
        metrics: {
          ...state.metrics,
          latency: (state.metrics.latency * 0.9) + (latency * 0.1), // Exponential moving average
        },
      })),
      
      recordSafetyEvent: () => set((state) => ({
        metrics: {
          ...state.metrics,
          safetyEvents: state.metrics.safetyEvents + 1,
        },
      })),

      // Model actions
      setActiveModel: (modelName) => set((state) => ({
        activeModel: modelName,
        availableModels: state.availableModels.map(model => ({
          ...model,
          isActive: model.name === modelName,
        })),
      })),
      
      updateModelInfo: (modelName, updates) => set((state) => ({
        availableModels: state.availableModels.map(model =>
          model.name === modelName ? { ...model, ...updates } : model
        ),
      })),
      
      addModel: (model) => set((state) => ({
        availableModels: [...state.availableModels, model],
      })),

      // Training actions
      startTrainingSession: (gestureLabel, sessionName) => set({
        trainingSession: {
          id: `session_${Date.now()}`,
          name: sessionName || `${gestureLabel}_${new Date().toISOString().split('T')[0]}`,
          startTime: new Date(),
          gestureLabel,
          samplesCollected: 0,
          status: 'active',
        },
        trainingData: [],
      }),
      
      pauseTrainingSession: () => set((state) => ({
        trainingSession: state.trainingSession ? {
          ...state.trainingSession,
          status: 'paused' as const,
        } : undefined,
      })),
      
      resumeTrainingSession: () => set((state) => ({
        trainingSession: state.trainingSession ? {
          ...state.trainingSession,
          status: 'active' as const,
        } : undefined,
      })),
      
      endTrainingSession: () => set((state) => ({
        trainingSession: state.trainingSession ? {
          ...state.trainingSession,
          endTime: new Date(),
          status: 'completed' as const,
        } : undefined,
      })),
      
      addTrainingData: (data) => set((state) => ({
        trainingData: [...state.trainingData, data],
        trainingSession: state.trainingSession ? {
          ...state.trainingSession,
          samplesCollected: state.trainingSession.samplesCollected + 1,
        } : undefined,
      })),
      
      clearTrainingData: () => set({ trainingData: [] }),

      // Settings actions
      updateSettings: (updates) => set((state) => ({
        settings: { ...state.settings, ...updates },
      })),

      // WebSocket actions
      setWebSocketStatus: (connected, latency = 0) => set({
        wsConnected: connected,
        wsLatency: latency,
      }),
    }),
    {
      name: 'motion-ai-store',
    }
  )
);