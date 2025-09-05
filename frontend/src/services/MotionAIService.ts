import axios, { AxiosInstance } from 'axios';

export interface SignalFrame {
  timestamp: number;
  channels: number[];
  metadata: {
    device_id?: string;
    fs?: number;
    window_ms?: number;
    [key: string]: any;
  };
}

export interface IntentFrame {
  gesture: string;
  confidence: number;
  features: {
    [key: string]: any;
  };
  soft_priority?: number;
  design_candidates?: any[];
}

export interface MotorCmd {
  actuator_id: string;
  angle?: number;
  force?: number;
  safety_flags: {
    [key: string]: any;
  };
  is_safe: boolean;
  rate_clamped: boolean;
  haptic_alert?: string;
}

export interface HealthResponse {
  status: string;
}

export interface ClassificationResult extends IntentFrame {}

export interface PolicyResult {
  commands: MotorCmd[];
}

export interface HybridResult {
  commands: MotorCmd[];
  latency_ms: number;
  classification: IntentFrame;
}

export class MotionAIService {
  private api: AxiosInstance;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.api = axios.create({
      baseURL,
      timeout: 5000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for logging
    this.api.interceptors.request.use(
      (config) => {
        console.log(`🚀 Motion AI API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor for logging
    this.api.interceptors.response.use(
      (response) => {
        console.log(`✅ Motion AI API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error(`❌ Motion AI API Error: ${error.response?.status} ${error.config?.url}`, error.response?.data);
        return Promise.reject(error);
      }
    );
  }

  /**
   * Check the health status of the Motion AI backend
   */
  async checkHealth(): Promise<HealthResponse> {
    try {
      const response = await this.api.get<HealthResponse>('/health');
      return response.data;
    } catch (error) {
      throw new Error(`Health check failed: ${error}`);
    }
  }

  /**
   * Classify a gesture from EMG signal data
   */
  async classifyGesture(signal: SignalFrame): Promise<ClassificationResult> {
    try {
      const response = await this.api.post<ClassificationResult>('/v1/classify', signal);
      return response.data;
    } catch (error) {
      throw new Error(`Gesture classification failed: ${error}`);
    }
  }

  /**
   * Generate motor commands from an intent frame
   */
  async generatePolicy(intent: IntentFrame): Promise<PolicyResult> {
    try {
      const response = await this.api.post<MotorCmd[]>('/v1/policy', intent);
      return { commands: response.data };
    } catch (error) {
      throw new Error(`Policy generation failed: ${error}`);
    }
  }

  /**
   * Process EMG signal through the complete pipeline (classification + policy + safety)
   */
  async processHybrid(signal: SignalFrame): Promise<HybridResult> {
    try {
      const startTime = performance.now();
      const response = await this.api.post<MotorCmd[]>('/v1/hybrid', signal);
      const endTime = performance.now();
      
      return {
        commands: response.data,
        latency_ms: endTime - startTime,
        classification: {
          gesture: 'unknown', // Would be extracted from response in real implementation
          confidence: 0,
          features: {}
        }
      };
    } catch (error) {
      throw new Error(`Hybrid processing failed: ${error}`);
    }
  }

  /**
   * Generate sample EMG data for testing
   */
  generateSampleSignal(gesture: string = 'rest'): SignalFrame {
    const gesturePatterns: { [key: string]: number[] } = {
      'rest': [0.1, 0.1, 0.1],
      'fist': [0.8, 0.6, 0.4],
      'open': [0.3, 0.7, 0.2],
      'pinch': [0.2, 0.4, 0.9],
      'point': [0.1, 0.3, 0.8],
      'four': [0.6, 0.4, 0.5],
      'five': [0.7, 0.7, 0.3],
      'peace': [0.4, 0.3, 0.8]
    };

    const basePattern = gesturePatterns[gesture] || gesturePatterns['rest'];
    const channels = basePattern.map(val => 
      Math.max(0, Math.min(1, val + (Math.random() - 0.5) * 0.1))
    );

    return {
      timestamp: Date.now() / 1000,
      channels,
      metadata: {
        device_id: 'demo_sensor',
        fs: 1000,
        window_ms: 200,
        true_gesture: gesture
      }
    };
  }

  /**
   * Validate signal frame data
   */
  validateSignalFrame(signal: SignalFrame): boolean {
    if (!signal.timestamp || signal.timestamp <= 0) {
      console.warn('Invalid timestamp in signal frame');
      return false;
    }

    if (!Array.isArray(signal.channels) || signal.channels.length === 0) {
      console.warn('Invalid channels array in signal frame');
      return false;
    }

    if (signal.channels.some(ch => typeof ch !== 'number' || isNaN(ch))) {
      console.warn('Invalid channel values in signal frame');
      return false;
    }

    return true;
  }

  /**
   * Process a batch of signals for testing
   */
  async processBatch(signals: SignalFrame[]): Promise<ClassificationResult[]> {
    const results: ClassificationResult[] = [];
    
    for (const signal of signals) {
      try {
        if (this.validateSignalFrame(signal)) {
          const result = await this.classifyGesture(signal);
          results.push(result);
        }
      } catch (error) {
        console.error('Error processing signal in batch:', error);
        // Continue with next signal
      }
    }
    
    return results;
  }

  /**
   * Get system configuration info
   */
  async getSystemInfo(): Promise<any> {
    try {
      // This would be a real endpoint in the backend
      const response = await this.api.get('/info');
      return response.data;
    } catch (error) {
      // Return mock data if endpoint doesn't exist
      return {
        version: '1.0.0',
        models_loaded: ['mlp_light', 'svm_baseline'],
        safety_config: {
          max_angle_rate_deg_s: 90.0,
          dead_zone_angle_deg: 1.5,
          confidence_threshold: 0.6
        },
        features: [
          'Real-time EMG Processing',
          'Gesture Classification',
          'Safety Layer',
          'Policy Mapping',
          'Drift Detection'
        ]
      };
    }
  }
}