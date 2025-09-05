import { io, Socket } from 'socket.io-client';
import { EMGData } from '../store/useAppStore';

export interface WebSocketConfig {
  url: string;
  autoConnect: boolean;
  reconnectAttempts: number;
  reconnectInterval: number;
}

export interface EMGStreamData {
  timestamp: number;
  channels: number[];
  gesture?: string;
  confidence?: number;
  features?: { [key: string]: number };
  metadata?: { [key: string]: any };
}

export interface SystemEvent {
  type: 'connection' | 'error' | 'safety' | 'model_update' | 'training';
  message: string;
  timestamp: number;
  data?: any;
}

export class WebSocketService {
  private socket: Socket | null = null;
  private config: WebSocketConfig;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private currentAttempt = 0;
  private isConnecting = false;

  // Event callbacks
  private onDataCallback?: (data: EMGStreamData) => void;
  private onEventCallback?: (event: SystemEvent) => void;
  private onConnectedCallback?: () => void;
  private onDisconnectedCallback?: () => void;
  private onErrorCallback?: (error: string) => void;

  constructor(config: Partial<WebSocketConfig> = {}) {
    this.config = {
      url: 'ws://localhost:8000',
      autoConnect: true,
      reconnectAttempts: 5,
      reconnectInterval: 3000,
      ...config,
    };

    if (this.config.autoConnect) {
      this.connect();
    }
  }

  /**
   * Connect to the WebSocket server
   */
  public connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket?.connected || this.isConnecting) {
        resolve();
        return;
      }

      this.isConnecting = true;
      console.log('🔌 Connecting to WebSocket server:', this.config.url);

      this.socket = io(this.config.url, {
        transports: ['websocket'],
        timeout: 5000,
        forceNew: true,
      });

      // Connection successful
      this.socket.on('connect', () => {
        console.log('✅ WebSocket connected');
        this.isConnecting = false;
        this.currentAttempt = 0;
        this.clearReconnectTimer();
        this.onConnectedCallback?.();
        resolve();
      });

      // Connection failed
      this.socket.on('connect_error', (error) => {
        console.error('❌ WebSocket connection error:', error);
        this.isConnecting = false;
        this.onErrorCallback?.(error.message || 'Connection failed');
        this.scheduleReconnect();
        reject(error);
      });

      // Disconnection
      this.socket.on('disconnect', (reason) => {
        console.log('🔌 WebSocket disconnected:', reason);
        this.onDisconnectedCallback?.();
        
        if (reason === 'io server disconnect') {
          // Server initiated disconnect, don't reconnect automatically
          console.log('Server disconnected the client');
        } else {
          // Client initiated or network disconnect, attempt reconnect
          this.scheduleReconnect();
        }
      });

      // EMG data stream
      this.socket.on('emg_data', (data: EMGStreamData) => {
        this.onDataCallback?.(data);
      });

      // System events
      this.socket.on('system_event', (event: SystemEvent) => {
        console.log('📡 System event:', event);
        this.onEventCallback?.(event);
      });

      // Model updates
      this.socket.on('model_update', (data) => {
        this.onEventCallback?.({
          type: 'model_update',
          message: 'Model configuration updated',
          timestamp: Date.now(),
          data,
        });
      });

      // Training updates
      this.socket.on('training_update', (data) => {
        this.onEventCallback?.({
          type: 'training',
          message: 'Training progress update',
          timestamp: Date.now(),
          data,
        });
      });

      // Generic error handling
      this.socket.on('error', (error) => {
        console.error('💥 WebSocket error:', error);
        this.onErrorCallback?.(error.message || 'Unknown error');
      });
    });
  }

  /**
   * Disconnect from the WebSocket server
   */
  public disconnect(): void {
    this.clearReconnectTimer();
    
    if (this.socket) {
      console.log('🔌 Disconnecting WebSocket');
      this.socket.disconnect();
      this.socket = null;
    }
  }

  /**
   * Check if connected
   */
  public isConnected(): boolean {
    return this.socket?.connected || false;
  }

  /**
   * Send a message to the server
   */
  public send(event: string, data: any): void {
    if (!this.socket?.connected) {
      console.warn('⚠️ Cannot send message: WebSocket not connected');
      return;
    }

    this.socket.emit(event, data);
  }

  /**
   * Start EMG data streaming
   */
  public startStreaming(config?: { sampleRate?: number; channels?: number }): void {
    this.send('start_streaming', config || {});
  }

  /**
   * Stop EMG data streaming
   */
  public stopStreaming(): void {
    this.send('stop_streaming', {});
  }

  /**
   * Send training data
   */
  public sendTrainingData(data: EMGData[], gestureLabel: string): void {
    this.send('training_data', {
      data,
      gestureLabel,
      timestamp: Date.now(),
    });
  }

  /**
   * Request model training
   */
  public requestTraining(config: {
    modelType: string;
    parameters?: { [key: string]: any };
    trainingData?: EMGData[];
  }): void {
    this.send('train_model', config);
  }

  /**
   * Request model switch
   */
  public switchModel(modelName: string): void {
    this.send('switch_model', { modelName });
  }

  /**
   * Send system command
   */
  public sendCommand(command: string, params?: any): void {
    this.send('system_command', { command, params });
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    if (this.currentAttempt >= this.config.reconnectAttempts) {
      console.log('🚫 Max reconnection attempts reached');
      this.onErrorCallback?.('Max reconnection attempts reached');
      return;
    }

    this.currentAttempt++;
    console.log(`🔄 Scheduling reconnect attempt ${this.currentAttempt}/${this.config.reconnectAttempts} in ${this.config.reconnectInterval}ms`);

    this.reconnectTimer = setTimeout(() => {
      console.log(`🔄 Reconnection attempt ${this.currentAttempt}/${this.config.reconnectAttempts}`);
      this.connect().catch(() => {
        // Error already handled in connect method
      });
    }, this.config.reconnectInterval);
  }

  /**
   * Clear reconnection timer
   */
  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  /**
   * Event listeners
   */
  public onData(callback: (data: EMGStreamData) => void): void {
    this.onDataCallback = callback;
  }

  public onEvent(callback: (event: SystemEvent) => void): void {
    this.onEventCallback = callback;
  }

  public onConnected(callback: () => void): void {
    this.onConnectedCallback = callback;
  }

  public onDisconnected(callback: () => void): void {
    this.onDisconnectedCallback = callback;
  }

  public onError(callback: (error: string) => void): void {
    this.onErrorCallback = callback;
  }

  /**
   * Get connection statistics
   */
  public getStats(): {
    connected: boolean;
    reconnectAttempts: number;
    maxAttempts: number;
    url: string;
  } {
    return {
      connected: this.isConnected(),
      reconnectAttempts: this.currentAttempt,
      maxAttempts: this.config.reconnectAttempts,
      url: this.config.url,
    };
  }

  /**
   * Update configuration
   */
  public updateConfig(newConfig: Partial<WebSocketConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Cleanup and destroy
   */
  public destroy(): void {
    this.clearReconnectTimer();
    this.disconnect();
    this.onDataCallback = undefined;
    this.onEventCallback = undefined;
    this.onConnectedCallback = undefined;
    this.onDisconnectedCallback = undefined;
    this.onErrorCallback = undefined;
  }
}

// Singleton instance
let wsInstance: WebSocketService | null = null;

export const getWebSocketService = (config?: Partial<WebSocketConfig>): WebSocketService => {
  if (!wsInstance) {
    wsInstance = new WebSocketService(config);
  }
  return wsInstance;
};

export const destroyWebSocketService = (): void => {
  if (wsInstance) {
    wsInstance.destroy();
    wsInstance = null;
  }
};