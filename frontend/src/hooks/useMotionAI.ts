import { useEffect, useCallback } from 'react';
import { useAppStore } from '../store/useAppStore';
import { getWebSocketService } from '../services/WebSocketService';
import { MotionAIService } from '../services/MotionAIService';

export const useMotionAI = () => {
  const {
    setConnectionStatus,
    setProcessingStatus,
    addEMGData,
    setCurrentGesture,
    incrementPredictions,
    setWebSocketStatus,
    isProcessing
  } = useAppStore();

  const motionAIService = new MotionAIService('http://localhost:8000');
  const wsService = getWebSocketService();

  const generateSampleData = useCallback(() => {
    const gestures = ['rest', 'fist', 'open', 'pinch', 'point'];
    const randomGesture = gestures[Math.floor(Math.random() * gestures.length)];
    
    const patterns: { [key: string]: number[] } = {
      'rest': [0.1, 0.1, 0.1],
      'fist': [0.8, 0.6, 0.4],
      'open': [0.3, 0.7, 0.2],
      'pinch': [0.2, 0.4, 0.9],
      'point': [0.1, 0.3, 0.8]
    };
    
    const basePattern = patterns[randomGesture] || patterns['rest'];
    const channels = basePattern.map(val => 
      Math.max(0, Math.min(1, val + (Math.random() - 0.5) * 0.1))
    );
    
    return {
      timestamp: Date.now(),
      channels,
      gesture: randomGesture,
      confidence: 0.7 + Math.random() * 0.25
    };
  }, []);

  const startLiveProcessing = useCallback(() => {
    setProcessingStatus(true);
    
    // Start simulation
    const interval = setInterval(() => {
      const data = generateSampleData();
      addEMGData(data);
      setCurrentGesture(data.gesture, data.confidence);
      incrementPredictions();
    }, 200);

    return () => clearInterval(interval);
  }, [setProcessingStatus, generateSampleData, addEMGData, setCurrentGesture, incrementPredictions]);

  const stopLiveProcessing = useCallback(() => {
    setProcessingStatus(false);
  }, [setProcessingStatus]);

  const checkHealth = useCallback(async () => {
    try {
      const health = await motionAIService.checkHealth();
      setConnectionStatus(health.status === 'ok');
      return health.status === 'ok';
    } catch (error) {
      setConnectionStatus(false);
      return false;
    }
  }, [setConnectionStatus]);

  useEffect(() => {
    checkHealth();
    
    // Setup WebSocket
    wsService.onConnected(() => setWebSocketStatus(true));
    wsService.onDisconnected(() => setWebSocketStatus(false));
    wsService.onData((data) => {
      addEMGData({
        timestamp: data.timestamp * 1000,
        channels: data.channels,
        gesture: data.gesture,
        confidence: data.confidence
      });
    });

    return () => wsService.destroy();
  }, []);

  return {
    startLiveProcessing,
    stopLiveProcessing,
    checkHealth,
    isProcessing
  };
};