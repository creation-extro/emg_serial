import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Box,
  Chip,
  Alert,
  Paper,
  Button,
  Switch,
  FormControlLabel,
  LinearProgress
} from '@mui/material';
import {
  SignalCellular4Bar,
  Security,
  Psychology,
  Speed,
  Timeline,
  PlayArrow,
  Stop
} from '@mui/icons-material';
import EMGVisualizer from './components/EMGVisualizer';
import GestureClassifier from './components/GestureClassifier';
import SafetyMonitor from './components/SafetyMonitor';
import MetricsPanel from './components/MetricsPanel';
import { MotionAIService } from './services/MotionAIService';
import './App.css';

interface SystemStatus {
  connected: boolean;
  processing: boolean;
  lastUpdate: string;
  errorMessage?: string;
}

interface EMGData {
  timestamp: number;
  channels: number[];
  gesture?: string;
  confidence?: number;
}

const App: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    connected: false,
    processing: false,
    lastUpdate: new Date().toISOString()
  });
  
  const [emgData, setEMGData] = useState<EMGData[]>([]);
  const [currentGesture, setCurrentGesture] = useState<string>('rest');
  const [gestureConfidence, setGestureConfidence] = useState<number>(0);
  const [isLiveMode, setIsLiveMode] = useState<boolean>(false);
  const [metrics, setMetrics] = useState({
    accuracy: 0,
    latency: 0,
    safetyEvents: 0,
    totalPredictions: 0
  });

  const motionAIService = new MotionAIService('http://localhost:8000');

  useEffect(() => {
    checkSystemHealth();
    if (isLiveMode) {
      startLiveProcessing();
    }
    return () => {
      if (isLiveMode) {
        stopLiveProcessing();
      }
    };
  }, [isLiveMode]);

  const checkSystemHealth = async () => {
    try {
      const health = await motionAIService.checkHealth();
      setSystemStatus(prev => ({
        ...prev,
        connected: health.status === 'ok',
        lastUpdate: new Date().toISOString(),
        errorMessage: undefined
      }));
    } catch (error) {
      setSystemStatus(prev => ({
        ...prev,
        connected: false,
        errorMessage: 'Failed to connect to Motion AI backend'
      }));
    }
  };

  const generateSampleEMGData = (): EMGData => {
    const gestures = ['rest', 'fist', 'open', 'pinch', 'point'];
    const randomGesture = gestures[Math.floor(Math.random() * gestures.length)];
    
    // Generate realistic EMG patterns based on gesture
    const gesturePatterns: { [key: string]: number[] } = {
      'rest': [0.1, 0.1, 0.1],
      'fist': [0.8, 0.6, 0.4],
      'open': [0.3, 0.7, 0.2],
      'pinch': [0.2, 0.4, 0.9],
      'point': [0.1, 0.3, 0.8]
    };
    
    const basePattern = gesturePatterns[randomGesture] || gesturePatterns['rest'];
    const channels = basePattern.map(val => val + (Math.random() - 0.5) * 0.1);
    
    return {
      timestamp: Date.now(),
      channels: channels,
      gesture: randomGesture,
      confidence: 0.7 + Math.random() * 0.25
    };
  };

  const startLiveProcessing = () => {
    const interval = setInterval(async () => {
      try {
        // Generate sample data (in real implementation, this would come from EMG sensors)
        const newData = generateSampleEMGData();
        
        // Process through Motion AI service
        const result = await motionAIService.classifyGesture({
          timestamp: newData.timestamp / 1000,
          channels: newData.channels,
          metadata: { device_id: 'demo_sensor' }
        });

        setCurrentGesture(result.gesture);
        setGestureConfidence(result.confidence);
        
        // Update EMG data buffer (keep last 100 points)
        setEMGData(prev => {
          const updated = [...prev, newData];
          return updated.slice(-100);
        });

        // Update metrics
        setMetrics(prev => ({
          ...prev,
          totalPredictions: prev.totalPredictions + 1,
          latency: Math.random() * 20 + 5, // Simulated latency
          accuracy: (prev.accuracy * 0.9) + (result.confidence * 0.1)
        }));

        setSystemStatus(prev => ({
          ...prev,
          processing: true,
          lastUpdate: new Date().toISOString()
        }));

      } catch (error) {
        console.error('Error in live processing:', error);
        setSystemStatus(prev => ({
          ...prev,
          processing: false,
          errorMessage: 'Live processing error'
        }));
      }
    }, 200); // 200ms interval (5 Hz)

    return () => clearInterval(interval);
  };

  const stopLiveProcessing = () => {
    setSystemStatus(prev => ({
      ...prev,
      processing: false
    }));
  };

  const handleToggleLiveMode = () => {
    setIsLiveMode(!isLiveMode);
  };

  const getStatusColor = () => {
    if (!systemStatus.connected) return 'error';
    if (systemStatus.processing) return 'success';
    return 'warning';
  };

  const getStatusText = () => {
    if (!systemStatus.connected) return 'Disconnected';
    if (systemStatus.processing) return 'Live Processing';
    return 'Connected';
  };

  return (
    <div className="App">
      <AppBar position="static" sx={{ backgroundColor: '#1976d2' }}>
        <Toolbar>
          <Psychology sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Motion AI - EMG Gesture Recognition System
          </Typography>
          <Chip
            icon={<SignalCellular4Bar />}
            label={getStatusText()}
            color={getStatusColor()}
            variant="outlined"
            sx={{ color: 'white', borderColor: 'white' }}
          />
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 2, mb: 2 }}>
        {systemStatus.errorMessage && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {systemStatus.errorMessage}
          </Alert>
        )}
        
        <Box sx={{ mb: 2 }}>
          <Card>
            <CardContent>
              <Grid container spacing={2} alignItems="center">
                <Grid item>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={isLiveMode}
                        onChange={handleToggleLiveMode}
                        disabled={!systemStatus.connected}
                      />
                    }
                    label="Live Processing Mode"
                  />
                </Grid>
                <Grid item>
                  {isLiveMode ? (
                    <Button
                      variant="contained"
                      color="error"
                      startIcon={<Stop />}
                      onClick={() => setIsLiveMode(false)}
                    >
                      Stop
                    </Button>
                  ) : (
                    <Button
                      variant="contained"
                      color="primary"
                      startIcon={<PlayArrow />}
                      onClick={() => setIsLiveMode(true)}
                      disabled={!systemStatus.connected}
                    >
                      Start
                    </Button>
                  )}
                </Grid>
                <Grid item xs>
                  <Typography variant="body2" color="textSecondary">
                    Last Update: {new Date(systemStatus.lastUpdate).toLocaleTimeString()}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Box>

        <Grid container spacing={3}>
          {/* EMG Signal Visualization */}
          <Grid item xs={12} md={8}>
            <Card>
              <CardHeader
                avatar={<Timeline />}
                title="EMG Signal Visualization"
                subheader="Real-time multi-channel EMG data"
              />
              <CardContent>
                <EMGVisualizer data={emgData} />
              </CardContent>
            </Card>
          </Grid>

          {/* Current Gesture Display */}
          <Grid item xs={12} md={4}>
            <Card>
              <CardHeader
                avatar={<Psychology />}
                title="Gesture Recognition"
                subheader="Current detected gesture"
              />
              <CardContent>
                <GestureClassifier
                  gesture={currentGesture}
                  confidence={gestureConfidence}
                  isProcessing={systemStatus.processing}
                />
              </CardContent>
            </Card>
          </Grid>

          {/* System Metrics */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader
                avatar={<Speed />}
                title="Performance Metrics"
                subheader="System performance indicators"
              />
              <CardContent>
                <MetricsPanel metrics={metrics} />
              </CardContent>
            </Card>
          </Grid>

          {/* Safety Monitor */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader
                avatar={<Security />}
                title="Safety Monitor"
                subheader="Safety system status"
              />
              <CardContent>
                <SafetyMonitor
                  currentGesture={currentGesture}
                  confidence={gestureConfidence}
                  safetyEvents={metrics.safetyEvents}
                />
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>
    </div>
  );
};

export default App;