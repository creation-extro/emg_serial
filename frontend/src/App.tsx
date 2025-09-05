import React, { useEffect } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Container, AppBar, Toolbar, Typography, Grid, Card, CardContent, CardHeader, Box } from '@mui/material';
import { Psychology, Timeline, Speed, Security } from '@mui/icons-material';
import EMGVisualizer from './components/EMGVisualizer';
import GestureClassifier from './components/GestureClassifier';
import SafetyMonitor from './components/SafetyMonitor';
import MetricsPanel from './components/MetricsPanel';
import { useAppStore } from './store/useAppStore';
import { useMotionAI } from './hooks/useMotionAI';
import './App.css';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
});

const App: React.FC = () => {
  const {
    emgData,
    currentGesture,
    gestureConfidence,
    isProcessing,
    metrics,
    isConnected
  } = useAppStore();

  const { startLiveProcessing, stopLiveProcessing } = useMotionAI();

  useEffect(() => {
    // Auto-start processing for demo
    if (!isProcessing) {
      startLiveProcessing();
    }
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static">
        <Toolbar>
          <Psychology sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Motion AI - EMG Gesture Recognition System
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="body2">
              Status: {isConnected ? 'Connected' : 'Disconnected'}
            </Typography>
            <Typography variant="body2">
              Processing: {isProcessing ? 'Active' : 'Inactive'}
            </Typography>
          </Box>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
        <Grid container spacing={3}>
          {/* EMG Signal Visualization */}
          <Grid item xs={12} lg={8}>
            <Card sx={{ height: '400px' }}>
              <CardHeader
                avatar={<Timeline />}
                title="EMG Signal Visualization"
                subheader="Real-time multi-channel EMG data"
              />
              <CardContent sx={{ height: 'calc(100% - 80px)' }}>
                <EMGVisualizer data={emgData} height={300} />
              </CardContent>
            </Card>
          </Grid>

          {/* Current Gesture Display */}
          <Grid item xs={12} lg={4}>
            <Card sx={{ height: '400px' }}>
              <CardHeader
                avatar={<Psychology />}
                title="Gesture Recognition"
                subheader="Current detected gesture"
              />
              <CardContent>
                <GestureClassifier
                  gesture={currentGesture}
                  confidence={gestureConfidence}
                  isProcessing={isProcessing}
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

          {/* System Status */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Advanced Motion AI Frontend
                </Typography>
                <Typography variant="body1" color="textSecondary">
                  This is the advanced React frontend for the Motion AI EMG gesture recognition system.
                  The application features real-time EMG signal processing, gesture classification,
                  performance analytics, and comprehensive system monitoring.
                </Typography>
                <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  <Typography variant="body2">
                    Total Predictions: {metrics.totalPredictions}
                  </Typography>
                  <Typography variant="body2">
                    Current Gesture: {currentGesture}
                  </Typography>
                  <Typography variant="body2">
                    Confidence: {(gestureConfidence * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2">
                    Latency: {metrics.latency.toFixed(1)}ms
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>
    </ThemeProvider>
  );
};

export default App;