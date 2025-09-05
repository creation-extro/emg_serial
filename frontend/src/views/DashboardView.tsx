import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
  LinearProgress,
  Button,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  Timeline,
  Psychology,
  Speed,
  Security,
  PlayArrow,
  Stop
} from '@mui/icons-material';
import EMGVisualizer from '../components/EMGVisualizer';
import GestureClassifier from '../components/GestureClassifier';
import SafetyMonitor from '../components/SafetyMonitor';
import MetricsPanel from '../components/MetricsPanel';
import { useAppStore } from '../store/useAppStore';
import { useMotionAI } from '../hooks/useMotionAI';

const DashboardView: React.FC = () => {
  const {
    emgData,
    currentGesture,
    gestureConfidence,
    isProcessing,
    metrics,
    isConnected
  } = useAppStore();

  const { startLiveProcessing, stopLiveProcessing } = useMotionAI();

  const handleToggleProcessing = () => {
    if (isProcessing) {
      stopLiveProcessing();
    } else {
      startLiveProcessing();
    }
  };

  return (
    <Box>
      {/* Control Panel */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item>
              <FormControlLabel
                control={
                  <Switch
                    checked={isProcessing}
                    onChange={handleToggleProcessing}
                    disabled={!isConnected}
                  />
                }
                label="Live Processing Mode"
              />
            </Grid>
            <Grid item>
              <Button
                variant={isProcessing ? "contained" : "outlined"}
                color={isProcessing ? "error" : "primary"}
                startIcon={isProcessing ? <Stop /> : <PlayArrow />}
                onClick={handleToggleProcessing}
                disabled={!isConnected}
              >
                {isProcessing ? 'Stop' : 'Start'}
              </Button>
            </Grid>
            <Grid item xs>
              <Typography variant="body2" color="textSecondary">
                Status: {isConnected ? 'Connected' : 'Disconnected'} | 
                Processing: {isProcessing ? 'Active' : 'Inactive'}
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

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
                System Status
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="textSecondary" gutterBottom>
                  Processing Status
                </Typography>
                <LinearProgress 
                  variant={isProcessing ? "indeterminate" : "determinate"}
                  value={isProcessing ? undefined : 100}
                  color={isProcessing ? "primary" : "success"}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardView;