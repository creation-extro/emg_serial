import React from 'react';
import {
  Box,
  Typography,
  LinearProgress,
  Avatar,
  Chip,
  Grid,
  Card,
  CardContent
} from '@mui/material';
import {
  PanTool,
  OpenWith,
  TouchApp,
  Fingerprint,
  Stop,
  TrendingUp,
  CheckCircle,
  Warning
} from '@mui/icons-material';

interface GestureClassifierProps {
  gesture: string;
  confidence: number;
  isProcessing: boolean;
}

const GestureClassifier: React.FC<GestureClassifierProps> = ({
  gesture,
  confidence,
  isProcessing
}) => {
  // Map gestures to icons and colors
  const gestureConfig: { [key: string]: { icon: React.ReactNode; color: string; label: string } } = {
    'rest': { 
      icon: <Stop />, 
      color: '#9e9e9e', 
      label: 'Rest' 
    },
    'fist': { 
      icon: <PanTool />, 
      color: '#f44336', 
      label: 'Fist/Grip' 
    },
    'open': { 
      icon: <OpenWith />, 
      color: '#4caf50', 
      label: 'Open Hand' 
    },
    'pinch': { 
      icon: <TouchApp />, 
      color: '#2196f3', 
      label: 'Pinch' 
    },
    'point': { 
      icon: <Fingerprint />, 
      color: '#ff9800', 
      label: 'Point' 
    },
    'four': { 
      icon: <OpenWith />, 
      color: '#9c27b0', 
      label: 'Four Fingers' 
    },
    'five': { 
      icon: <OpenWith />, 
      color: '#00bcd4', 
      label: 'Five Fingers' 
    },
    'peace': { 
      icon: <TouchApp />, 
      color: '#795548', 
      label: 'Peace Sign' 
    }
  };

  const currentGesture = gestureConfig[gesture] || gestureConfig['rest'];
  const confidencePercentage = Math.round(confidence * 100);
  
  // Determine confidence level
  const getConfidenceLevel = (conf: number) => {
    if (conf >= 0.8) return { level: 'High', color: '#4caf50', icon: <CheckCircle /> };
    if (conf >= 0.6) return { level: 'Medium', color: '#ff9800', icon: <TrendingUp /> };
    return { level: 'Low', color: '#f44336', icon: <Warning /> };
  };

  const confidenceLevel = getConfidenceLevel(confidence);

  return (
    <Box>
      {/* Main Gesture Display */}
      <Box sx={{ textAlign: 'center', mb: 3 }}>
        <Avatar
          sx={{
            width: 80,
            height: 80,
            bgcolor: currentGesture.color,
            mx: 'auto',
            mb: 2,
            fontSize: '2rem'
          }}
        >
          {currentGesture.icon}
        </Avatar>
        
        <Typography variant="h4" component="h2" gutterBottom>
          {currentGesture.label}
        </Typography>
        
        <Chip
          label={isProcessing ? 'Processing...' : 'Idle'}
          color={isProcessing ? 'primary' : 'default'}
          variant={isProcessing ? 'filled' : 'outlined'}
        />
      </Box>

      {/* Confidence Display */}
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item>
              <Avatar sx={{ bgcolor: confidenceLevel.color, width: 32, height: 32 }}>
                {confidenceLevel.icon}
              </Avatar>
            </Grid>
            <Grid item xs>
              <Typography variant="subtitle1" gutterBottom>
                Confidence: {confidencePercentage}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={confidencePercentage}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: confidenceLevel.color
                  }
                }}
              />
            </Grid>
            <Grid item>
              <Typography variant="caption" color="textSecondary">
                {confidenceLevel.level}
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Gesture Legend */}
      <Box>
        <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
          Available Gestures:
        </Typography>
        <Grid container spacing={1}>
          {Object.entries(gestureConfig).map(([key, config]) => (
            <Grid item key={key}>
              <Chip
                icon={config.icon}
                label={config.label}
                size="small"
                variant={key === gesture ? 'filled' : 'outlined'}
                sx={{
                  backgroundColor: key === gesture ? config.color : 'transparent',
                  color: key === gesture ? 'white' : config.color,
                  borderColor: config.color,
                  '& .MuiChip-icon': {
                    color: key === gesture ? 'white' : config.color
                  }
                }}
              />
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Processing Indicator */}
      {isProcessing && (
        <Box sx={{ mt: 2 }}>
          <LinearProgress color="primary" />
          <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block', textAlign: 'center' }}>
            Real-time gesture classification active
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default GestureClassifier;