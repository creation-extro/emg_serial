import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  Alert,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Avatar
} from '@mui/material';
import {
  Security,
  Warning,
  CheckCircle,
  Speed,
  Block,
  HighlightOff,
  Vibration
} from '@mui/icons-material';

interface SafetyMonitorProps {
  currentGesture: string;
  confidence: number;
  safetyEvents: number;
}

const SafetyMonitor: React.FC<SafetyMonitorProps> = ({
  currentGesture,
  confidence,
  safetyEvents
}) => {
  // Safety thresholds and configuration
  const safetyConfig = {
    confidenceThreshold: 0.6,
    maxAngleRate: 90, // degrees per second
    deadZone: 1.5, // degrees
    hysteresisHigh: 2.0,
    hysteresisLow: 1.0
  };

  // Calculate safety status
  const getSafetyStatus = () => {
    const isConfidenceSafe = confidence >= safetyConfig.confidenceThreshold;
    const isGestureSafe = ['rest', 'open'].includes(currentGesture);
    
    if (isConfidenceSafe && isGestureSafe) {
      return {
        level: 'SAFE',
        color: '#4caf50',
        icon: <CheckCircle />,
        message: 'All systems operating safely'
      };
    } else if (isConfidenceSafe) {
      return {
        level: 'CAUTION',
        color: '#ff9800',
        icon: <Warning />,
        message: 'Active gesture detected - monitoring'
      };
    } else {
      return {
        level: 'WARNING',
        color: '#f44336',
        icon: <HighlightOff />,
        message: 'Low confidence - safety measures active'
      };
    }
  };

  const safetyStatus = getSafetyStatus();

  // Safety mechanisms status
  const safetyMechanisms = [
    {
      name: 'Confidence Threshold',
      active: confidence < safetyConfig.confidenceThreshold,
      description: `Minimum ${safetyConfig.confidenceThreshold * 100}% confidence required`,
      icon: <Security />,
      status: confidence >= safetyConfig.confidenceThreshold ? 'PASS' : 'ACTIVE'
    },
    {
      name: 'Dead Zone Filter',
      active: false, // Would be determined by actual movement
      description: `Filters movements < ${safetyConfig.deadZone}°`,
      icon: <Block />,
      status: 'MONITORING'
    },
    {
      name: 'Rate Limiting',
      active: false, // Would be determined by actual rate
      description: `Max speed: ${safetyConfig.maxAngleRate}°/s`,
      icon: <Speed />,
      status: 'MONITORING'
    },
    {
      name: 'Hysteresis Control',
      active: false,
      description: `Prevents oscillation (${safetyConfig.hysteresisLow}° - ${safetyConfig.hysteresisHigh}°)`,
      icon: <Vibration />,
      status: 'MONITORING'
    }
  ];

  return (
    <Box>
      {/* Overall Safety Status */}
      <Card 
        variant="outlined" 
        sx={{ 
          mb: 2, 
          borderColor: safetyStatus.color,
          backgroundColor: `${safetyStatus.color}10`
        }}
      >
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item>
              <Avatar sx={{ bgcolor: safetyStatus.color }}>
                {safetyStatus.icon}
              </Avatar>
            </Grid>
            <Grid item xs>
              <Typography variant="h6" sx={{ color: safetyStatus.color, fontWeight: 'bold' }}>
                {safetyStatus.level}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {safetyStatus.message}
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Confidence Monitor */}
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Confidence Level
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ width: '100%' }}>
              <LinearProgress
                variant="determinate"
                value={confidence * 100}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: confidence >= safetyConfig.confidenceThreshold ? '#4caf50' : '#f44336'
                  }
                }}
              />
            </Box>
            <Typography variant="body2" sx={{ minWidth: 35 }}>
              {Math.round(confidence * 100)}%
            </Typography>
          </Box>
          <Typography variant="caption" color="textSecondary">
            Threshold: {safetyConfig.confidenceThreshold * 100}%
          </Typography>
        </CardContent>
      </Card>

      {/* Safety Mechanisms */}
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Safety Mechanisms
          </Typography>
          <List dense>
            {safetyMechanisms.map((mechanism, index) => (
              <ListItem key={index} sx={{ px: 0 }}>
                <ListItemIcon>
                  <Avatar 
                    sx={{ 
                      width: 32, 
                      height: 32,
                      bgcolor: mechanism.active ? '#f44336' : 
                              mechanism.status === 'PASS' ? '#4caf50' : '#9e9e9e'
                    }}
                  >
                    {mechanism.icon}
                  </Avatar>
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body2">{mechanism.name}</Typography>
                      <Chip
                        label={mechanism.status}
                        size="small"
                        color={
                          mechanism.status === 'PASS' ? 'success' :
                          mechanism.status === 'ACTIVE' ? 'error' : 'default'
                        }
                        variant="outlined"
                      />
                    </Box>
                  }
                  secondary={mechanism.description}
                />
              </ListItem>
            ))}
          </List>
        </CardContent>
      </Card>

      {/* Safety Events Counter */}
      <Card variant="outlined">
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Safety Events
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="h4" color="primary">
                {safetyEvents}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Total Interventions
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="h4" color="success.main">
                {Math.max(0, 100 - safetyEvents)}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Safe Operations
              </Typography>
            </Grid>
          </Grid>
          
          {safetyEvents > 10 && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              High number of safety interventions detected. Consider system recalibration.
            </Alert>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default SafetyMonitor;