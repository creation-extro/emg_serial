import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Box as MuiBox,
  Typography
} from '@mui/material';
import { ThreeDRotation } from '@mui/icons-material';
import { useAppStore } from '../store/useAppStore';

// Temporary placeholder - 3D visualization temporarily disabled for compilation
const Gesture3DVisualization: React.FC = () => {
  const { currentGesture, gestureConfidence } = useAppStore();
  
  return (
    <Card sx={{ height: '600px' }}>
      <CardHeader
        avatar={<ThreeDRotation />}
        title="3D Gesture Visualization"
        subheader="Interactive 3D representation of gestures"
      />
      <CardContent sx={{ height: 'calc(100% - 80px)' }}>
        <MuiBox sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          height: '100%',
          flexDirection: 'column',
          gap: 2,
          bgcolor: '#f5f5f5',
          borderRadius: 1
        }}>
          <ThreeDRotation sx={{ fontSize: 60, color: 'primary.main' }} />
          <Typography variant="h6" color="primary">
            {currentGesture.toUpperCase()}
          </Typography>
          <Typography variant="body1" color="textSecondary">
            Confidence: {(gestureConfidence * 100).toFixed(1)}%
          </Typography>
          <Typography variant="body2" color="textSecondary">
            3D visualization temporarily disabled
          </Typography>
        </MuiBox>
      </CardContent>
    </Card>
  );
};

export default Gesture3DVisualization;