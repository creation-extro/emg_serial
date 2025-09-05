import React from 'react';
import { Grid, Typography, Box } from '@mui/material';
import Gesture3DVisualization from '../components/Gesture3DVisualization';

const Visualization3DView: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        3D Gesture Visualization
      </Typography>
      <Typography variant="body1" color="textSecondary" paragraph>
        Interactive 3D visualization of hand gestures and EMG signal patterns.
        Use the controls to customize the view and analyze gesture recognition in real-time.
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Gesture3DVisualization />
        </Grid>
      </Grid>
    </Box>
  );
};

export default Visualization3DView;