import React, { useState } from 'react';
import {
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  LinearProgress,
  Chip,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Stepper,
  Step,
  StepLabel,
  StepContent
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Pause,
  Delete,
  Save,
  Download,
  Upload,
  School,
  DataObject,
  Timeline,
  CheckCircle
} from '@mui/icons-material';
import { useAppStore } from '../store/useAppStore';

const gestureOptions = [
  { value: 'rest', label: 'Rest Position' },
  { value: 'fist', label: 'Fist/Grip' },
  { value: 'open', label: 'Open Hand' },
  { value: 'pinch', label: 'Pinch' },
  { value: 'point', label: 'Point' },
  { value: 'four', label: 'Four Fingers' },
  { value: 'five', label: 'Five Fingers' },
  { value: 'peace', label: 'Peace Sign' },
  { value: 'thumbs_up', label: 'Thumbs Up' },
  { value: 'custom', label: 'Custom Gesture' }
];

const TrainingSessionCard: React.FC = () => {
  const {
    trainingSession,
    trainingData,
    startTrainingSession,
    pauseTrainingSession,
    resumeTrainingSession,
    endTrainingSession,
    clearTrainingData
  } = useAppStore();

  const [gestureLabel, setGestureLabel] = useState('');
  const [sessionName, setSessionName] = useState('');
  const [customGesture, setCustomGesture] = useState('');
  const [targetSamples, setTargetSamples] = useState(100);

  const handleStartSession = () => {
    const finalGestureLabel = gestureLabel === 'custom' ? customGesture : gestureLabel;
    if (finalGestureLabel) {
      startTrainingSession(finalGestureLabel, sessionName || undefined);
    }
  };

  const progress = trainingSession ? 
    Math.min((trainingSession.samplesCollected / targetSamples) * 100, 100) : 0;

  return (
    <Card>
      <CardHeader
        avatar={<School />}
        title="Training Session"
        subheader="Collect EMG data for gesture training"
      />
      <CardContent>
        {!trainingSession ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              label="Session Name (Optional)"
              value={sessionName}
              onChange={(e) => setSessionName(e.target.value)}
              placeholder="e.g., Morning Training Session"
              fullWidth
            />
            
            <FormControl fullWidth>
              <InputLabel>Gesture Type</InputLabel>
              <Select
                value={gestureLabel}
                onChange={(e) => setGestureLabel(e.target.value)}
                label="Gesture Type"
              >
                {gestureOptions.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {gestureLabel === 'custom' && (
              <TextField
                label="Custom Gesture Name"
                value={customGesture}
                onChange={(e) => setCustomGesture(e.target.value)}
                placeholder="Enter custom gesture name"
                fullWidth
                required
              />
            )}

            <TextField
              label="Target Samples"
              type="number"
              value={targetSamples}
              onChange={(e) => setTargetSamples(parseInt(e.target.value) || 100)}
              inputProps={{ min: 10, max: 1000 }}
              fullWidth
            />

            <Button
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={handleStartSession}
              disabled={!gestureLabel || (gestureLabel === 'custom' && !customGesture)}
              fullWidth
            >
              Start Training Session
            </Button>
          </Box>
        ) : (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Box>
              <Typography variant="h6" gutterBottom>
                {trainingSession.name}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Gesture: {trainingSession.gestureLabel}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Started: {trainingSession.startTime.toLocaleTimeString()}
              </Typography>
            </Box>

            <Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">
                  Samples Collected: {trainingSession.samplesCollected}/{targetSamples}
                </Typography>
                <Typography variant="body2">
                  {progress.toFixed(1)}%
                </Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={progress}
                color={progress >= 100 ? 'success' : 'primary'}
              />
            </Box>

            <Chip
              label={trainingSession.status.toUpperCase()}
              color={
                trainingSession.status === 'active' ? 'success' :
                trainingSession.status === 'paused' ? 'warning' : 'default'
              }
              sx={{ alignSelf: 'flex-start' }}
            />

            <Box sx={{ display: 'flex', gap: 1 }}>
              {trainingSession.status === 'active' && (
                <Button
                  variant="outlined"
                  startIcon={<Pause />}
                  onClick={pauseTrainingSession}
                >
                  Pause
                </Button>
              )}
              
              {trainingSession.status === 'paused' && (
                <Button
                  variant="contained"
                  startIcon={<PlayArrow />}
                  onClick={resumeTrainingSession}
                >
                  Resume
                </Button>
              )}

              <Button
                variant="contained"
                color="success"
                startIcon={<Stop />}
                onClick={endTrainingSession}
              >
                Complete Session
              </Button>

              <Button
                variant="outlined"
                color="error"
                startIcon={<Delete />}
                onClick={() => {
                  endTrainingSession();
                  clearTrainingData();
                }}
              >
                Cancel
              </Button>
            </Box>

            {progress >= 100 && (
              <Alert severity="success">
                Target number of samples reached! You can continue collecting or complete the session.
              </Alert>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

const TrainingDataManager: React.FC = () => {
  const { trainingData, clearTrainingData } = useAppStore();
  const [exportDialogOpen, setExportDialogOpen] = useState(false);

  const exportData = () => {
    const dataBlob = new Blob([JSON.stringify(trainingData, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(dataBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_data_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setExportDialogOpen(false);
  };

  const getDataSummary = () => {
    const gestureGroups = trainingData.reduce((acc, sample) => {
      const gesture = sample.gesture || 'unknown';
      acc[gesture] = (acc[gesture] || 0) + 1;
      return acc;
    }, {} as { [key: string]: number });

    return gestureGroups;
  };

  const dataSummary = getDataSummary();

  return (
    <Card>
      <CardHeader
        avatar={<DataObject />}
        title="Training Data Manager"
        subheader={`${trainingData.length} samples collected`}
        action={
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              startIcon={<Download />}
              onClick={() => setExportDialogOpen(true)}
              disabled={trainingData.length === 0}
              size="small"
            >
              Export
            </Button>
            <Button
              startIcon={<Upload />}
              size="small"
              component="label"
            >
              Import
              <input type="file" hidden accept=".json" />
            </Button>
          </Box>
        }
      />
      <CardContent>
        {trainingData.length === 0 ? (
          <Typography variant="body2" color="textSecondary">
            No training data collected yet. Start a training session to begin collecting EMG samples.
          </Typography>
        ) : (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Data Distribution
            </Typography>
            <List dense>
              {Object.entries(dataSummary).map(([gesture, count]) => (
                <ListItem key={gesture}>
                  <ListItemText
                    primary={gesture}
                    secondary={`${count} samples`}
                  />
                  <ListItemSecondaryAction>
                    <Chip label={count} size="small" />
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
            
            <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
              <Button
                variant="outlined"
                color="error"
                startIcon={<Delete />}
                onClick={clearTrainingData}
                size="small"
              >
                Clear All Data
              </Button>
            </Box>
          </Box>
        )}

        <Dialog open={exportDialogOpen} onClose={() => setExportDialogOpen(false)}>
          <DialogTitle>Export Training Data</DialogTitle>
          <DialogContent>
            <Typography variant="body2" paragraph>
              Export {trainingData.length} samples to JSON format for use in model training.
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Data Summary:
            </Typography>
            <List dense>
              {Object.entries(dataSummary).map(([gesture, count]) => (
                <ListItem key={gesture}>
                  <ListItemText primary={`${gesture}: ${count} samples`} />
                </ListItem>
              ))}
            </List>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setExportDialogOpen(false)}>Cancel</Button>
            <Button onClick={exportData} variant="contained">Export</Button>
          </DialogActions>
        </Dialog>
      </CardContent>
    </Card>
  );
};

const TrainingGuide: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      label: 'Prepare Setup',
      content: 'Ensure EMG sensors are properly connected and positioned on your forearm. Clean the electrode contact area.'
    },
    {
      label: 'Choose Gesture',
      content: 'Select the gesture you want to train. Start with basic gestures like rest, fist, and open hand.'
    },
    {
      label: 'Collect Data',
      content: 'Start the training session and perform the gesture consistently. Hold the gesture for 2-3 seconds per sample.'
    },
    {
      label: 'Review & Export',
      content: 'Check the collected data distribution and export for model training when you have sufficient samples.'
    }
  ];

  return (
    <Card>
      <CardHeader
        avatar={<Timeline />}
        title="Training Guide"
        subheader="Step-by-step instructions"
      />
      <CardContent>
        <Stepper activeStep={activeStep} orientation="vertical">
          {steps.map((step, index) => (
            <Step key={step.label}>
              <StepLabel
                optional={
                  index === steps.length - 1 ? (
                    <Typography variant="caption">Last step</Typography>
                  ) : null
                }
              >
                {step.label}
              </StepLabel>
              <StepContent>
                <Typography variant="body2" paragraph>
                  {step.content}
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Button
                    variant="contained"
                    onClick={() => setActiveStep(index + 1)}
                    sx={{ mt: 1, mr: 1 }}
                    disabled={index === steps.length - 1}
                  >
                    {index === steps.length - 1 ? 'Finish' : 'Continue'}
                  </Button>
                  <Button
                    disabled={index === 0}
                    onClick={() => setActiveStep(index - 1)}
                    sx={{ mt: 1, mr: 1 }}
                  >
                    Back
                  </Button>
                </Box>
              </StepContent>
            </Step>
          ))}
        </Stepper>
        {activeStep === steps.length && (
          <Box sx={{ mt: 2, p: 2, bgcolor: 'success.light', borderRadius: 1 }}>
            <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CheckCircle /> All steps completed - you're ready to train models!
            </Typography>
            <Button onClick={() => setActiveStep(0)} sx={{ mt: 1 }}>
              Reset Guide
            </Button>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

const TrainingView: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Training Data Collection
      </Typography>
      <Typography variant="body1" color="textSecondary" paragraph>
        Collect EMG data samples for training machine learning models. 
        Follow the guide to ensure high-quality data collection.
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <TrainingSessionCard />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <TrainingGuide />
        </Grid>

        <Grid item xs={12}>
          <TrainingDataManager />
        </Grid>
      </Grid>
    </Box>
  );
};

export default TrainingView;