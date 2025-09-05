import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  TextField,
  Switch,
  FormControlLabel,
  Button,
  Divider,
  Box,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Tune,
  Save,
  RestoreOutlined,
  Wifi
} from '@mui/icons-material';
import { useAppStore } from '../store/useAppStore';

const SettingsView: React.FC = () => {
  const { settings, updateSettings } = useAppStore();
  const [localSettings, setLocalSettings] = React.useState(settings);
  const [hasChanges, setHasChanges] = React.useState(false);

  React.useEffect(() => {
    setHasChanges(JSON.stringify(settings) !== JSON.stringify(localSettings));
  }, [settings, localSettings]);

  const handleSave = () => {
    updateSettings(localSettings);
    setHasChanges(false);
  };

  const handleReset = () => {
    setLocalSettings(settings);
    setHasChanges(false);
  };

  const updateLocalSetting = (key: keyof typeof settings, value: any) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }));
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      <Typography variant="body1" color="textSecondary" paragraph>
        Configure system parameters and preferences for optimal EMG signal processing.
      </Typography>

      {hasChanges && (
        <Alert severity="info" sx={{ mb: 3 }}>
          You have unsaved changes. Don't forget to save your settings.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Signal Processing Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              avatar={<Tune />}
              title="Signal Processing"
              subheader="EMG signal acquisition and processing parameters"
            />
            <CardContent>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                <TextField
                  label="Sampling Rate (Hz)"
                  type="number"
                  value={localSettings.samplingRate}
                  onChange={(e) => updateLocalSetting('samplingRate', parseInt(e.target.value))}
                  inputProps={{ min: 100, max: 2000 }}
                  helperText="Higher rates provide better signal quality but increase processing load"
                />

                <Box>
                  <Typography gutterBottom>Window Size (ms): {localSettings.windowSize}</Typography>
                  <Slider
                    value={localSettings.windowSize}
                    onChange={(_, value) => updateLocalSetting('windowSize', value)}
                    min={50}
                    max={500}
                    step={50}
                    marks={[
                      { value: 100, label: '100ms' },
                      { value: 200, label: '200ms' },
                      { value: 300, label: '300ms' }
                    ]}
                  />
                </Box>

                <Box>
                  <Typography gutterBottom>Hop Size (ms): {localSettings.hopSize}</Typography>
                  <Slider
                    value={localSettings.hopSize}
                    onChange={(_, value) => updateLocalSetting('hopSize', value)}
                    min={25}
                    max={200}
                    step={25}
                    marks={[
                      { value: 50, label: '50ms' },
                      { value: 100, label: '100ms' },
                      { value: 150, label: '150ms' }
                    ]}
                  />
                </Box>

                <Box>
                  <Typography gutterBottom>Confidence Threshold: {localSettings.confidenceThreshold}</Typography>
                  <Slider
                    value={localSettings.confidenceThreshold}
                    onChange={(_, value) => updateLocalSetting('confidenceThreshold', value)}
                    min={0.1}
                    max={1.0}
                    step={0.05}
                    marks={[
                      { value: 0.5, label: '0.5' },
                      { value: 0.7, label: '0.7' },
                      { value: 0.9, label: '0.9' }
                    ]}
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Display Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              avatar={<SettingsIcon />}
              title="Display & UI"
              subheader="Interface and visualization preferences"
            />
            <CardContent>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  label="Max Data Points"
                  type="number"
                  value={localSettings.maxDataPoints}
                  onChange={(e) => updateLocalSetting('maxDataPoints', parseInt(e.target.value))}
                  inputProps={{ min: 100, max: 5000 }}
                  helperText="Maximum number of data points to display in charts"
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={localSettings.enableFeatureViz}
                      onChange={(e) => updateLocalSetting('enableFeatureViz', e.target.checked)}
                    />
                  }
                  label="Enable Feature Visualization"
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={localSettings.enable3DViz}
                      onChange={(e) => updateLocalSetting('enable3DViz', e.target.checked)}
                    />
                  }
                  label="Enable 3D Visualization"
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={localSettings.autoSave}
                      onChange={(e) => updateLocalSetting('autoSave', e.target.checked)}
                    />
                  }
                  label="Auto-save Training Data"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Connection Settings */}
        <Grid item xs={12}>
          <Card>
            <CardHeader
              avatar={<Wifi />}
              title="Connection Settings"
              subheader="Backend API and WebSocket configuration"
            />
            <CardContent>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="API Base URL"
                    value="http://localhost:8000"
                    disabled
                    fullWidth
                    helperText="Backend API endpoint (read-only)"
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="WebSocket URL"
                    value="ws://localhost:8000"
                    disabled
                    fullWidth
                    helperText="WebSocket endpoint for real-time data (read-only)"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Action Buttons */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                <Button
                  variant="outlined"
                  startIcon={<RestoreOutlined />}
                  onClick={handleReset}
                  disabled={!hasChanges}
                >
                  Reset Changes
                </Button>
                <Button
                  variant="contained"
                  startIcon={<Save />}
                  onClick={handleSave}
                  disabled={!hasChanges}
                >
                  Save Settings
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Current Settings Summary */}
        <Grid item xs={12}>
          <Card>
            <CardHeader
              title="Current Configuration"
              subheader="Summary of active settings"
            />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="textSecondary">Sampling Rate</Typography>
                  <Typography variant="body2">{settings.samplingRate} Hz</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="textSecondary">Window Size</Typography>
                  <Typography variant="body2">{settings.windowSize} ms</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="textSecondary">Hop Size</Typography>
                  <Typography variant="body2">{settings.hopSize} ms</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="textSecondary">Confidence Threshold</Typography>
                  <Typography variant="body2">{settings.confidenceThreshold}</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="textSecondary">Max Data Points</Typography>
                  <Typography variant="body2">{settings.maxDataPoints}</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="textSecondary">Feature Viz</Typography>
                  <Typography variant="body2">{settings.enableFeatureViz ? 'Enabled' : 'Disabled'}</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="textSecondary">3D Viz</Typography>
                  <Typography variant="body2">{settings.enable3DViz ? 'Enabled' : 'Disabled'}</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="textSecondary">Auto Save</Typography>
                  <Typography variant="body2">{settings.autoSave ? 'Enabled' : 'Disabled'}</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SettingsView;