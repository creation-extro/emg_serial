import React from 'react';
import {
  Grid,
  Typography,
  Box,
  Tabs,
  Tab,
  Card,
  CardContent
} from '@mui/material';
import { 
  Analytics as AnalyticsIcon,
  DataUsage,
  Timeline,
  Assessment
} from '@mui/icons-material';
import FeatureExtractionVisualizer from '../components/FeatureExtractionVisualizer';
import MetricsPanel from '../components/MetricsPanel';
import { useAppStore } from '../store/useAppStore';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analytics-tabpanel-${index}`}
      aria-labelledby={`analytics-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ py: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const AnalyticsView: React.FC = () => {
  const [tabValue, setTabValue] = React.useState(0);
  const { metrics } = useAppStore();

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Analytics Dashboard
      </Typography>
      <Typography variant="body1" color="textSecondary" paragraph>
        Comprehensive analysis of EMG signal processing, feature extraction, and system performance.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            aria-label="analytics tabs"
          >
            <Tab 
              icon={<DataUsage />}
              label="Feature Analysis" 
              id="analytics-tab-0"
              aria-controls="analytics-tabpanel-0"
            />
            <Tab 
              icon={<Assessment />}
              label="Performance Metrics" 
              id="analytics-tab-1"
              aria-controls="analytics-tabpanel-1"
            />
            <Tab 
              icon={<Timeline />}
              label="Signal Quality" 
              id="analytics-tab-2"
              aria-controls="analytics-tabpanel-2"
            />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <FeatureExtractionVisualizer />
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <MetricsPanel metrics={metrics} />
            </Grid>
            
            {/* Additional performance visualizations */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    System Performance Summary
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" color="textSecondary">
                      Total Predictions: {metrics.totalPredictions.toLocaleString()}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Average Latency: {metrics.latency.toFixed(2)}ms
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Safety Events: {metrics.safetyEvents}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Average Confidence: {(metrics.averageConfidence * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Gesture Distribution
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    {Object.entries(metrics.gestureDistribution).map(([gesture, count]) => (
                      <Typography key={gesture} variant="body2" color="textSecondary">
                        {gesture}: {count} detections
                      </Typography>
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Signal Quality Analysis
                  </Typography>
                  <Typography variant="body1" paragraph>
                    Signal quality metrics and analysis tools will be displayed here.
                    This includes noise detection, signal-to-noise ratio, electrode contact quality,
                    and drift detection results.
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Feature under development...
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Card>
    </Box>
  );
};

export default AnalyticsView;