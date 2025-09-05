import React, { useMemo } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Divider
} from '@mui/material';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
  AreaChart,
  Area
} from 'recharts';
import {
  DataUsage,
  TrendingUp,
  Timeline,
  Assessment
} from '@mui/icons-material';
import { useAppStore } from '../store/useAppStore';

// Mock feature extraction function (in real app, this would come from backend)
const extractFeatures = (emgData: Array<{ channels: number[] }>) => {
  if (emgData.length === 0) return {};

  const features: { [key: string]: number } = {};
  const channelCount = emgData[0]?.channels?.length || 3;

  for (let ch = 0; ch < channelCount; ch++) {
    const channelData = emgData.map(d => d.channels[ch] || 0);
    
    // Time domain features
    const mean = channelData.reduce((a, b) => a + b, 0) / channelData.length;
    const variance = channelData.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / channelData.length;
    const rms = Math.sqrt(channelData.reduce((a, b) => a + b * b, 0) / channelData.length);
    const maxVal = Math.max(...channelData);
    const minVal = Math.min(...channelData);
    
    features[`ch${ch + 1}_mean`] = mean;
    features[`ch${ch + 1}_std`] = Math.sqrt(variance);
    features[`ch${ch + 1}_rms`] = rms;
    features[`ch${ch + 1}_range`] = maxVal - minVal;
    features[`ch${ch + 1}_energy`] = channelData.reduce((a, b) => a + b * b, 0);
  }

  // Cross-channel features
  if (channelCount >= 2) {
    const ch1 = emgData.map(d => d.channels[0] || 0);
    const ch2 = emgData.map(d => d.channels[1] || 0);
    
    let correlation = 0;
    const mean1 = ch1.reduce((a, b) => a + b, 0) / ch1.length;
    const mean2 = ch2.reduce((a, b) => a + b, 0) / ch2.length;
    
    let num = 0, den1 = 0, den2 = 0;
    for (let i = 0; i < ch1.length; i++) {
      const diff1 = ch1[i] - mean1;
      const diff2 = ch2[i] - mean2;
      num += diff1 * diff2;
      den1 += diff1 * diff1;
      den2 += diff2 * diff2;
    }
    
    correlation = num / Math.sqrt(den1 * den2) || 0;
    features['cross_correlation'] = correlation;
  }

  return features;
};

const FeatureRadarChart: React.FC<{ features: { [key: string]: number } }> = ({ features }) => {
  const radarData = useMemo(() => {
    const featureKeys = Object.keys(features).filter(key => 
      key.includes('mean') || key.includes('rms') || key.includes('energy')
    );
    
    return featureKeys.map(key => ({
      feature: key.replace(/ch\d+_/, '').toUpperCase(),
      value: Math.min(features[key] * 100, 100), // Normalize to 0-100
      fullName: key
    }));
  }, [features]);

  return (
    <ResponsiveContainer width="100%" height={300}>
      <RadarChart data={radarData}>
        <PolarGrid />
        <PolarAngleAxis dataKey="feature" tick={{ fontSize: 12 }} />
        <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10 }} />
        <Radar
          name="Feature Intensity"
          dataKey="value"
          stroke="#2196f3"
          fill="#2196f3"
          fillOpacity={0.3}
          strokeWidth={2}
        />
        <Tooltip 
          formatter={(value, name, props) => [
            `${typeof value === 'number' ? value.toFixed(2) : value}%`,
            props.payload?.fullName || name
          ]}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
};

const FeatureBarChart: React.FC<{ features: { [key: string]: number } }> = ({ features }) => {
  const barData = useMemo(() => {
    return Object.entries(features)
      .filter(([key]) => key.includes('std') || key.includes('range'))
      .map(([key, value]) => ({
        name: key.replace(/ch\d+_/, 'CH'),
        value: value * 1000, // Scale for visibility
        channel: key.includes('ch1') ? 'Channel 1' : 
                key.includes('ch2') ? 'Channel 2' : 'Channel 3'
      }));
  }, [features]);

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={barData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip formatter={(value) => [typeof value === 'number' ? value.toFixed(3) : value, 'Value']} />
        <Bar dataKey="value" fill="#4caf50" />
      </BarChart>
    </ResponsiveContainer>
  );
};

const FeatureTimeline: React.FC<{ 
  featureHistory: Array<{ timestamp: number; features: { [key: string]: number } }> 
}> = ({ featureHistory }) => {
  const timelineData = useMemo(() => {
    return featureHistory.slice(-20).map((entry, index) => ({
      time: index,
      ch1_rms: entry.features.ch1_rms || 0,
      ch2_rms: entry.features.ch2_rms || 0,
      ch3_rms: entry.features.ch3_rms || 0,
      timestamp: new Date(entry.timestamp).toLocaleTimeString()
    }));
  }, [featureHistory]);

  return (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={timelineData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip 
          labelFormatter={(label, payload) => 
            payload?.[0]?.payload?.timestamp || `Time ${label}`
          }
        />
        <Area 
          type="monotone" 
          dataKey="ch1_rms" 
          stackId="1" 
          stroke="#f44336" 
          fill="#f44336" 
          fillOpacity={0.6}
          name="Channel 1 RMS"
        />
        <Area 
          type="monotone" 
          dataKey="ch2_rms" 
          stackId="1" 
          stroke="#4caf50" 
          fill="#4caf50" 
          fillOpacity={0.6}
          name="Channel 2 RMS"
        />
        <Area 
          type="monotone" 
          dataKey="ch3_rms" 
          stackId="1" 
          stroke="#2196f3" 
          fill="#2196f3" 
          fillOpacity={0.6}
          name="Channel 3 RMS"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};

const FeatureExtractionVisualizer: React.FC = () => {
  const { emgData, realtimeFeatures } = useAppStore();
  
  // Extract features from recent EMG data
  const features = useMemo(() => {
    const recentData = emgData.slice(-50); // Last 50 samples
    const extractedFeatures = extractFeatures(recentData);
    
    // Merge with real-time features from store
    return { ...extractedFeatures, ...realtimeFeatures };
  }, [emgData, realtimeFeatures]);

  // Create feature history for timeline
  const featureHistory = useMemo(() => {
    const history = [];
    const windowSize = 10;
    
    for (let i = windowSize; i < emgData.length; i += 5) {
      const window = emgData.slice(i - windowSize, i);
      const windowFeatures = extractFeatures(window);
      history.push({
        timestamp: emgData[i]?.timestamp || Date.now(),
        features: windowFeatures
      });
    }
    
    return history;
  }, [emgData]);

  const getFeatureIntensity = (value: number) => {
    if (value > 0.1) return { color: 'error', label: 'High' };
    if (value > 0.05) return { color: 'warning', label: 'Medium' };
    return { color: 'success', label: 'Low' };
  };

  const primaryFeatures = [
    { key: 'ch1_rms', label: 'CH1 RMS', unit: 'mV' },
    { key: 'ch2_rms', label: 'CH2 RMS', unit: 'mV' },
    { key: 'ch3_rms', label: 'CH3 RMS', unit: 'mV' },
    { key: 'cross_correlation', label: 'Cross Correlation', unit: '' }
  ];

  return (
    <Grid container spacing={3}>
      {/* Feature Overview Cards */}
      <Grid item xs={12}>
        <Grid container spacing={2}>
          {primaryFeatures.map((feature) => {
            const value = features[feature.key] || 0;
            const intensity = getFeatureIntensity(value);
            
            return (
              <Grid item xs={12} sm={6} md={3} key={feature.key}>
                <Card variant="outlined">
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      <DataUsage color="primary" />
                      <Typography variant="subtitle2">
                        {feature.label}
                      </Typography>
                    </Box>
                    <Typography variant="h5" sx={{ fontWeight: 'bold', mb: 1 }}>
                      {value.toFixed(4)} {feature.unit}
                    </Typography>
                    <Chip 
                      label={intensity.label} 
                      color={intensity.color as any}
                      size="small"
                    />
                    <LinearProgress
                      variant="determinate"
                      value={Math.min(value * 1000, 100)}
                      color={intensity.color as any}
                      sx={{ mt: 1 }}
                    />
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      </Grid>

      {/* Radar Chart */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader
            avatar={<Assessment />}
            title="Feature Distribution"
            subheader="Radar view of feature intensities"
          />
          <CardContent>
            <FeatureRadarChart features={features} />
          </CardContent>
        </Card>
      </Grid>

      {/* Bar Chart */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader
            avatar={<TrendingUp />}
            title="Variability Features"
            subheader="Standard deviation and range by channel"
          />
          <CardContent>
            <FeatureBarChart features={features} />
          </CardContent>
        </Card>
      </Grid>

      {/* Feature Timeline */}
      <Grid item xs={12}>
        <Card>
          <CardHeader
            avatar={<Timeline />}
            title="Feature Evolution"
            subheader="RMS values over time"
          />
          <CardContent>
            <FeatureTimeline featureHistory={featureHistory} />
          </CardContent>
        </Card>
      </Grid>

      {/* Feature Details */}
      <Grid item xs={12}>
        <Card>
          <CardHeader
            title="All Features"
            subheader={`${Object.keys(features).length} features extracted`}
          />
          <CardContent>
            <Grid container spacing={1}>
              {Object.entries(features).map(([key, value]) => (
                <Grid item xs={12} sm={6} md={4} key={key}>
                  <Box sx={{ p: 1, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                    <Typography variant="caption" color="textSecondary">
                      {key}
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {value.toFixed(6)}
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default FeatureExtractionVisualizer;