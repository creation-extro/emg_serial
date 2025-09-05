import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Chip
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';

interface MetricsPanelProps {
  metrics: {
    accuracy: number;
    latency: number;
    safetyEvents: number;
    totalPredictions: number;
  };
}

const MetricsPanel: React.FC<MetricsPanelProps> = ({ metrics }) => {
  // Performance targets
  const targets = {
    accuracy: 0.85, // 85%
    latency: 20, // 20ms
    safetyRate: 0.05 // 5%
  };

  // Calculate derived metrics
  const safetyRate = metrics.totalPredictions > 0 ? 
    (metrics.safetyEvents / metrics.totalPredictions) : 0;

  // Performance data for charts
  const performanceData = [
    {
      metric: 'Accuracy',
      current: Math.round(metrics.accuracy * 100),
      target: Math.round(targets.accuracy * 100),
      unit: '%'
    },
    {
      metric: 'Latency',
      current: Math.round(metrics.latency),
      target: targets.latency,
      unit: 'ms'
    },
    {
      metric: 'Safety Rate',
      current: Math.round(safetyRate * 100),
      target: Math.round(targets.safetyRate * 100),
      unit: '%'
    }
  ];

  // Status indicators
  const getStatusColor = (current: number, target: number, isInverse: boolean = false) => {
    const ratio = current / target;
    if (isInverse) {
      // For metrics like latency and safety rate, lower is better
      if (ratio <= 0.8) return '#4caf50'; // Green
      if (ratio <= 1.0) return '#ff9800'; // Orange
      return '#f44336'; // Red
    } else {
      // For metrics like accuracy, higher is better
      if (ratio >= 1.0) return '#4caf50'; // Green
      if (ratio >= 0.8) return '#ff9800'; // Orange
      return '#f44336'; // Red
    }
  };

  // Distribution data for pie chart
  const gestureDistribution = [
    { name: 'Rest', value: 35, color: '#9e9e9e' },
    { name: 'Fist', value: 20, color: '#f44336' },
    { name: 'Open', value: 25, color: '#4caf50' },
    { name: 'Pinch', value: 12, color: '#2196f3' },
    { name: 'Point', value: 8, color: '#ff9800' }
  ];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Card elevation={3} sx={{ p: 1 }}>
          <Typography variant="body2">
            <strong>{data.metric}</strong>
          </Typography>
          <Typography variant="body2">
            Current: {data.current}{data.unit}
          </Typography>
          <Typography variant="body2">
            Target: {data.target}{data.unit}
          </Typography>
        </Card>
      );
    }
    return null;
  };

  return (
    <Box>
      {/* Key Metrics Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={4}>
          <Card>
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="primary">
                {Math.round(metrics.accuracy * 100)}%
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Accuracy
              </Typography>
              <LinearProgress
                variant="determinate"
                value={Math.min(100, (metrics.accuracy / targets.accuracy) * 100)}
                sx={{ mt: 1, height: 4 }}
                color={metrics.accuracy >= targets.accuracy ? 'success' : 'warning'}
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={4}>
          <Card>
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="primary">
                {Math.round(metrics.latency)}ms
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Latency
              </Typography>
              <LinearProgress
                variant="determinate"
                value={Math.min(100, Math.max(0, 100 - (metrics.latency / targets.latency) * 100))}
                sx={{ mt: 1, height: 4 }}
                color={metrics.latency <= targets.latency ? 'success' : 'warning'}
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={4}>
          <Card>
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="primary">
                {metrics.totalPredictions}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Total Predictions
              </Typography>
              <LinearProgress
                variant="determinate"
                value={Math.min(100, (metrics.totalPredictions / 1000) * 100)}
                sx={{ mt: 1, height: 4 }}
                color="info"
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Performance Comparison Chart */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Performance vs Targets
          </Typography>
          <Box sx={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="current" fill="#2196f3" name="Current" />
                <Bar dataKey="target" fill="#ff9800" name="Target" />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>

      {/* Status Indicators */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={4}>
          <Box sx={{ textAlign: 'center' }}>
            <Chip
              label="Accuracy"
              color={metrics.accuracy >= targets.accuracy ? 'success' : 'warning'}
              variant={metrics.accuracy >= targets.accuracy ? 'filled' : 'outlined'}
            />
          </Box>
        </Grid>
        <Grid item xs={4}>
          <Box sx={{ textAlign: 'center' }}>
            <Chip
              label="Latency"
              color={metrics.latency <= targets.latency ? 'success' : 'warning'}
              variant={metrics.latency <= targets.latency ? 'filled' : 'outlined'}
            />
          </Box>
        </Grid>
        <Grid item xs={4}>
          <Box sx={{ textAlign: 'center' }}>
            <Chip
              label="Safety"
              color={safetyRate <= targets.safetyRate ? 'success' : 'warning'}
              variant={safetyRate <= targets.safetyRate ? 'filled' : 'outlined'}
            />
          </Box>
        </Grid>
      </Grid>

      {/* Gesture Distribution */}
      <Card>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Gesture Distribution
          </Typography>
          <Box sx={{ height: 150, display: 'flex', justifyContent: 'center' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={gestureDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={30}
                  outerRadius={60}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {gestureDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  formatter={(value: any) => [`${value}%`, 'Percentage']}
                />
              </PieChart>
            </ResponsiveContainer>
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: 1, mt: 1 }}>
            {gestureDistribution.map((item, index) => (
              <Chip
                key={index}
                label={`${item.name}: ${item.value}%`}
                size="small"
                sx={{ 
                  backgroundColor: item.color,
                  color: 'white',
                  fontSize: '0.7rem'
                }}
              />
            ))}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default MetricsPanel;