import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { Box, Typography, Paper } from '@mui/material';

interface EMGData {
  timestamp: number;
  channels: number[];
  gesture?: string;
  confidence?: number;
}

interface EMGVisualizerProps {
  data: EMGData[];
  height?: number;
}

const EMGVisualizer: React.FC<EMGVisualizerProps> = ({ 
  data, 
  height = 300 
}) => {
  // Transform data for the chart
  const chartData = data.map((point, index) => ({
    time: index,
    timestamp: new Date(point.timestamp).toLocaleTimeString(),
    ch1: point.channels[0] || 0,
    ch2: point.channels[1] || 0,
    ch3: point.channels[2] || 0,
    gesture: point.gesture,
    confidence: point.confidence
  }));

  // Custom tooltip to show gesture information
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="body2">
            <strong>Time:</strong> {data.timestamp}
          </Typography>
          <Typography variant="body2">
            <strong>Channel 1:</strong> {data.ch1.toFixed(3)}
          </Typography>
          <Typography variant="body2">
            <strong>Channel 2:</strong> {data.ch2.toFixed(3)}
          </Typography>
          <Typography variant="body2">
            <strong>Channel 3:</strong> {data.ch3.toFixed(3)}
          </Typography>
          {data.gesture && (
            <Typography variant="body2">
              <strong>Gesture:</strong> {data.gesture} ({(data.confidence * 100).toFixed(1)}%)
            </Typography>
          )}
        </Paper>
      );
    }
    return null;
  };

  if (data.length === 0) {
    return (
      <Box 
        sx={{ 
          height, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          backgroundColor: '#f5f5f5',
          borderRadius: 1
        }}
      >
        <Typography variant="body1" color="textSecondary">
          No EMG data available. Start live processing to see real-time signals.
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={chartData}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis 
            dataKey="time" 
            tick={{ fontSize: 12 }}
            stroke="#666"
          />
          <YAxis 
            domain={[0, 1]}
            tick={{ fontSize: 12 }}
            stroke="#666"
            label={{ value: 'Amplitude', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Line
            type="monotone"
            dataKey="ch1"
            stroke="#2196f3"
            strokeWidth={2}
            dot={false}
            name="Channel 1"
            connectNulls={false}
          />
          <Line
            type="monotone"
            dataKey="ch2"
            stroke="#4caf50"
            strokeWidth={2}
            dot={false}
            name="Channel 2"
            connectNulls={false}
          />
          <Line
            type="monotone"
            dataKey="ch3"
            stroke="#ff9800"
            strokeWidth={2}
            dot={false}
            name="Channel 3"
            connectNulls={false}
          />
        </LineChart>
      </ResponsiveContainer>
      
      <Box sx={{ mt: 1, display: 'flex', justifyContent: 'center', gap: 2 }}>
        <Typography variant="caption" color="textSecondary">
          Sampling Rate: 1000 Hz | Window: 200ms | Channels: 3
        </Typography>
      </Box>
    </Box>
  );
};

export default EMGVisualizer;