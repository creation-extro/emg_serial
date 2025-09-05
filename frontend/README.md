# Motion AI - React Frontend

This is the React frontend for the Motion AI EMG Gesture Recognition System. It provides a modern web interface to interact with the Motion AI backend API.

## Features

- **Real-time EMG Visualization**: Live multi-channel EMG signal display
- **Gesture Recognition Dashboard**: Current gesture display with confidence levels
- **Safety Monitoring**: Real-time safety system status and alerts
- **Performance Metrics**: System performance indicators and statistics
- **Interactive Controls**: Start/stop live processing, system configuration
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Technology Stack

- **React 18** with TypeScript
- **Material-UI (MUI)** for component library
- **Recharts** for data visualization
- **Axios** for API communication
- **React Hooks** for state management

## Components

### Core Components

1. **App.tsx** - Main application component with system status management
2. **EMGVisualizer.tsx** - Real-time EMG signal chart component
3. **GestureClassifier.tsx** - Gesture display and recognition interface
4. **SafetyMonitor.tsx** - Safety system status and alerts
5. **MetricsPanel.tsx** - Performance metrics and statistics

### Services

- **MotionAIService.ts** - API client for Motion AI backend communication

## Installation

```bash
# Install dependencies
npm install

# Install additional packages
npm install axios recharts @types/react-router-dom react-router-dom @emotion/react @emotion/styled @mui/material @mui/icons-material
```

## Development

```bash
# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test
```

## Configuration

The frontend is configured to connect to the Motion AI backend at `http://localhost:8000`. You can modify this in `src/services/MotionAIService.ts`.

## API Integration

The frontend communicates with the Motion AI backend through the following endpoints:

- `GET /health` - System health check
- `POST /v1/classify` - Gesture classification
- `POST /v1/policy` - Policy generation
- `POST /v1/hybrid` - End-to-end processing

## Usage

1. **Start the Backend**: Ensure the Motion AI FastAPI backend is running on port 8000
2. **Start the Frontend**: Run `npm start` to start the React development server
3. **Open Browser**: Navigate to `http://localhost:3000`
4. **Connect**: The app will automatically attempt to connect to the backend
5. **Start Processing**: Toggle the "Live Processing Mode" switch to begin real-time processing

## Features Overview

### Dashboard Layout

- **Header**: System status and connection indicator
- **Control Panel**: Live processing controls and system status
- **EMG Visualizer**: Real-time multi-channel EMG signal display
- **Gesture Classifier**: Current gesture and confidence level
- **Metrics Panel**: Performance statistics and charts
- **Safety Monitor**: Safety system status and alerts

### Real-time Features

- **Live EMG Data**: Displays real-time EMG signals from 3 channels
- **Gesture Recognition**: Shows current detected gesture with confidence
- **Safety Alerts**: Real-time safety system monitoring
- **Performance Metrics**: Live accuracy, latency, and safety statistics

### Interactive Elements

- **Start/Stop Processing**: Toggle live mode on/off
- **Gesture Legend**: Visual guide to available gestures
- **Safety Configuration**: View current safety thresholds
- **Metrics Charts**: Interactive performance visualization

## Customization

### Adding New Gestures

1. Update the gesture configuration in `GestureClassifier.tsx`
2. Add corresponding icons and colors
3. Update the gesture patterns in `MotionAIService.ts`

### Modifying Charts

1. Edit the chart components in respective files
2. Customize colors, scales, and layouts in the component props
3. Add new chart types using Recharts library

### Styling

1. Modify the Material-UI theme in `index.tsx`
2. Update component-specific styles in `App.css`
3. Add custom CSS classes as needed

## Deployment

### Production Build

```bash
npm run build
```

This creates a `build` folder with optimized production files.

### Docker Deployment

```dockerfile
# Use Node.js base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Install serve to run the app
RUN npm install -g serve

# Expose port
EXPOSE 3000

# Start the app
CMD ["serve", "-s", "build", "-l", "3000"]
```

## Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_WEBSOCKET_URL=ws://localhost:8000/ws
REACT_APP_VERSION=1.0.0
```

## Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   - Ensure Motion AI backend is running on port 8000
   - Check CORS configuration in the backend
   - Verify network connectivity

2. **Charts Not Displaying**
   - Check that Recharts is properly installed
   - Verify data format matches chart expectations
   - Check browser console for JavaScript errors

3. **Real-time Updates Not Working**
   - Verify WebSocket connection (if implemented)
   - Check that live processing is enabled
   - Ensure proper state management in React components

### Development Tips

1. Use React Developer Tools for debugging
2. Monitor network requests in browser DevTools
3. Check console logs for API communication
4. Use the built-in error boundaries for error handling

## Contributing

1. Follow the existing code structure and naming conventions
2. Add TypeScript types for all new interfaces
3. Include proper error handling in components
4. Test responsive design on different screen sizes
5. Update this README for any new features

## License

This project is part of the Motion AI EMG Gesture Recognition System.