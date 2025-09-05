import React from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Box,
  Typography,
  Divider,
  Chip,
  Avatar
} from '@mui/material';
import {
  Dashboard,
  Psychology,
  Analytics,
  Settings,
  SchoolOutlined,
  ThreeDRotation
} from '@mui/icons-material';
import { useAppStore } from '../store/useAppStore';

interface SidebarProps {
  drawerWidth: number;
  mobileOpen?: boolean;
  onMobileToggle?: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ 
  drawerWidth, 
  mobileOpen = false, 
  onMobileToggle 
}) => {
  const { currentView, setView, isConnected, metrics } = useAppStore();

  const navigationItems = [
    { id: 'dashboard' as const, label: 'Dashboard', icon: <Dashboard /> },
    { id: '3d-viz' as const, label: '3D Visualization', icon: <ThreeDRotation /> },
    { id: 'analytics' as const, label: 'Analytics', icon: <Analytics /> },
    { id: 'training' as const, label: 'Training', icon: <SchoolOutlined /> },
    { id: 'settings' as const, label: 'Settings', icon: <Settings /> }
  ];

  const drawerContent = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Toolbar sx={{ px: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Avatar sx={{ bgcolor: '#1976d2', width: 32, height: 32 }}>
            <Psychology />
          </Avatar>
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 'bold', lineHeight: 1 }}>
              Motion AI
            </Typography>
            <Typography variant="caption" color="textSecondary">
              EMG Recognition
            </Typography>
          </Box>
        </Box>
      </Toolbar>

      <Divider />

      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <List sx={{ px: 1, py: 2 }}>
          {navigationItems.map((item) => (
            <ListItem key={item.id} disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton
                selected={currentView === item.id}
                onClick={() => setView(item.id)}
                sx={{
                  borderRadius: 2,
                  '&.Mui-selected': {
                    bgcolor: 'primary.main',
                    color: 'primary.contrastText',
                    '& .MuiListItemIcon-root': { color: 'inherit' }
                  }
                }}
              >
                <ListItemIcon sx={{ minWidth: 40 }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText primary={item.label} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>

      <Divider />

      <Box sx={{ p: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 'bold' }}>
          Status
        </Typography>
        <Chip
          label={isConnected ? 'Connected' : 'Disconnected'}
          color={isConnected ? 'success' : 'error'}
          size="small"
        />
        <Box sx={{ mt: 1, fontSize: '0.7rem' }}>
          <Typography variant="caption">
            Predictions: {metrics.totalPredictions}
          </Typography>
        </Box>
      </Box>
    </Box>
  );

  return (
    <>
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={onMobileToggle}
        sx={{
          display: { xs: 'block', md: 'none' },
          '& .MuiDrawer-paper': { width: drawerWidth }
        }}
      >
        {drawerContent}
      </Drawer>

      <Drawer
        variant="permanent"
        sx={{
          display: { xs: 'none', md: 'block' },
          '& .MuiDrawer-paper': { width: drawerWidth }
        }}
        open
      >
        {drawerContent}
      </Drawer>
    </>
  );
};

export default Sidebar;