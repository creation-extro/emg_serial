import React, { useState } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  useTheme,
  useMediaQuery,
  Breadcrumbs,
  Link,
  Chip
} from '@mui/material';
import {
  Menu as MenuIcon,
  Home,
  Notifications,
  Settings as SettingsIcon,
  Dashboard,
  ThreeDRotation,
  Analytics,
  School,
  NavigateNext
} from '@mui/icons-material';
import Sidebar from './Sidebar';
import { useAppStore } from '../store/useAppStore';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);
  
  const { currentView, isConnected, isProcessing } = useAppStore();
  
  const drawerWidth = 280;

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const getViewTitle = (view: string) => {
    const titles = {
      dashboard: 'Dashboard',
      '3d-viz': '3D Visualization',
      analytics: 'Analytics',
      training: 'Training',
      settings: 'Settings'
    };
    return titles[view as keyof typeof titles] || 'Motion AI';
  };

  const getViewIcon = (view: string) => {
    const icons = {
      dashboard: <Dashboard />,
      '3d-viz': <ThreeDRotation />,
      analytics: <Analytics />,
      training: <School />,
      settings: <SettingsIcon />
    };
    return icons[view as keyof typeof icons] || <NavigateNext />;
  };

  const getBreadcrumbs = (view: string) => {
    const breadcrumbs = [
      { label: 'Motion AI', href: '/', icon: <Home sx={{ mr: 0.5 }} fontSize="inherit" /> }
    ];
    
    if (view !== 'dashboard') {
      breadcrumbs.push({ 
        label: getViewTitle(view), 
        href: `/${view}`,
        icon: React.cloneElement(getViewIcon(view), { sx: { mr: 0.5 }, fontSize: 'inherit' })
      });
    }
    
    return breadcrumbs;
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
          zIndex: theme.zIndex.drawer + 1,
          bgcolor: 'background.paper',
          color: 'text.primary',
          borderBottom: '1px solid',
          borderColor: 'divider',
          boxShadow: 'none'
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>

          <Box sx={{ flex: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
              <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
                {getViewTitle(currentView)}
              </Typography>
              {isProcessing && (
                <Chip 
                  label="Processing" 
                  color="primary" 
                  size="small"
                  variant="outlined"
                />
              )}
            </Box>
            
            <Breadcrumbs aria-label="breadcrumb" sx={{ fontSize: '0.875rem' }}>
              {getBreadcrumbs(currentView).map((item, index) => (
                <Link
                  key={index}
                  underline="hover"
                  color="inherit"
                  href={item.href}
                  sx={{ display: 'flex', alignItems: 'center' }}
                  onClick={(e) => e.preventDefault()}
                >
                  {item.icon}
                  {item.label}
                </Link>
              ))}
            </Breadcrumbs>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              label={isConnected ? 'Connected' : 'Disconnected'}
              color={isConnected ? 'success' : 'error'}
              variant="outlined"
              size="small"
            />
            
            <IconButton color="inherit">
              <Notifications />
            </IconButton>
            
            <IconButton color="inherit">
              <SettingsIcon />
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Sidebar */}
      <Sidebar
        drawerWidth={drawerWidth}
        mobileOpen={mobileOpen}
        onMobileToggle={handleDrawerToggle}
      />

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { md: `calc(100% - ${drawerWidth}px)` },
          height: '100vh',
          overflow: 'auto',
          bgcolor: 'background.default'
        }}
      >
        <Toolbar /> {/* Spacer for AppBar */}
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      </Box>
    </Box>
  );
};

export default DashboardLayout;