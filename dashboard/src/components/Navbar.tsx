import React from 'react';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  useTheme,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  BugReport as BugIcon,
  Code as CodeIcon,
  PlayArrow as PlayIcon,
  SmartToy as AgentIcon,
} from '@mui/icons-material';

const Navbar: React.FC = () => {
  const theme = useTheme();
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: <DashboardIcon /> },
    { path: '/predictions', label: 'Predictions', icon: <BugIcon /> },
    { path: '/test-generation', label: 'Test Generation', icon: <CodeIcon /> },
    { path: '/test-execution', label: 'Test Execution', icon: <PlayIcon /> },
    { path: '/agent', label: 'AI Agent', icon: <AgentIcon /> },
  ];

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography
          variant="h6"
          component="div"
          sx={{ flexGrow: 0, mr: 4, fontWeight: 'bold' }}
        >
          AI Testing Agent
        </Typography>
        
        <Box sx={{ flexGrow: 1, display: 'flex', gap: 1 }}>
          {navItems.map((item) => (
            <Button
              key={item.path}
              component={RouterLink}
              to={item.path}
              startIcon={item.icon}
              sx={{
                color: 'white',
                backgroundColor: location.pathname === item.path 
                  ? 'rgba(255, 255, 255, 0.1)' 
                  : 'transparent',
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                },
              }}
            >
              {item.label}
            </Button>
          ))}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar; 