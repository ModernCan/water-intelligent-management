## Mini-Project 6: Water Network Digital Dashboard

### Overview
In this mini-project, you'll create a simple dashboard with React that visualizes water network status and allows natural language querying of the network. This tool will provide an intuitive interface for users to interact with your water network intelligence system.

### Learning Objectives
- Build a user-friendly interface for water network data
- Integrate LLM-powered chat for network queries
- Create simple data visualizations for network status
- Practice front-end development with React

### Dependencies
- **Phase 4 Content**: Complete at least Weeks 1-4 of the Intelligence Layer Phase
- **Skills Required**: JavaScript, React, Basic REST API integration
- **Previous Mini-Projects**: Access to API endpoints from any previous mini-project

### Estimated Time: 1 week

### Project Steps

#### Step 1: Setup Project Structure
1. Create a new React application:
```bash
npx create-react-app water-network-dashboard
cd water-network-dashboard
```

2. Install dependencies:
```bash
npm install axios recharts @mui/material @mui/icons-material react-router-dom
```

3. Create the project directory structure:
```
src/
├── components/
│   ├── Dashboard.js
│   ├── NetworkMap.js
│   ├── StatusPanel.js
│   ├── ChatInterface.js
│   └── ComponentDetails.js
├── api/
│   └── api.js
├── context/
│   └── AppContext.js
├── utils/
│   └── formatters.js
├── App.js
└── index.js
```

#### Step 2: Create API Client
1. Implement the API client in `src/api/api.js`:

```javascript
import axios from 'axios';

// Base URL for your backend API
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Network components API
export const getComponents = async (params = {}) => {
  try {
    const response = await api.get('/components', { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching components:', error);
    throw error;
  }
};

export const getComponentById = async (id) => {
  try {
    const response = await api.get(`/components/${id}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching component ${id}:`, error);
    throw error;
  }
};

// Network status API
export const getNetworkStatus = async () => {
  try {
    const response = await api.get('/status');
    return response.data;
  } catch (error) {
    console.error('Error fetching network status:', error);
    throw error;
  }
};

// Chat/query API
export const queryNetwork = async (query) => {
  try {
    const response = await api.post('/query', { query });
    return response.data;
  } catch (error) {
    console.error('Error querying network:', error);
    throw error;
  }
};

// Mock API for development
export const mockApi = {
  getComponents: async () => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return {
      components: [
        { id: 'VLV001', type: 'Valve', status: 'Open', installDate: '2015-03-15' },
        { id: 'PIP001', type: 'Pipe', material: 'PVC', diameter: 200, length: 500 },
        { id: 'JCT001', type: 'Junction', elevation: 100 },
        { id: 'TNK001', type: 'Tank', capacity: 5000, level: 0.8 },
        { id: 'PMP001', type: 'Pump', status: 'Active', flowRate: 100 }
      ]
    };
  },
  
  getComponentById: async (id) => {
    await new Promise(resolve => setTimeout(resolve, 300));
    
    const components = {
      'VLV001': { id: 'VLV001', type: 'Valve', valveType: 'Gate', diameter: 300, status: 'Open', installDate: '2015-03-15', lastMaintenance: '2022-05-10' },
      'PIP001': { id: 'PIP001', type: 'Pipe', material: 'PVC', diameter: 200, length: 500, installDate: '2015-03-12', pressure: 65 },
      'JCT001': { id: 'JCT001', type: 'Junction', elevation: 100, connections: 3 },
      'TNK001': { id: 'TNK001', type: 'Tank', capacity: 5000, level: 0.8, volume: 4000, elevation: 240 },
      'PMP001': { id: 'PMP001', type: 'Pump', status: 'Active', flowRate: 100, power: 75, efficiency: 0.85 }
    };
    
    return components[id] || { error: 'Component not found' };
  },
  
  getNetworkStatus: async () => {
    await new Promise(resolve => setTimeout(resolve, 700));
    
    return {
      overallStatus: 'Healthy',
      components: {
        total: 150,
        healthy: 142,
        warning: 5,
        critical: 3
      },
      alerts: [
        { id: 'ALT001', component: 'VLV002', type: 'Warning', message: 'Pressure drop detected', time: '2023-04-09T10:15:00Z' },
        { id: 'ALT002', component: 'PIP015', type: 'Critical', message: 'Possible leak detected', time: '2023-04-09T08:30:00Z' }
      ],
      metrics: {
        averagePressure: 68,
        totalFlow: 1250,
        waterQuality: 'Good'
      }
    };
  },
  
  queryNetwork: async (query) => {
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    return {
      query,
      answer: `Here is the answer to your query about "${query}". This is a simulated response that would come from your LLM-powered backend.`,
      results: [
        { id: 'VLV001', type: 'Valve', relevance: 0.95 },
        { id: 'PIP001', type: 'Pipe', relevance: 0.82 }
      ]
    };
  }
};

// Export both real and mock APIs
export default {
  real: {
    getComponents,
    getComponentById,
    getNetworkStatus,
    queryNetwork
  },
  mock: mockApi
};
```

#### Step 3: Create App Context
1. Implement the app context in `src/context/AppContext.js`:

```javascript
import React, { createContext, useState, useEffect } from 'react';
import api from '../api/api';

// Use mock API for development
const apiClient = api.mock;

// Create context
export const AppContext = createContext();

export const AppProvider = ({ children }) => {
  // State
  const [components, setComponents] = useState([]);
  const [networkStatus, setNetworkStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedComponent, setSelectedComponent] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);

  // Fetch initial data
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        setLoading(true);
        
        // Fetch components and network status in parallel
        const [componentsData, statusData] = await Promise.all([
          apiClient.getComponents(),
          apiClient.getNetworkStatus()
        ]);
        
        setComponents(componentsData.components);
        setNetworkStatus(statusData);
        setError(null);
      } catch (err) {
        setError('Failed to fetch network data');
        console.error('Error fetching initial data:', err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchInitialData();
  }, []);
  
  // Select a component
  const selectComponent = async (id) => {
    try {
      setLoading(true);
      const componentData = await apiClient.getComponentById(id);
      setSelectedComponent(componentData);
    } catch (err) {
      setError(`Failed to fetch component ${id}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Submit a query
  const submitQuery = async (query) => {
    try {
      // Add user message to chat history
      setChatHistory(prev => [...prev, { role: 'user', content: query }]);
      
      // Call the API
      const response = await apiClient.queryNetwork(query);
      
      // Add system response to chat history
      setChatHistory(prev => [...prev, { role: 'system', content: response.answer, results: response.results }]);
      
      return response;
    } catch (err) {
      setError('Failed to process query');
      return null;
    }
  };
  
  // Clear chat history
  const clearChat = () => {
    setChatHistory([]);
  };
  
  // Context value
  const contextValue = {
    components,
    networkStatus,
    loading,
    error,
    selectedComponent,
    chatHistory,
    selectComponent,
    submitQuery,
    clearChat
  };
  
  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
};
```

#### Step 4: Create Dashboard Components
1. Create the main dashboard in `src/components/Dashboard.js`:

```javascript
import React, { useContext } from 'react';
import { AppContext } from '../context/AppContext';
import { Container, Grid, Paper, Typography, Box, CircularProgress } from '@mui/material';
import StatusPanel from './StatusPanel';
import NetworkMap from './NetworkMap';
import ChatInterface from './ChatInterface';
import ComponentDetails from './ComponentDetails';

const Dashboard = () => {
  const { loading, error } = useContext(AppContext);
  
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }
  
  if (error) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Typography variant="h6" color="error">{error}</Typography>
      </Box>
    );
  }
  
  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Water Network Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Status Panel */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 240 }}>
            <StatusPanel />
          </Paper>
        </Grid>
        
        {/* Network Map */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 240 }}>
            <NetworkMap />
          </Paper>
        </Grid>
        
        {/* Component Details */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', minHeight: 300 }}>
            <ComponentDetails />
          </Paper>
        </Grid>
        
        {/* Chat Interface */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', minHeight: 300 }}>
            <ChatInterface />
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;
```

2. Create the status panel in `src/components/StatusPanel.js`:

```javascript
import React, { useContext } from 'react';
import { AppContext } from '../context/AppContext';
import { Typography, Box, Chip, Divider, List, ListItem, ListItemText, Alert } from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

const StatusPanel = () => {
  const { networkStatus } = useContext(AppContext);
  
  if (!networkStatus) {
    return <Typography>Status information not available</Typography>;
  }
  
  const statusColor = () => {
    switch (networkStatus.overallStatus) {
      case 'Healthy': return 'success';
      case 'Warning': return 'warning';
      case 'Critical': return 'error';
      default: return 'info';
    }
  };
  
  const { healthy, warning, critical } = networkStatus.components;
  const componentData = [
    { name: 'Healthy', value: healthy, color: '#4caf50' },
    { name: 'Warning', value: warning, color: '#ff9800' },
    { name: 'Critical', value: critical, color: '#f44336' }
  ];
  
  return (
    <Box sx={{ height: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" component="h2">
          Network Status
        </Typography>
        <Chip 
          label={networkStatus.overallStatus} 
          color={statusColor()} 
          variant="outlined" 
        />
      </Box>
      
      <Box sx={{ display: 'flex', height: '70%' }}>
        <Box sx={{ width: '50%', height: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={componentData}
                cx="50%"
                cy="50%"
                innerRadius={30}
                outerRadius={60}
                paddingAngle={1}
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}`}
              >
                {componentData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Box>
        
        <Box sx={{ width: '50%' }}>
          <Typography variant="subtitle2">Metrics</Typography>
          <List dense>
            <ListItem>
              <ListItemText 
                primary="Avg. Pressure" 
                secondary={`${networkStatus.metrics.averagePressure} psi`} 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="Total Flow" 
                secondary={`${networkStatus.metrics.totalFlow} gpm`} 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="Water Quality" 
                secondary={networkStatus.metrics.waterQuality} 
              />
            </ListItem>
          </List>
        </Box>
      </Box>
      
      {networkStatus.alerts.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Divider />
          <Typography variant="subtitle2" sx={{ mt: 1 }}>Recent Alerts</Typography>
          {networkStatus.alerts.map(alert => (
            <Alert 
              key={alert.id} 
              severity={alert.type.toLowerCase()}
              sx={{ mt: 1, py: 0 }}
            >
              {alert.message} ({alert.component})
            </Alert>
          ))}
        </Box>
      )}
    </Box>
  );
};

export default StatusPanel;
```

3. Create the network map in `src/components/NetworkMap.js`:

```javascript
import React, { useContext, useEffect, useRef } from 'react';
import { AppContext } from '../context/AppContext';
import { Typography, Box } from '@mui/material';

// This is a simplified visualization - in a real application,
// you would use a proper mapping library like Leaflet or Mapbox,
// or a graph visualization library like D3 or Cytoscape

const NetworkMap = () => {
  const { components, selectComponent } = useContext(AppContext);
  const canvasRef = useRef(null);
  
  // Map dimensions and scaling
  const width = 800;
  const height = 200;
  const nodeRadius = 8;
  
  // Generate pseudo-random positions based on component ID
  const getPosition = (id) => {
    // Create a deterministic but seemingly random position
    const hash = id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    
    return {
      x: 50 + (hash % (width - 100)),
      y: 50 + ((hash * 13) % (height - 100))
    };
  };
  
  // Get node color based on component type
  const getNodeColor = (type) => {
    switch (type) {
      case 'Valve': return '#e91e63';
      case 'Pipe': return '#2196f3';
      case 'Junction': return '#ff9800';
      case 'Tank': return '#673ab7';
      case 'Pump': return '#4caf50';
      default: return '#9e9e9e';
    }
  };
  
  // Draw the network
  useEffect(() => {
    if (!components || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Set canvas dimensions considering device pixel ratio
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);
    
    // Draw connections (simplified)
    if (components.length > 1) {
      ctx.strokeStyle = '#ccc';
      ctx.lineWidth = 1;
      
      for (let i = 0; i < components.length - 1; i++) {
        const pos1 = getPosition(components[i].id);
        const pos2 = getPosition(components[i + 1].id);
        
        ctx.beginPath();
        ctx.moveTo(pos1.x, pos1.y);
        ctx.lineTo(pos2.x, pos2.y);
        ctx.stroke();
      }
    }
    
    // Draw nodes
    components.forEach(component => {
      const pos = getPosition(component.id);
      
      // Draw node
      ctx.fillStyle = getNodeColor(component.type);
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, nodeRadius, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw label
      ctx.fillStyle = '#000';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(component.id, pos.x, pos.y + nodeRadius + 12);
    });
    
  }, [components]);
  
  // Handle canvas click for component selection
  const handleCanvasClick = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Check if any node was clicked
    for (const component of components) {
      const pos = getPosition(component.id);
      const distance = Math.sqrt((pos.x - x) ** 2 + (pos.y - y) ** 2);
      
      if (distance <= nodeRadius) {
        selectComponent(component.id);
        break;
      }
    }
  };
  
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h6" component="h2" gutterBottom>
        Network Map
      </Typography>
      
      <Box sx={{ flex: 1, overflow: 'hidden' }}>
        <canvas 
          ref={canvasRef} 
          width={width} 
          height={height} 
          onClick={handleCanvasClick}
          style={{ cursor: 'pointer' }}
        />
      </Box>
      
      <Typography variant="caption" align="center">
        Click on a component to view details
      </Typography>
    </Box>
  );
};

export default NetworkMap;
```

4. Create the component details panel in `src/components/ComponentDetails.js`:

```javascript
import React, { useContext } from 'react';
import { AppContext } from '../context/AppContext';
import { 
  Typography, 
  Box, 
  List, 
  ListItem, 
  ListItemText, 
  Divider, 
  Chip,
  Grid,
  Paper
} from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const ComponentDetails = () => {
  const { selectedComponent } = useContext(AppContext);
  
  if (!selectedComponent) {
    return (
      <Box sx={{ p: 2, display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <Typography color="text.secondary">
          Select a component from the network map to view details
        </Typography>
      </Box>
    );
  }
  
  const { id, type } = selectedComponent;
  
  // Get type-specific color
  const getTypeColor = () => {
    switch (type) {
      case 'Valve': return 'error';
      case 'Pipe': return 'primary';
      case 'Junction': return 'warning';
      case 'Tank': return 'secondary';
      case 'Pump': return 'success';
      default: return 'default';
    }
  };
  
  // Get component properties excluding id and type
  const properties = Object.entries(selectedComponent)
    .filter(([key]) => !['id', 'type'].includes(key));
  
  // Prepare data for chart
  const chartData = properties
    .filter(([_, value]) => typeof value === 'number' && value > 0)
    .map(([key, value]) => ({
      name: key,
      value
    }));
  
  return (
    <Box sx={{ height: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" component="h2">
          Component Details
        </Typography>
        <Chip label={type} color={getTypeColor()} />
      </Box>
      
      <Typography variant="h5" gutterBottom>
        {id}
      </Typography>
      
      <Grid container spacing={2}>
        <Grid item xs={6}>
          <List dense>
            {properties.map(([key, value]) => (
              <ListItem key={key}>
                <ListItemText 
                  primary={key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                  secondary={value}
                />
              </ListItem>
            ))}
          </List>
        </Grid>
        
        <Grid item xs={6}>
          {chartData.length > 0 && (
            <Paper sx={{ p: 1, height: '100%' }}>
              <Typography variant="subtitle2" align="center" gutterBottom>
                Component Metrics
              </Typography>
              <ResponsiveContainer width="100%" height={150}>
                <BarChart data={chartData}>
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default ComponentDetails;
```

5. Create the chat interface in `src/components/ChatInterface.js`:

```javascript
import React, { useContext, useState, useRef, useEffect } from 'react';
import { AppContext } from '../context/AppContext';
import { 
  Typography, 
  Box, 
  TextField, 
  Button, 
  List, 
  ListItem, 
  ListItemText, 
  Paper,
  Chip,
  IconButton,
  CircularProgress
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import DeleteIcon from '@mui/icons-material/Delete';

const ChatInterface = () => {
  const { chatHistory, submitQuery, clearChat, selectComponent } = useContext(AppContext);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  
  // Scroll to bottom when chat history changes
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatHistory]);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!query.trim()) return;
    
    setLoading(true);
    
    try {
      await submitQuery(query);
      setQuery('');
    } finally {
      setLoading(false);
    }
  };
  
  const handleResultClick = (id) => {
    selectComponent(id);
  };
  
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" component="h2">
          Network Assistant
        </Typography>
        
        <IconButton 
          size="small" 
          onClick={clearChat} 
          disabled={chatHistory.length === 0}
          title="Clear chat history"
        >
          <DeleteIcon />
        </IconButton>
      </Box>
      
      <Paper 
        sx={{ 
          flex: 1, 
          mb: 2, 
          p: 2, 
          overflow: 'auto', 
          bgcolor: '#f5f5f5',
          maxHeight: 300
        }}
      >
        {chatHistory.length === 0 ? (
          <Box sx={{ height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <Typography color="text.secondary">
              Ask a question about the water network
            </Typography>
          </Box>
        ) : (
          <List>
            {chatHistory.map((message, index) => (
              <ListItem 
                key={index} 
                alignItems="flex-start"
                sx={{ 
                  bgcolor: message.role === 'user' ? 'transparent' : '#e0e0e0',
                  borderRadius: 2,
                  mb: 1
                }}
              >
                <Box sx={{ width: '100%' }}>
                  <ListItemText 
                    primary={
                      <Typography 
                        variant="body1" 
                        component="span" 
                        sx={{ fontWeight: message.role === 'user' ? 'bold' : 'normal' }}
                      >
                        {message.role === 'user' ? 'You: ' : 'Assistant: '}
                      </Typography>
                    }
                    secondary={message.content}
                    secondaryTypographyProps={{ 
                      component: 'div',
                      variant: 'body2' 
                    }}
                  />
                  
                  {message.results && message.results.length > 0 && (
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="caption">Related Components:</Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                        {message.results.map(result => (
                          <Chip 
                            key={result.id}
                            label={`${result.id} (${result.type})`}
                            size="small"
                            onClick={() => handleResultClick(result.id)}
                            clickable
                          />
                        ))}
                      </Box>
                    </Box>
                  )}
                </Box>
              </ListItem>
            ))}
            <div ref={messagesEndRef} />
          </List>
        )}
      </Paper>
      
      <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex' }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Ask about the water network..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={loading}
          size="small"
        />
        <Button 
          type="submit" 
          variant="contained" 
          color="primary" 
          disabled={!query.trim() || loading}
          sx={{ ml: 1 }}
        >
          {loading ? <CircularProgress size={24} /> : <SendIcon />}
        </Button>
      </Box>
    </Box>
  );
};

export default ChatInterface;
```

#### Step 5: Set Up App and Routing
1. Update `src/App.js`:

```javascript
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { CssBaseline, Container, Box, AppBar, Toolbar, Typography } from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { AppProvider } from './context/AppContext';
import Dashboard from './components/Dashboard';

// Create a theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#673ab7',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppProvider>
        <Router>
          <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
            <AppBar position="static">
              <Toolbar>
                <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                  Water Network Intelligence System
                </Typography>
              </Toolbar>
            </AppBar>
            
            <Container component="main" sx={{ flexGrow: 1, py: 3 }}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
              </Routes>
            </Container>
            
            <Box component="footer" sx={{ py: 2, bgcolor: 'background.paper', textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                Water Network Dashboard • Mini-Project 6
              </Typography>
            </Box>
          </Box>
        </Router>
      </AppProvider>
    </ThemeProvider>
  );
}

export default App;
```

#### Step 6: Run and Test
1. Start the development server:
```bash
npm start
```

2. Open your browser to http://localhost:3000

3. Test the dashboard:
   - View the network map
   - Click on components to see details
   - Try using the chat interface with queries like:
     - "What is the status of VLV001?"
     - "Show me all components in the north zone"
     - "Are there any pressure issues in the network?"

### Deliverables
1. A functional React dashboard for water network management
2. Components for visualizing network status and components
3. An interactive network map visualization
4. A chat interface for natural language queries
5. Integration with backend APIs (mock or real)

### Extensions
1. Add authentication for different user roles
2. Implement a more sophisticated network visualization
3. Create a historical data view with time-series charts
4. Add a maintenance scheduling interface
5. Implement real-time updates using WebSockets
6. Create a mobile-responsive layout for field use

### Relation to Main Project
This mini-project implements the user interface aspects from Phase 4 of the main project. The dashboard you build will demonstrate how to create intuitive visualizations and interfaces for your Water Network Intelligence System. These frontend components will be essential for making your system accessible and useful to water utility staff with different roles and technical backgrounds.
