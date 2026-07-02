import React, { useState, useEffect, useCallback } from 'react';
import './App.css';

// Components
import Login from './components/Login';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import CameraGrid from './components/CameraGrid';
import IntelligenceSettings from './components/IntelligenceSettings';
import RecentActivity from './components/RecentActivity';
import PersonManagement from './components/PersonManagement';
import PlateManagement from './components/PlateManagement';
import Logs from './components/Logs';
import AlarmManagement from './components/AlarmManagement';
import Statistics from './components/Statistics';
import UserManagement from './components/UserManagement';
import ZoneManagement from './components/ZoneManagement';
import CameraManagement from './components/CameraManagement';

// API base URL
const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

// Initial camera — only the laptop webcam is always present
const INITIAL_CAMERAS = [
  { id: 'CAM-01', status: 'inactive', stream: null, location: 'Laptop Camera' },
];

// Helper to get auth headers for API calls
const getAuthHeaders = () => {
  const token = localStorage.getItem('auth_token');
  return token ? { 'Authorization': `Bearer ${token}` } : {};
};

function App() {
  // Auth state
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    return !!localStorage.getItem('auth_token');
  });
  const [userRole, setUserRole] = useState(() => {
    try {
      const user = JSON.parse(localStorage.getItem('auth_user') || '{}');
      return user.role || 'user';
    } catch { return 'user'; }
  });
  const [userName, setUserName] = useState(() => {
    try {
      const user = JSON.parse(localStorage.getItem('auth_user') || '{}');
      return user.full_name || user.username || '';
    } catch { return ''; }
  });
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const isAdmin = userRole === 'admin';

  // App state
  const [cameras, setCameras] = useState(INITIAL_CAMERAS);
  const [alerts, setAlerts] = useState([]);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [openAlarmId, setOpenAlarmId] = useState(null);
  const [alarmRefreshKey, setAlarmRefreshKey] = useState(0);
  const [systemStatus, setSystemStatus] = useState({});
  const [isConnected, setIsConnected] = useState(false);
  const [recentLogs, setRecentLogs] = useState([]);
  const [ws, setWs] = useState(null);
  const [faceDetectionEnabled, setFaceDetectionEnabled] = useState(true);
  const [plateDetectionEnabled, setPlateDetectionEnabled] = useState(true);
  const [demographicsEnabled, setDemographicsEnabled] = useState(true);
  const [fireDetectionEnabled, setFireDetectionEnabled] = useState(true);
  const [harEnabled, setHarEnabled] = useState(true);
  const [weaponDetectionEnabled, setWeaponDetectionEnabled] = useState(true);

  // WebSocket connection
  useEffect(() => {
    if (!isAuthenticated) return;

    const connectWebSocket = () => {
      // The backend authenticates the WebSocket via a ?token=<JWT> query param
      // (browsers can't set Authorization headers on a WebSocket).
      const token = localStorage.getItem('auth_token');
      if (!token) return;
      const websocket = new WebSocket(`${WS_URL}?token=${encodeURIComponent(token)}`);

      websocket.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
      };

      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'video_frame') {
          const cameraId = data.camera_id || 'CAM-01';
          
          // Update the matching camera slot with the frame, or add it if new
          setCameras(prevCameras => {
            const exists = prevCameras.some(c => c.id === cameraId);
            if (exists) {
              return prevCameras.map(camera =>
                camera.id === cameraId
                  ? { ...camera, stream: `data:image/jpeg;base64,${data.frame}`, status: 'active' }
                  : camera
              );
            }
            // New camera arrived via WebSocket — add it
            return [...prevCameras, {
              id: cameraId,
              status: 'active',
              stream: `data:image/jpeg;base64,${data.frame}`,
              location: `IP Camera`
            }];
          });

          // Process alerts from detection results
          const newAlerts = [];

          // Face detection alerts
          if (data.face_results) {
            data.face_results.forEach(result => {
              if (result.name === 'Unknown') {
                newAlerts.push({
                  cameraId: cameraId,
                  type: 'unauthorized',
                  severity: 'critical',
                  timestamp: new Date().toISOString(),
                  description: `Unauthorized person detected with ${result.confidence}% confidence`
                });
              }
            });
          }

          // Fire detection alerts — only surface confirmed + alert-raising detections,
          // matching the backend gate for alarms and drawn bounding boxes.
          if (data.fire_results && data.fire_results.length > 0) {
            const firedAlerts = data.fire_results.filter(r => r.confirmed && r.alert);
            firedAlerts.forEach(r => {
              const cls = (r.class || 'fire').toUpperCase();
              newAlerts.push({
                cameraId: cameraId,
                type: 'fire',
                severity: r.class === 'fire' ? 'critical' : 'high',
                timestamp: new Date().toISOString(),
                description: `${cls} detected (${(r.confidence * 100).toFixed(0)}% confidence)`
              });
            });
          }

          // HAR alerts
          if (data.har_results && data.har_results.length > 0) {
            const harDetections = data.har_results.filter(r => r.class !== 'normal');
            harDetections.forEach(det => {
              newAlerts.push({
                cameraId: cameraId,
                type: 'har',
                severity: det.severity || 'high',
                timestamp: new Date().toISOString(),
                description: `${det.action_label || det.class.toUpperCase()} detected (${(det.confidence * 100).toFixed(0)}% confidence)`
              });
            });
          }

          // Weapon detection alerts
          if (data.weapon_results && data.weapon_results.length > 0) {
            data.weapon_results.forEach(det => {
              newAlerts.push({
                cameraId: cameraId,
                type: 'weapon',
                severity: det.severity || 'critical',
                timestamp: new Date().toISOString(),
                description: `${det.class ? det.class.toUpperCase() : 'WEAPON'} detected (${(det.confidence * 100).toFixed(0)}% confidence)`
              });
            });
          }

          // Update alerts
          if (newAlerts.length > 0) {
            setAlerts(prevAlerts => [...newAlerts, ...prevAlerts].slice(0, 100));
            // Nudge the Recent Activity feed to re-fetch the newly persisted alarm
            setAlarmRefreshKey(k => k + 1);
          }

          // Update recent logs
          const newLogs = [];
          if (data.face_results) {
            data.face_results.forEach(result => {
              newLogs.push({ type: 'face', ...result, timestamp: data.timestamp });
            });
          }
          if (data.plate_results) {
            data.plate_results.forEach(result => {
              newLogs.push({ type: 'plate', ...result, timestamp: data.timestamp });
            });
          }
          if (data.har_results) {
            data.har_results.forEach(result => {
              if (result.class !== 'normal') {
                newLogs.push({ type: 'har', ...result, timestamp: data.timestamp });
              }
            });
          }
          if (data.weapon_results) {
            data.weapon_results.forEach(result => {
              newLogs.push({ type: 'weapon', ...result, timestamp: data.timestamp });
            });
          }

          if (newLogs.length > 0) {
            setRecentLogs(prevLogs => [...newLogs, ...prevLogs].slice(0, 200));
          }
        }
      };

      websocket.onclose = (event) => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        // 1008 = policy violation: the server rejected our token (missing/
        // invalid/expired). Don't hammer it with reconnects — the user needs
        // to re-authenticate.
        if (event.code === 1008) {
          console.warn('WebSocket auth rejected; not reconnecting.');
          return;
        }
        setTimeout(connectWebSocket, 3000);
      };

      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      setWs(websocket);
    };

    connectWebSocket();

    return () => {
      if (ws) ws.close();
    };
  }, [isAuthenticated]);

  // Fetch system status
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/status`, {
        headers: getAuthHeaders()
      });
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Error fetching status:', error);
    }
  }, []);

  useEffect(() => {
    if (!isAuthenticated) return;
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchStatus, isAuthenticated]);

  // API functions
  const startCamera = async () => {
    try {
      await fetch(`${API_BASE}/api/streams`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ source: 'local' })
      });
      fetchStatus();
    } catch (error) {
      console.error('Error starting camera:', error);
    }
  };

  const stopCamera = async () => {
    try {
      await fetch(`${API_BASE}/api/streams/CAM-01`, { method: 'DELETE', headers: getAuthHeaders() });
      fetchStatus();
    } catch (error) {
      console.error('Error stopping camera:', error);
    }
  };

  const addIpCamera = async (url, cameraId, location) => {
    try {
      const response = await fetch(`${API_BASE}/api/streams`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ source: 'ip', url, camera_id: cameraId, location })
      });
      const data = await response.json();
      if (data.status === 'success') {
        // Add the camera slot so it appears immediately while waiting for frames
        setCameras(prevCameras => {
          const exists = prevCameras.some(c => c.id === cameraId);
          if (exists) return prevCameras;
          return [...prevCameras, { id: cameraId, status: 'inactive', stream: null, location }];
        });
      }
      fetchStatus();
      return data;
    } catch (error) {
      console.error('Error adding IP camera:', error);
      throw error;
    }
  };

  const removeIpCamera = async (cameraId) => {
    try {
      await fetch(`${API_BASE}/api/streams/${encodeURIComponent(cameraId)}`, {
        method: 'DELETE',
        headers: getAuthHeaders()
      });
      // Remove the camera from the list entirely
      setCameras(prevCameras => prevCameras.filter(c => c.id !== cameraId));
      fetchStatus();
    } catch (error) {
      console.error('Error removing IP camera:', error);
    }
  };

  const toggleFaceDetection = async (enabled) => {
    try {
      await fetch(`${API_BASE}/api/face/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ enabled })
      });
      setFaceDetectionEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling face detection:', error);
    }
  };

  const togglePlateDetection = async (enabled) => {
    try {
      await fetch(`${API_BASE}/api/plate/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ enabled })
      });
      setPlateDetectionEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling plate detection:', error);
    }
  };

  const toggleDemographics = async (enabled) => {
    try {
      await fetch(`${API_BASE}/api/demographics/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ enabled })
      });
      setDemographicsEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling demographics:', error);
    }
  };

  const toggleFireDetection = async (enabled) => {
    try {
      await fetch(`${API_BASE}/api/fire/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ enabled })
      });
      setFireDetectionEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling fire detection:', error);
    }
  };

  const toggleHar = async (enabled) => {
    try {
      await fetch(`${API_BASE}/api/har/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ enabled })
      });
      setHarEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling HAR:', error);
    }
  };

  const toggleWeaponDetection = async (enabled) => {
    try {
      await fetch(`${API_BASE}/api/weapon/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ enabled })
      });
      setWeaponDetectionEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling weapon detection:', error);
    }
  };

  // Open a specific alarm's detail from the Recent Activity feed
  const handleOpenAlarm = (alarmId) => {
    setOpenAlarmId(alarmId);
    setActiveTab('alarms');
  };

  const handleLogin = (user) => {
    setIsAuthenticated(true);
    setUserRole(user.role || 'user');
    setUserName(user.full_name || user.username || '');
  };

  const handleLogout = async () => {
    // Revoke the token server-side first (best-effort). If this fails — e.g.
    // network error — we still clear local state and the token expires naturally.
    try {
      await fetch(`${API_BASE}/api/logout`, {
        method: 'POST',
        headers: getAuthHeaders(),
      });
    } catch (error) {
      console.error('Error revoking token on logout:', error);
    }
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');
    setIsAuthenticated(false);
    setUserRole('user');
    setUserName('');
    if (ws) ws.close();
  };

  // Request notification permission
  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  // Show login if not authenticated
  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <div className="min-h-screen bg-[#F0F2F5] font-sans text-slate-900">
      <Header
        onMenuClick={() => setIsSidebarOpen(true)}
        isConnected={isConnected}
        onLogout={handleLogout}
        userName={userName}
        userRole={userRole}
      />

      <Sidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        activeTab={activeTab}
        onNavigate={setActiveTab}
        userRole={userRole}
      />

      <main className="p-6">
        {activeTab === 'dashboard' && (
          <div className="max-w-[1600px] mx-auto grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Main Content - Camera Grid */}
            <div className="lg:col-span-3">
              <CameraGrid
                cameras={cameras}
                alerts={alerts}
                onAddIpCamera={addIpCamera}
                onRemoveIpCamera={removeIpCamera}
                systemStatus={systemStatus}
                onStartCamera={startCamera}
                onStopCamera={stopCamera}
              />
            </div>

            {/* Right Sidebar - Settings & Activity */}
            <div className="space-y-6">
              <IntelligenceSettings
                faceDetectionEnabled={faceDetectionEnabled}
                plateDetectionEnabled={plateDetectionEnabled}
                demographicsEnabled={demographicsEnabled}
                fireDetectionEnabled={fireDetectionEnabled}
                harEnabled={harEnabled}
                weaponDetectionEnabled={weaponDetectionEnabled}
                onToggleFaceDetection={toggleFaceDetection}
                onTogglePlateDetection={togglePlateDetection}
                onToggleDemographics={toggleDemographics}
                onToggleFireDetection={toggleFireDetection}
                onToggleHar={toggleHar}
                onToggleWeaponDetection={toggleWeaponDetection}
                isAdmin={isAdmin}
              />
              <RecentActivity onAlarmClick={handleOpenAlarm} refreshKey={alarmRefreshKey} />
            </div>
          </div>
        )}

        {activeTab === 'persons' && isAdmin && (
          <div className="max-w-[1600px] mx-auto">
            <PersonManagement />
          </div>
        )}

        {activeTab === 'alarms' && (
          <div className="max-w-[1600px] mx-auto">
            <AlarmManagement
              initialAlarmId={openAlarmId}
              onInitialAlarmHandled={() => setOpenAlarmId(null)}
            />
          </div>
        )}

        {activeTab === 'plates' && isAdmin && (
          <PlateManagement />
        )}

        {activeTab === 'logs' && (
          <div className="max-w-[1600px] mx-auto">
            <Logs />
          </div>
        )}

        {activeTab === 'stats' && isAdmin && (
          <div className="max-w-[1600px] mx-auto">
            <Statistics systemStatus={systemStatus} />
          </div>
        )}

        {activeTab === 'users' && isAdmin && (
          <div className="max-w-[1600px] mx-auto">
            <UserManagement />
          </div>
        )}

        {activeTab === 'zones' && isAdmin && (
          <div className="max-w-[1600px] mx-auto">
            <ZoneManagement />
          </div>
        )}

        {activeTab === 'cameras' && isAdmin && (
          <div className="max-w-[1600px] mx-auto">
            <CameraManagement />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
