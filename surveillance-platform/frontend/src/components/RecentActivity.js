import React, { useState, useEffect, useCallback } from 'react';

const API_BASE = 'http://localhost:8000';

const getAuthHeaders = () => {
  const token = localStorage.getItem('auth_token');
  return token ? { 'Authorization': `Bearer ${token}` } : {};
};

const typeLabels = {
  face: 'Unauthorized Face',
  fire: 'Fire / Smoke',
  har: 'Action Recognition',
  weapon: 'Weapon Detected',
  unauthorized_zone: 'Restricted Zone',
};

// `refreshKey` changes whenever a new live alert arrives, prompting a re-fetch
// so the feed picks up the freshly persisted alarm.
const RecentActivity = ({ onAlarmClick, refreshKey }) => {
  const [alarms, setAlarms] = useState([]);

  const fetchRecent = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/alarms?limit=10&offset=0`, { headers: getAuthHeaders() });
      const data = await res.json();
      setAlarms(data.alarms || []);
    } catch (err) {
      console.error('Error fetching recent activity:', err);
    }
  }, []);

  useEffect(() => {
    fetchRecent();
    const interval = setInterval(fetchRecent, 10000);
    return () => clearInterval(interval);
  }, [fetchRecent]);

  // Re-fetch promptly when a new live alert comes in over the WebSocket
  useEffect(() => {
    if (refreshKey) fetchRecent();
  }, [refreshKey, fetchRecent]);

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-slate-800 mb-6">Recent Activity</h3>

      <div className="space-y-2 max-h-80 overflow-y-auto pr-2">
        {alarms.length > 0 ? (
          alarms.map((alarm) => (
            <button
              key={alarm.id}
              onClick={() => onAlarmClick && onAlarmClick(alarm.id)}
              className="w-full flex gap-4 text-left rounded-lg p-2 -m-2 hover:bg-gray-50 transition-colors cursor-pointer focus:outline-none focus:ring-2 focus:ring-[#3374D0]/30"
              title="Open alarm details"
            >
              <div className={`w-2.5 h-2.5 rounded-full mt-1.5 shrink-0 ${
                alarm.status === 'unresolved' ? 'bg-red-500' : 'bg-green-500'
              }`} />
              <div className="space-y-1">
                <p className="text-sm text-slate-700 leading-tight">
                  <span className="font-semibold text-slate-900 mr-2">
                    {formatTime(alarm.created_at)}
                  </span>
                  {typeLabels[alarm.type] || alarm.description || 'System Event'}
                  {alarm.camera_id && (
                    <span className="block text-xs text-gray-400 mt-0.5">({alarm.camera_id})</span>
                  )}
                </p>
              </div>
            </button>
          ))
        ) : (
          <div className="flex gap-4">
            <div className="w-2.5 h-2.5 rounded-full mt-1.5 shrink-0 bg-green-500" />
            <div className="space-y-1">
              <p className="text-sm text-slate-700 leading-tight">
                <span className="font-semibold text-slate-900 mr-2">Now</span>
                System Operational - No recent alerts
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RecentActivity;
