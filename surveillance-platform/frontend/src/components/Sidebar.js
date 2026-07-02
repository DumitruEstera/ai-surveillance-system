import React, { useState, useEffect } from 'react';
import { LayoutDashboard, UserCheck, Car, FileText, BarChart3, X, Users, Bell, MapPin, Video } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const allNavItems = [
  { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard', adminOnly: false },
  { id: 'alarms', icon: Bell, label: 'Alarms', adminOnly: false },
  { id: 'persons', icon: UserCheck, label: 'Person Management', adminOnly: true },
  { id: 'zones', icon: MapPin, label: 'Zone Management', adminOnly: true },
  { id: 'cameras', icon: Video, label: 'Camera Management', adminOnly: true },
  { id: 'plates', icon: Car, label: 'Plate Management', adminOnly: true },
  { id: 'logs', icon: FileText, label: 'Logs', adminOnly: false },
  { id: 'stats', icon: BarChart3, label: 'Statistics', adminOnly: true },
  { id: 'users', icon: Users, label: 'User Management', adminOnly: true },
];

const Sidebar = ({ isOpen, onClose, activeTab, onNavigate, userRole }) => {
  const [unresolvedCount, setUnresolvedCount] = useState(0);

  useEffect(() => {
    const fetchCount = async () => {
      try {
        const token = localStorage.getItem('auth_token');
        if (!token) return;
        const res = await fetch(`${API_BASE}/api/alarms/stats`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await res.json();
        setUnresolvedCount(data.unresolved || 0);
      } catch {}
    };
    fetchCount();
    const interval = setInterval(fetchCount, 10000);
    return () => clearInterval(interval);
  }, []);

  const isAdmin = userRole === 'admin';
  const navItems = allNavItems.filter(item => !item.adminOnly || isAdmin);
  if (!isOpen) return null;

  const handleNavigate = (tabId) => {
    onNavigate(tabId);
    onClose();
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/20 backdrop-blur-sm z-50 transition-opacity"
        onClick={onClose}
      />

      {/* Sidebar Panel */}
      <div className="fixed top-0 left-0 h-full w-[280px] bg-white border-r border-gray-200 z-[60] flex flex-col pt-20 shadow-2xl sidebar-slide-in">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-4 p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <X className="w-5 h-5 text-gray-500" />
        </button>

        <nav className="flex-1 px-4 space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleNavigate(item.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                  isActive
                    ? 'bg-[#EBF2FF] text-[#3374D0] font-medium'
                    : 'text-gray-600 hover:bg-gray-50'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span>{item.label}</span>
                {item.id === 'alarms' && unresolvedCount > 0 && (
                  <span className="ml-auto px-2 py-0.5 rounded-full text-xs font-semibold bg-red-500 text-white min-w-[20px] text-center">
                    {unresolvedCount > 99 ? '99+' : unresolvedCount}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>
    </>
  );
};

export default Sidebar;
