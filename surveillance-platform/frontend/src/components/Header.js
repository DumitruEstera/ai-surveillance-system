import React from 'react';
import { Menu, Shield, LogOut } from 'lucide-react';

const Header = ({
  onMenuClick,
  isConnected,
  onLogout,
  userName,
  userRole
}) => {
  return (
    <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6 sticky top-0 z-40">
      <div className="flex items-center gap-4">
        <button
          onClick={onMenuClick}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          title="Open Menu"
        >
          <Menu className="w-6 h-6 text-gray-600" />
        </button>
        <div
          className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
          onClick={onMenuClick}
        >
          <div className="w-8 h-8 bg-slate-100 rounded flex items-center justify-center">
            <Shield className="w-5 h-5 text-slate-700" />
          </div>
          <h1 className="text-lg font-semibold text-slate-800 hidden sm:block">Security Dashboard</h1>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Connection Status */}
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
          isConnected
            ? 'bg-green-50 text-green-700 border border-green-200'
            : 'bg-red-50 text-red-700 border border-red-200'
        }`}>
          <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>

        {/* User Info */}
        <div className="flex items-center gap-2 px-3 py-1.5">
          <span className="text-sm text-slate-600 hidden md:inline">{userName || userRole}</span>
          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
            userRole === 'admin'
              ? 'bg-purple-50 text-purple-700 border border-purple-200'
              : 'bg-slate-50 text-slate-600 border border-slate-200'
          }`}>
            {userRole === 'admin' ? 'Admin' : 'User'}
          </span>
        </div>

        {/* Logout */}
        <button
          onClick={onLogout}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          title="Logout"
        >
          <LogOut className="w-5 h-5 text-gray-600" />
        </button>
      </div>
    </header>
  );
};

export default Header;
