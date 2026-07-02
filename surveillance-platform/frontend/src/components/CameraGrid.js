import React, { useState } from 'react';
import { X, Video, VideoOff, Plus, Smartphone, Monitor } from 'lucide-react';

const CameraGrid = ({ cameras = [], alerts = [], onAddIpCamera, onRemoveIpCamera, systemStatus, onStartCamera, onStopCamera }) => {
  const cam01Streaming = systemStatus?.cam01_streaming || false;
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [showConnectForm, setShowConnectForm] = useState(false);
  const [ip, setIp] = useState('');
  const [port, setPort] = useState('8080');
  const [cameraName, setCameraName] = useState('');
  const [connecting, setConnecting] = useState(false);

  // CAM-01 is always the laptop; the rest are IP cameras
  const cam01 = cameras.find(c => c.id === 'CAM-01') || { id: 'CAM-01', status: 'inactive', stream: null, location: 'Laptop Camera' };
  const ipCameras = cameras.filter(c => c.id !== 'CAM-01');

  // Generate next camera ID based on existing cameras
  const getNextCameraId = () => {
    const existingNums = cameras
      .map(c => parseInt(c.id.replace('CAM-', ''), 10))
      .filter(n => !isNaN(n));
    const next = Math.max(1, ...existingNums) + 1;
    return `CAM-${String(next).padStart(2, '0')}`;
  };

  const hasAlert = (cameraId) => {
    return alerts.some(alert => {
      const alertTime = new Date(alert.timestamp);
      const now = new Date();
      return alert.cameraId === cameraId && (now - alertTime) < 24 * 60 * 60 * 1000;
    });
  };

  const now = new Date();
  const timestamp = now.toISOString().replace('T', ' ').substring(0, 19);

  const handleConnect = async () => {
    if (!ip.trim() || !port.trim()) return;
    const url = `http://${ip.trim()}:${port.trim()}/video`;
    const cameraId = getNextCameraId();
    const location = cameraName.trim() || 'IP Camera';
    setConnecting(true);
    try {
      await onAddIpCamera(url, cameraId, location);
      setShowConnectForm(false);
      setIp('');
      setPort('8080');
      setCameraName('');
    } catch (err) {
      // error handled upstream
    } finally {
      setConnecting(false);
    }
  };

  const renderCameraSlot = (camera) => {
    const isActive = camera.status === 'active' && camera.stream;
    const hasActiveAlert = hasAlert(camera.id);
    const isIpCamera = camera.id !== 'CAM-01';
    const isLaptopCam = camera.id === 'CAM-01';

    return (
      <div
        key={camera.id}
        className="bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm group cursor-pointer"
        onClick={() => isActive && setSelectedCamera(camera)}
      >
        <div className="relative aspect-video bg-black">
          {isActive ? (
            <img
              src={camera.stream}
              alt={`Camera ${camera.id}`}
              className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity"
            />
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-gray-500">
              <VideoOff className="w-10 h-10 mb-3 opacity-40" />
              <span className="text-sm opacity-60">Offline</span>
            </div>
          )}

          {hasActiveAlert && (
            <div className="absolute top-3 right-3 w-3 h-3 bg-red-500 rounded-full animate-pulse" />
          )}

          {/* Camera ID badge */}
          <div className="absolute top-3 left-3">
            <span className="px-2 py-1 bg-black/50 backdrop-blur-md text-white text-[10px] font-medium rounded uppercase tracking-wider">
              {camera.id}
            </span>
          </div>

          {/* Timestamp */}
          {isActive && !isLaptopCam && (
            <div className="absolute top-3 right-3">
              <span className="px-2 py-1 bg-black/50 backdrop-blur-md text-white text-[10px] font-mono rounded">
                {timestamp}
              </span>
            </div>
          )}

          {/* Start/Stop button for laptop camera */}
          {isLaptopCam && onStartCamera && onStopCamera && (
            <button
              onClick={(e) => { e.stopPropagation(); cam01Streaming ? onStopCamera() : onStartCamera(); }}
              className={`absolute top-3 right-3 z-10 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                cam01Streaming
                  ? 'bg-red-500/90 text-white hover:bg-red-600'
                  : 'bg-[#3374D0] text-white hover:bg-[#2861B0]'
              }`}
              title={cam01Streaming ? 'Stop Laptop Camera' : 'Start Laptop Camera'}
            >
              <Monitor className="w-3.5 h-3.5" />
              <span>{cam01Streaming ? 'Stop' : 'Start'}</span>
            </button>
          )}

          {/* Disconnect button for IP cameras */}
          {isIpCamera && (
            <button
              onClick={(e) => { e.stopPropagation(); onRemoveIpCamera(camera.id); }}
              className="absolute top-3 right-3 z-10 px-2 py-1 bg-red-500/80 backdrop-blur-md text-white text-[10px] font-medium rounded hover:bg-red-600/90 transition-colors uppercase tracking-wider"
            >
              Disconnect
            </button>
          )}

          {/* Location label */}
          <div className="absolute bottom-3 left-3">
            <span className="px-2 py-1 bg-black/50 backdrop-blur-md text-white text-[10px] font-medium rounded uppercase tracking-wider">
              {camera.location}
            </span>
          </div>

          {/* Status dot */}
          <div className={`absolute bottom-3 right-3 w-2.5 h-2.5 rounded-full ${
            isActive ? 'bg-green-500' : 'bg-gray-500'
          }`} />
        </div>
      </div>
    );
  };

  const renderAddCameraSlot = () => (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm">
      <div className="relative aspect-video bg-black flex items-center justify-center">
        {showConnectForm ? (
          <div className="flex flex-col items-center gap-4 p-6" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-2 text-white mb-1">
              <Smartphone className="w-5 h-5" />
              <span className="text-sm font-medium">Connect IP Camera</span>
            </div>

            <div className="flex flex-col gap-3 w-full max-w-xs">
              <div>
                <label className="text-gray-400 text-xs mb-1 block">Camera Name (optional)</label>
                <input
                  type="text"
                  value={cameraName}
                  onChange={(e) => setCameraName(e.target.value)}
                  placeholder="e.g. Phone Camera"
                  className="w-full px-3 py-2 text-sm bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-[#3374D0]"
                />
              </div>
              <div>
                <label className="text-gray-400 text-xs mb-1 block">IP Address</label>
                <input
                  type="text"
                  value={ip}
                  onChange={(e) => setIp(e.target.value)}
                  placeholder="192.168.1.100"
                  className="w-full px-3 py-2 text-sm bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-[#3374D0]"
                  autoFocus
                  onKeyDown={(e) => e.key === 'Enter' && handleConnect()}
                />
              </div>
              <div>
                <label className="text-gray-400 text-xs mb-1 block">Port</label>
                <input
                  type="text"
                  value={port}
                  onChange={(e) => setPort(e.target.value)}
                  placeholder="8080"
                  className="w-full px-3 py-2 text-sm bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-[#3374D0]"
                  onKeyDown={(e) => e.key === 'Enter' && handleConnect()}
                />
              </div>
            </div>

            <div className="flex gap-2 mt-1">
              <button
                onClick={handleConnect}
                disabled={connecting || !ip.trim()}
                className="px-4 py-2 rounded-lg text-sm font-medium bg-[#3374D0] text-white hover:bg-[#2861B0] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {connecting ? 'Connecting...' : 'Connect'}
              </button>
              <button
                onClick={() => { setShowConnectForm(false); setIp(''); setPort('8080'); setCameraName(''); }}
                className="px-4 py-2 rounded-lg text-sm font-medium bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-gray-500">
            <button
              onClick={() => setShowConnectForm(true)}
              className="w-16 h-16 rounded-full bg-gray-800 hover:bg-gray-700 border-2 border-dashed border-gray-600 hover:border-[#3374D0] flex items-center justify-center transition-all group"
            >
              <Plus className="w-8 h-8 text-gray-500 group-hover:text-[#3374D0] transition-colors" />
            </button>
            <span className="text-sm opacity-60">Add IP Camera</span>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-slate-800">Camera Grid</h2>
        <span className="text-sm text-gray-500">{cameras.filter(c => c.status === 'active' && c.stream).length} / {cameras.length} active</span>
      </div>

      {/* Dynamic camera grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {renderCameraSlot(cam01)}
        {ipCameras.map(cam => renderCameraSlot(cam))}
        {renderAddCameraSlot()}
      </div>

      {/* Camera Modal */}
      {selectedCamera && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedCamera(null)}
        >
          <div
            className="bg-white rounded-xl shadow-2xl w-full max-w-4xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
              <div className="flex items-center gap-3">
                <Video className="w-5 h-5 text-[#3374D0]" />
                <h2 className="text-lg font-semibold text-slate-800">{selectedCamera.id}</h2>
                <span className="text-sm text-gray-500">- {selectedCamera.location || 'Unknown'}</span>
              </div>
              <button
                onClick={() => setSelectedCamera(null)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>
            <div className="bg-black aspect-video">
              {(() => {
                const liveCamera = cameras.find(cam => cam.id === selectedCamera.id);
                const stream = liveCamera?.stream || selectedCamera.stream;
                return stream ? (
                  <img
                    src={stream}
                    alt={`Camera ${selectedCamera.id}`}
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-500">
                    <p>No stream available</p>
                  </div>
                );
              })()}
            </div>
            <div className="px-6 py-4 border-t border-gray-200 flex gap-6 text-sm">
              <div>
                <span className="text-gray-500">Status: </span>
                <span className={`font-medium ${
                  (cameras.find(cam => cam.id === selectedCamera.id)?.status || selectedCamera.status) === 'active'
                    ? 'text-green-600' : 'text-gray-500'
                }`}>
                  {(cameras.find(cam => cam.id === selectedCamera.id)?.status || selectedCamera.status).toUpperCase()}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Location: </span>
                <span className="font-medium text-slate-700">{selectedCamera.location || 'Not specified'}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CameraGrid;
