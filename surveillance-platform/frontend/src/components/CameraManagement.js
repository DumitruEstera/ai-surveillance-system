import React, { useState, useEffect, useCallback } from 'react';
import { Video, Plus, Trash2, Edit3, X, Check, Lock, Unlock } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const getAuthHeaders = () => {
  const token = localStorage.getItem('auth_token');
  return token
    ? { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' }
    : { 'Content-Type': 'application/json' };
};

const CameraManagement = () => {
  const [cameras, setCameras] = useState([]);
  const [zones, setZones] = useState([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingCamera, setEditingCamera] = useState(null);
  const [message, setMessage] = useState({ text: '', type: '' });

  const [newCameraId, setNewCameraId] = useState('');
  const [newName, setNewName] = useState('');
  const [newLocation, setNewLocation] = useState('');
  const [newZoneId, setNewZoneId] = useState('');
  const [newStreamUrl, setNewStreamUrl] = useState('');
  const [newType, setNewType] = useState('ip');

  const [editName, setEditName] = useState('');
  const [editLocation, setEditLocation] = useState('');
  const [editZoneId, setEditZoneId] = useState('');
  const [editStreamUrl, setEditStreamUrl] = useState('');

  const showMessage = (text, type = 'success') => {
    setMessage({ text, type });
    setTimeout(() => setMessage({ text: '', type: '' }), 3500);
  };

  const fetchAll = useCallback(async () => {
    try {
      const [camRes, zoneRes] = await Promise.all([
        fetch(`${API_BASE}/api/cameras`, { headers: getAuthHeaders() }),
        fetch(`${API_BASE}/api/zones`, { headers: getAuthHeaders() }),
      ]);
      if (camRes.ok) {
        const data = await camRes.json();
        setCameras(data.cameras || []);
      }
      if (zoneRes.ok) {
        const data = await zoneRes.json();
        setZones(data.zones || []);
      }
    } catch (e) {
      console.error('Error fetching cameras/zones:', e);
    }
  }, []);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  const handleCreate = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch(`${API_BASE}/api/cameras`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          camera_id: newCameraId.trim(),
          name: newName || null,
          location: newLocation || null,
          zone_id: newZoneId ? parseInt(newZoneId, 10) : null,
          stream_url: newStreamUrl || null,
          camera_type: newType,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        showMessage(`Camera '${newCameraId}' saved`);
        setShowCreateForm(false);
        setNewCameraId(''); setNewName(''); setNewLocation('');
        setNewZoneId(''); setNewStreamUrl(''); setNewType('ip');
        fetchAll();
      } else {
        showMessage(data.detail || 'Failed to save camera', 'error');
      }
    } catch (e) {
      showMessage('Error saving camera', 'error');
    }
  };

  const handleUpdate = async (cameraId) => {
    try {
      const body = {
        name: editName || null,
        location: editLocation || null,
        stream_url: editStreamUrl || null,
      };
      if (editZoneId === '') {
        body.clear_zone = true;
      } else {
        body.zone_id = parseInt(editZoneId, 10);
      }
      const res = await fetch(`${API_BASE}/api/cameras/${encodeURIComponent(cameraId)}`, {
        method: 'PUT',
        headers: getAuthHeaders(),
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (res.ok) {
        showMessage('Camera updated');
        setEditingCamera(null);
        fetchAll();
      } else {
        showMessage(data.detail || 'Failed to update camera', 'error');
      }
    } catch (e) {
      showMessage('Error updating camera', 'error');
    }
  };

  const handleDelete = async (cameraId) => {
    if (!window.confirm(`Delete camera "${cameraId}" from the registry? (This does not disconnect the live stream.)`)) return;
    try {
      const res = await fetch(`${API_BASE}/api/cameras/${encodeURIComponent(cameraId)}`, {
        method: 'DELETE', headers: getAuthHeaders(),
      });
      if (res.ok) {
        showMessage(`Camera '${cameraId}' deleted`);
        fetchAll();
      } else {
        const data = await res.json();
        showMessage(data.detail || 'Failed to delete camera', 'error');
      }
    } catch (e) {
      showMessage('Error deleting camera', 'error');
    }
  };

  const startEdit = (camera) => {
    setEditingCamera(camera.camera_id);
    setEditName(camera.name || '');
    setEditLocation(camera.location || '');
    setEditZoneId(camera.zone_id != null ? String(camera.zone_id) : '');
    setEditStreamUrl(camera.stream_url || '');
  };

  return (
    <div>
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <Video className="w-6 h-6 text-[#3374D0]" />
            <h2 className="text-xl font-semibold text-slate-800">Camera Management</h2>
          </div>
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="flex items-center gap-2 px-4 py-2 bg-[#3374D0] text-white rounded-lg hover:bg-[#2861B0] transition-colors text-sm font-medium"
          >
            <Plus className="w-4 h-4" />
            Add Camera
          </button>
        </div>
        <p className="text-sm text-gray-500">
          Register cameras and assign them to zones. IP cameras you connect from the Dashboard are added automatically.
        </p>
      </div>

      {message.text && (
        <div className={`mb-4 p-3 rounded-lg text-sm ${
          message.type === 'error'
            ? 'bg-red-50 border border-red-200 text-red-700'
            : 'bg-green-50 border border-green-200 text-green-700'
        }`}>
          {message.text}
        </div>
      )}

      {showCreateForm && (
        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm mb-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Add / Update Camera</h3>
          <form onSubmit={handleCreate} className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Camera ID *</label>
              <input
                type="text"
                value={newCameraId}
                onChange={(e) => setNewCameraId(e.target.value)}
                placeholder="e.g. CAM-03"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0]"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Friendly Name</label>
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="e.g. Front Door"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0]"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
              <input
                type="text"
                value={newLocation}
                onChange={(e) => setNewLocation(e.target.value)}
                placeholder="e.g. Lobby"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0]"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Zone</label>
              <select
                value={newZoneId}
                onChange={(e) => setNewZoneId(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0]"
              >
                <option value="">— No zone —</option>
                {zones.map(z => (
                  <option key={z.id} value={z.id}>
                    {z.name}{z.is_restricted ? ' (restricted)' : ''}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Stream URL</label>
              <input
                type="text"
                value={newStreamUrl}
                onChange={(e) => setNewStreamUrl(e.target.value)}
                placeholder="http://192.168.1.5:8080/video"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0]"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Type</label>
              <select
                value={newType}
                onChange={(e) => setNewType(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0]"
              >
                <option value="ip">IP Camera</option>
                <option value="local">Local (webcam)</option>
              </select>
            </div>
            <div className="sm:col-span-2 flex gap-3">
              <button
                type="submit"
                className="px-4 py-2 bg-[#3374D0] text-white rounded-lg hover:bg-[#2861B0] transition-colors text-sm font-medium"
              >
                Save Camera
              </button>
              <button
                type="button"
                onClick={() => setShowCreateForm(false)}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Camera ID</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name / Location</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Zone</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {cameras.map((c) => (
              <tr key={c.camera_id} className="hover:bg-gray-50">
                <td className="px-6 py-4 text-sm font-mono font-medium text-slate-800">{c.camera_id}</td>
                <td className="px-6 py-4">
                  {editingCamera === c.camera_id ? (
                    <div className="flex flex-col gap-1">
                      <input
                        type="text"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        placeholder="Name"
                        className="px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0]"
                      />
                      <input
                        type="text"
                        value={editLocation}
                        onChange={(e) => setEditLocation(e.target.value)}
                        placeholder="Location"
                        className="px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0]"
                      />
                      <input
                        type="text"
                        value={editStreamUrl}
                        onChange={(e) => setEditStreamUrl(e.target.value)}
                        placeholder="Stream URL"
                        className="px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0]"
                      />
                    </div>
                  ) : (
                    <div>
                      <div className="text-sm text-slate-800">{c.name || c.location || '-'}</div>
                      {c.location && c.location !== c.name && (
                        <div className="text-xs text-gray-500">{c.location}</div>
                      )}
                      {c.stream_url && (
                        <div className="text-xs text-gray-400 font-mono truncate max-w-[280px]">{c.stream_url}</div>
                      )}
                    </div>
                  )}
                </td>
                <td className="px-6 py-4">
                  {editingCamera === c.camera_id ? (
                    <select
                      value={editZoneId}
                      onChange={(e) => setEditZoneId(e.target.value)}
                      className="px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0]"
                    >
                      <option value="">— No zone —</option>
                      {zones.map(z => (
                        <option key={z.id} value={z.id}>
                          {z.name}{z.is_restricted ? ' (restricted)' : ''}
                        </option>
                      ))}
                    </select>
                  ) : c.zone_name ? (
                    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                      c.zone_is_restricted
                        ? 'bg-red-50 text-red-700 border border-red-200'
                        : 'bg-blue-50 text-blue-700 border border-blue-200'
                    }`}>
                      {c.zone_is_restricted ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
                      {c.zone_name}
                    </span>
                  ) : (
                    <span className="text-sm text-gray-400">Unassigned</span>
                  )}
                </td>
                <td className="px-6 py-4 text-sm text-gray-600 capitalize">{c.camera_type || 'ip'}</td>
                <td className="px-6 py-4 text-right">
                  {editingCamera === c.camera_id ? (
                    <div className="flex items-center justify-end gap-1">
                      <button
                        onClick={() => handleUpdate(c.camera_id)}
                        className="p-1.5 text-green-600 hover:bg-green-50 rounded transition-colors"
                        title="Save"
                      >
                        <Check className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => setEditingCamera(null)}
                        className="p-1.5 text-gray-500 hover:bg-gray-100 rounded transition-colors"
                        title="Cancel"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ) : (
                    <div className="flex items-center justify-end gap-1">
                      <button
                        onClick={() => startEdit(c)}
                        className="p-1.5 text-slate-500 hover:bg-slate-100 rounded transition-colors"
                        title="Edit camera"
                      >
                        <Edit3 className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleDelete(c.camera_id)}
                        className="p-1.5 text-red-500 hover:bg-red-50 rounded transition-colors"
                        title="Delete from registry"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
            {cameras.length === 0 && (
              <tr>
                <td colSpan="5" className="px-6 py-12 text-center text-gray-500">
                  <Video className="w-10 h-10 mx-auto mb-3 text-gray-300" />
                  <p className="text-sm">No cameras registered yet.</p>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default CameraManagement;
