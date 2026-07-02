import React, { useState, useEffect, useCallback } from 'react';
import { MapPin, Plus, Trash2, Edit3, X, Check, Lock, Unlock } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const getAuthHeaders = () => {
  const token = localStorage.getItem('auth_token');
  return token
    ? { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' }
    : { 'Content-Type': 'application/json' };
};

const ZoneManagement = () => {
  const [zones, setZones] = useState([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingZone, setEditingZone] = useState(null);
  const [message, setMessage] = useState({ text: '', type: '' });

  const [newName, setNewName] = useState('');
  const [newDescription, setNewDescription] = useState('');
  const [newRestricted, setNewRestricted] = useState(false);

  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editRestricted, setEditRestricted] = useState(false);

  const showMessage = (text, type = 'success') => {
    setMessage({ text, type });
    setTimeout(() => setMessage({ text: '', type: '' }), 3500);
  };

  const fetchZones = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/zones`, { headers: getAuthHeaders() });
      if (res.ok) {
        const data = await res.json();
        setZones(data.zones || []);
      }
    } catch (e) {
      console.error('Error fetching zones:', e);
    }
  }, []);

  useEffect(() => { fetchZones(); }, [fetchZones]);

  const handleCreate = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch(`${API_BASE}/api/zones`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          name: newName,
          description: newDescription || null,
          is_restricted: newRestricted,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        showMessage(`Zone '${newName}' created`);
        setShowCreateForm(false);
        setNewName(''); setNewDescription(''); setNewRestricted(false);
        fetchZones();
      } else {
        showMessage(data.detail || 'Failed to create zone', 'error');
      }
    } catch (e) {
      showMessage('Error creating zone', 'error');
    }
  };

  const handleUpdate = async (zoneId) => {
    try {
      const res = await fetch(`${API_BASE}/api/zones/${zoneId}`, {
        method: 'PUT',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          name: editName || null,
          description: editDescription,
          is_restricted: editRestricted,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        showMessage('Zone updated');
        setEditingZone(null);
        fetchZones();
      } else {
        showMessage(data.detail || 'Failed to update zone', 'error');
      }
    } catch (e) {
      showMessage('Error updating zone', 'error');
    }
  };

  const handleDelete = async (zoneId, name) => {
    if (!window.confirm(`Delete zone "${name}"? Cameras in it will become unassigned and persons will lose this zone from their authorized list.`)) return;
    try {
      const res = await fetch(`${API_BASE}/api/zones/${zoneId}`, {
        method: 'DELETE', headers: getAuthHeaders(),
      });
      if (res.ok) {
        showMessage(`Zone '${name}' deleted`);
        fetchZones();
      } else {
        const data = await res.json();
        showMessage(data.detail || 'Failed to delete zone', 'error');
      }
    } catch (e) {
      showMessage('Error deleting zone', 'error');
    }
  };

  const startEdit = (zone) => {
    setEditingZone(zone.id);
    setEditName(zone.name);
    setEditDescription(zone.description || '');
    setEditRestricted(!!zone.is_restricted);
  };

  return (
    <div>
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <MapPin className="w-6 h-6 text-[#3374D0]" />
            <h2 className="text-xl font-semibold text-slate-800">Zone Management</h2>
          </div>
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="flex items-center gap-2 px-4 py-2 bg-[#3374D0] text-white rounded-lg hover:bg-[#2861B0] transition-colors text-sm font-medium"
          >
            <Plus className="w-4 h-4" />
            Add Zone
          </button>
        </div>
        <p className="text-sm text-gray-500">
          Group cameras into named areas. Mark a zone as restricted so only authorized employees can enter without triggering an alarm.
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
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Create New Zone</h3>
          <form onSubmit={handleCreate} className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Zone Name *</label>
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="e.g. Main Entrance, Server Room"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm"
                required
              />
            </div>
            <div className="flex items-center gap-3 pt-6">
              <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={newRestricted}
                  onChange={(e) => setNewRestricted(e.target.checked)}
                  className="w-4 h-4 accent-[#3374D0]"
                />
                Restricted zone (authorization required)
              </label>
            </div>
            <div className="sm:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
              <textarea
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                rows={2}
                placeholder="Optional notes about this zone"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm"
              />
            </div>
            <div className="sm:col-span-2 flex gap-3">
              <button
                type="submit"
                className="px-4 py-2 bg-[#3374D0] text-white rounded-lg hover:bg-[#2861B0] transition-colors text-sm font-medium"
              >
                Create Zone
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
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Description</th>
              <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Restricted</th>
              <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Cameras</th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {zones.map((z) => (
              <tr key={z.id} className="hover:bg-gray-50">
                <td className="px-6 py-4">
                  {editingZone === z.id ? (
                    <input
                      type="text"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      className="px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0] w-full max-w-[240px]"
                    />
                  ) : (
                    <div className="text-sm font-medium text-slate-800">{z.name}</div>
                  )}
                </td>
                <td className="px-6 py-4">
                  {editingZone === z.id ? (
                    <input
                      type="text"
                      value={editDescription}
                      onChange={(e) => setEditDescription(e.target.value)}
                      className="px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0] w-full"
                    />
                  ) : (
                    <span className="text-sm text-gray-600">{z.description || '-'}</span>
                  )}
                </td>
                <td className="px-6 py-4 text-center">
                  {editingZone === z.id ? (
                    <input
                      type="checkbox"
                      checked={editRestricted}
                      onChange={(e) => setEditRestricted(e.target.checked)}
                      className="w-4 h-4 accent-[#3374D0]"
                    />
                  ) : (
                    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                      z.is_restricted
                        ? 'bg-red-50 text-red-700 border border-red-200'
                        : 'bg-slate-50 text-slate-600 border border-slate-200'
                    }`}>
                      {z.is_restricted ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
                      {z.is_restricted ? 'Restricted' : 'Open'}
                    </span>
                  )}
                </td>
                <td className="px-6 py-4 text-center text-sm text-gray-700">{z.camera_count ?? 0}</td>
                <td className="px-6 py-4 text-right">
                  {editingZone === z.id ? (
                    <div className="flex items-center justify-end gap-1">
                      <button
                        onClick={() => handleUpdate(z.id)}
                        className="p-1.5 text-green-600 hover:bg-green-50 rounded transition-colors"
                        title="Save"
                      >
                        <Check className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => setEditingZone(null)}
                        className="p-1.5 text-gray-500 hover:bg-gray-100 rounded transition-colors"
                        title="Cancel"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ) : (
                    <div className="flex items-center justify-end gap-1">
                      <button
                        onClick={() => startEdit(z)}
                        className="p-1.5 text-slate-500 hover:bg-slate-100 rounded transition-colors"
                        title="Edit zone"
                      >
                        <Edit3 className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleDelete(z.id, z.name)}
                        className="p-1.5 text-red-500 hover:bg-red-50 rounded transition-colors"
                        title="Delete zone"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
            {zones.length === 0 && (
              <tr>
                <td colSpan="5" className="px-6 py-12 text-center text-gray-500">
                  <MapPin className="w-10 h-10 mx-auto mb-3 text-gray-300" />
                  <p className="text-sm">No zones defined yet. Create your first zone to start grouping cameras.</p>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ZoneManagement;
