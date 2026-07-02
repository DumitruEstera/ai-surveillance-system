import React, { useState, useEffect, useCallback } from 'react';
import { Bell, CheckCircle, XCircle, AlertTriangle, Eye, ChevronLeft, ChevronRight, X } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const getAuthHeaders = () => {
  const token = localStorage.getItem('auth_token');
  return token ? { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' } : {};
};

const severityConfig = {
  critical: { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200', dot: 'bg-red-500' },
  high: { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200', dot: 'bg-orange-500' },
  medium: { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200', dot: 'bg-amber-500' },
  low: { bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-200', dot: 'bg-blue-500' },
};

const statusConfig = {
  unresolved: { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200', label: 'Unresolved' },
  resolved: { bg: 'bg-green-50', text: 'text-green-700', border: 'border-green-200', label: 'Resolved' },
  false_alarm: { bg: 'bg-gray-50', text: 'text-gray-600', border: 'border-gray-200', label: 'False Alarm' },
};

const typeLabels = {
  face: 'Unauthorized Face',
  fire: 'Fire / Smoke',
  har: 'Action Recognition',
  weapon: 'Weapon Detected',
  unauthorized_zone: 'Restricted Zone',
};

const AlarmManagement = ({ initialAlarmId = null, onInitialAlarmHandled }) => {
  const [alarms, setAlarms] = useState([]);
  const [total, setTotal] = useState(0);
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [selectedAlarm, setSelectedAlarm] = useState(null);
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [noteText, setNoteText] = useState('');
  const limit = 20;

  // Filters
  const [filterStatus, setFilterStatus] = useState('');
  const [filterType, setFilterType] = useState('');
  const [filterSeverity, setFilterSeverity] = useState('');

  const fetchAlarms = useCallback(async () => {
    try {
      const params = new URLSearchParams({ limit, offset: page * limit });
      if (filterStatus) params.append('status', filterStatus);
      if (filterType) params.append('type', filterType);
      if (filterSeverity) params.append('severity', filterSeverity);

      const res = await fetch(`${API_BASE}/api/alarms?${params}`, { headers: getAuthHeaders() });
      const data = await res.json();
      setAlarms(data.alarms || []);
      setTotal(data.total || 0);
    } catch (err) {
      console.error('Error fetching alarms:', err);
    } finally {
      setLoading(false);
    }
  }, [page, filterStatus, filterType, filterSeverity]);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/alarms/stats`, { headers: getAuthHeaders() });
      const data = await res.json();
      setStats(data);
    } catch (err) {
      console.error('Error fetching alarm stats:', err);
    }
  }, []);

  useEffect(() => {
    fetchAlarms();
    fetchStats();
  }, [fetchAlarms, fetchStats]);

  // Auto-refresh every 10s
  useEffect(() => {
    const interval = setInterval(() => {
      fetchAlarms();
      fetchStats();
    }, 10000);
    return () => clearInterval(interval);
  }, [fetchAlarms, fetchStats]);

  const updateAlarm = async (alarmId, status, notes = null) => {
    try {
      const body = { status };
      if (notes !== null) body.notes = notes;
      await fetch(`${API_BASE}/api/alarms/${alarmId}`, {
        method: 'PATCH',
        headers: getAuthHeaders(),
        body: JSON.stringify(body)
      });
      fetchAlarms();
      fetchStats();
      if (selectedAlarm && selectedAlarm.id === alarmId) {
        setSelectedAlarm(null);
      }
    } catch (err) {
      console.error('Error updating alarm:', err);
    }
  };

  const bulkUpdate = async (status) => {
    if (selectedIds.size === 0) return;
    try {
      await fetch(`${API_BASE}/api/alarms/bulk-update`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({ alarm_ids: [...selectedIds], status })
      });
      setSelectedIds(new Set());
      fetchAlarms();
      fetchStats();
    } catch (err) {
      console.error('Error bulk updating alarms:', err);
    }
  };

  const openDetail = async (alarmId) => {
    try {
      const res = await fetch(`${API_BASE}/api/alarms/${alarmId}`, { headers: getAuthHeaders() });
      const data = await res.json();
      setSelectedAlarm(data);
      setNoteText(data.notes || '');
    } catch (err) {
      console.error('Error fetching alarm detail:', err);
    }
  };

  // When navigated here from Recent Activity, auto-open that alarm's detail
  useEffect(() => {
    if (initialAlarmId) {
      openDetail(initialAlarmId);
      if (onInitialAlarmHandled) onInitialAlarmHandled();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialAlarmId]);

  const toggleSelect = (id) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === alarms.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(alarms.map(a => a.id)));
    }
  };

  const totalPages = Math.ceil(total / limit);

  return (
    <div>
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <Bell className="w-6 h-6 text-[#3374D0]" />
          <h2 className="text-xl font-semibold text-slate-800">Alarm Management</h2>
        </div>
        <p className="text-sm text-gray-500">Review, resolve, or dismiss security alarms</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
          <div className="text-2xl font-semibold text-red-600">{stats.unresolved || 0}</div>
          <div className="text-xs text-gray-500 mt-1">Unresolved</div>
        </div>
        <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
          <div className="text-2xl font-semibold text-orange-600">{stats.critical_unresolved || 0}</div>
          <div className="text-xs text-gray-500 mt-1">Critical</div>
        </div>
        <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
          <div className="text-2xl font-semibold text-green-600">{stats.resolved || 0}</div>
          <div className="text-xs text-gray-500 mt-1">Resolved</div>
        </div>
        <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
          <div className="text-2xl font-semibold text-gray-500">{stats.false_alarm || 0}</div>
          <div className="text-xs text-gray-500 mt-1">False Alarms</div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-4">
        <select
          className="px-3 py-2 rounded-lg border border-gray-200 text-sm bg-white"
          value={filterStatus}
          onChange={e => { setFilterStatus(e.target.value); setPage(0); }}
        >
          <option value="">All Statuses</option>
          <option value="unresolved">Unresolved</option>
          <option value="resolved">Resolved</option>
          <option value="false_alarm">False Alarm</option>
        </select>
        <select
          className="px-3 py-2 rounded-lg border border-gray-200 text-sm bg-white"
          value={filterType}
          onChange={e => { setFilterType(e.target.value); setPage(0); }}
        >
          <option value="">All Types</option>
          <option value="face">Face</option>
          <option value="fire">Fire</option>
          <option value="har">Action</option>
          <option value="weapon">Weapon</option>
          <option value="unauthorized_zone">Restricted Zone</option>
        </select>
        <select
          className="px-3 py-2 rounded-lg border border-gray-200 text-sm bg-white"
          value={filterSeverity}
          onChange={e => { setFilterSeverity(e.target.value); setPage(0); }}
        >
          <option value="">All Severities</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>

        {/* Bulk actions */}
        {selectedIds.size > 0 && (
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-sm text-gray-600">{selectedIds.size} selected</span>
            <button
              onClick={() => bulkUpdate('resolved')}
              className="px-3 py-2 rounded-lg text-sm font-medium bg-green-50 text-green-700 border border-green-200 hover:bg-green-100 transition-colors"
            >
              Resolve All
            </button>
            <button
              onClick={() => bulkUpdate('false_alarm')}
              className="px-3 py-2 rounded-lg text-sm font-medium bg-gray-50 text-gray-600 border border-gray-200 hover:bg-gray-100 transition-colors"
            >
              Mark False Alarm
            </button>
          </div>
        )}
      </div>

      {/* Alarm Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {/* Table Header */}
        <div className="grid grid-cols-12 gap-4 px-6 py-3 border-b border-gray-200 bg-gray-50 text-xs font-medium text-gray-500 uppercase tracking-wider">
          <div className="col-span-1 flex items-center">
            <input
              type="checkbox"
              checked={alarms.length > 0 && selectedIds.size === alarms.length}
              onChange={toggleSelectAll}
              className="rounded border-gray-300"
            />
          </div>
          <div className="col-span-2">Time</div>
          <div className="col-span-1">Type</div>
          <div className="col-span-1">Severity</div>
          <div className="col-span-3">Description</div>
          <div className="col-span-1">Camera</div>
          <div className="col-span-1">Status</div>
          <div className="col-span-2">Actions</div>
        </div>

        {/* Table Body */}
        <div className="divide-y divide-gray-100 max-h-[500px] overflow-y-auto">
          {loading ? (
            <div className="px-6 py-12 text-center text-gray-500 text-sm">Loading alarms...</div>
          ) : alarms.length === 0 ? (
            <div className="px-6 py-12 text-center text-gray-500 text-sm">No alarms found</div>
          ) : (
            alarms.map(alarm => {
              const sev = severityConfig[alarm.severity] || severityConfig.medium;
              const st = statusConfig[alarm.status] || statusConfig.unresolved;
              return (
                <div key={alarm.id} className={`grid grid-cols-12 gap-4 px-6 py-3 items-center text-sm hover:bg-gray-50 transition-colors ${alarm.status === 'unresolved' ? '' : 'opacity-70'}`}>
                  <div className="col-span-1">
                    <input
                      type="checkbox"
                      checked={selectedIds.has(alarm.id)}
                      onChange={() => toggleSelect(alarm.id)}
                      className="rounded border-gray-300"
                    />
                  </div>
                  <div className="col-span-2 text-gray-600 text-xs">
                    {new Date(alarm.created_at).toLocaleString()}
                  </div>
                  <div className="col-span-1">
                    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium border ${
                      alarm.type === 'face' ? 'bg-blue-50 text-blue-700 border-blue-200' :
                      alarm.type === 'fire' ? 'bg-red-50 text-red-700 border-red-200' :
                      alarm.type === 'har' ? 'bg-purple-50 text-purple-700 border-purple-200' :
                      alarm.type === 'unauthorized_zone' ? 'bg-amber-50 text-amber-700 border-amber-200' :
                      'bg-rose-50 text-rose-700 border-rose-200'
                    }`}>
                      {typeLabels[alarm.type] || alarm.type}
                    </span>
                  </div>
                  <div className="col-span-1">
                    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium border ${sev.bg} ${sev.text} ${sev.border}`}>
                      <span className={`w-1.5 h-1.5 rounded-full ${sev.dot}`}></span>
                      {alarm.severity}
                    </span>
                  </div>
                  <div className="col-span-3 text-slate-700 truncate" title={alarm.description}>
                    {alarm.description}
                  </div>
                  <div className="col-span-1 text-gray-500 text-xs">{alarm.camera_id}</div>
                  <div className="col-span-1">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${st.bg} ${st.text} ${st.border}`}>
                      {st.label}
                    </span>
                  </div>
                  <div className="col-span-2 flex items-center gap-1">
                    <button
                      onClick={() => openDetail(alarm.id)}
                      className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
                      title="View details"
                    >
                      <Eye className="w-4 h-4 text-gray-500" />
                    </button>
                    {alarm.status === 'unresolved' && (
                      <>
                        <button
                          onClick={() => updateAlarm(alarm.id, 'resolved')}
                          className="p-1.5 rounded-lg hover:bg-green-50 transition-colors"
                          title="Resolve"
                        >
                          <CheckCircle className="w-4 h-4 text-green-600" />
                        </button>
                        <button
                          onClick={() => updateAlarm(alarm.id, 'false_alarm')}
                          className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
                          title="Mark as false alarm"
                        >
                          <XCircle className="w-4 h-4 text-gray-400" />
                        </button>
                      </>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-6 py-3 border-t border-gray-200 bg-gray-50">
            <span className="text-xs text-gray-500">
              Showing {page * limit + 1}-{Math.min((page + 1) * limit, total)} of {total}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                className="p-1.5 rounded-lg hover:bg-gray-200 disabled:opacity-40 transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-xs text-gray-600">Page {page + 1} of {totalPages}</span>
              <button
                onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="p-1.5 rounded-lg hover:bg-gray-200 disabled:opacity-40 transition-colors"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Alarm Detail Modal */}
      {selectedAlarm && (
        <>
          <div className="fixed inset-0 bg-black/30 backdrop-blur-sm z-50" onClick={() => setSelectedAlarm(null)} />
          <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
              {/* Modal Header */}
              <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
                <div className="flex items-center gap-3">
                  <AlertTriangle className={`w-5 h-5 ${
                    selectedAlarm.severity === 'critical' ? 'text-red-500' :
                    selectedAlarm.severity === 'high' ? 'text-orange-500' :
                    'text-amber-500'
                  }`} />
                  <h3 className="text-lg font-semibold text-slate-800">Alarm Details</h3>
                  <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${
                    (statusConfig[selectedAlarm.status] || statusConfig.unresolved).bg
                  } ${(statusConfig[selectedAlarm.status] || statusConfig.unresolved).text} ${
                    (statusConfig[selectedAlarm.status] || statusConfig.unresolved).border
                  }`}>
                    {(statusConfig[selectedAlarm.status] || statusConfig.unresolved).label}
                  </span>
                </div>
                <button onClick={() => setSelectedAlarm(null)} className="p-2 hover:bg-gray-100 rounded-lg">
                  <X className="w-5 h-5 text-gray-500" />
                </button>
              </div>

              <div className="p-6 space-y-5">
                {/* Snapshot */}
                {selectedAlarm.snapshot && (
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2 block">Snapshot at Detection</label>
                    <img
                      src={`data:image/jpeg;base64,${selectedAlarm.snapshot}`}
                      alt="Alarm snapshot"
                      className="w-full rounded-lg border border-gray-200"
                    />
                  </div>
                )}

                {/* Info Grid */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Type</label>
                    <div className="mt-1 text-sm font-medium text-slate-800">{typeLabels[selectedAlarm.type] || selectedAlarm.type}</div>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Severity</label>
                    <div className="mt-1">
                      <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium border ${
                        (severityConfig[selectedAlarm.severity] || severityConfig.medium).bg
                      } ${(severityConfig[selectedAlarm.severity] || severityConfig.medium).text} ${
                        (severityConfig[selectedAlarm.severity] || severityConfig.medium).border
                      }`}>
                        {selectedAlarm.severity}
                      </span>
                    </div>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Camera</label>
                    <div className="mt-1 text-sm text-slate-800">{selectedAlarm.camera_id}</div>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Created</label>
                    <div className="mt-1 text-sm text-slate-800">{new Date(selectedAlarm.created_at).toLocaleString()}</div>
                  </div>
                </div>

                {/* Description */}
                <div>
                  <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Description</label>
                  <div className="mt-1 text-sm text-slate-700">{selectedAlarm.description}</div>
                </div>

                {/* Detection Metadata */}
                {selectedAlarm.detection_metadata && (
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Detection Data</label>
                    <div className="mt-1 bg-gray-50 rounded-lg p-3 text-xs font-mono text-gray-600">
                      {Object.entries(
                        typeof selectedAlarm.detection_metadata === 'string'
                          ? JSON.parse(selectedAlarm.detection_metadata)
                          : selectedAlarm.detection_metadata
                      ).map(([k, v]) => (
                        <div key={k}><span className="text-gray-800 font-medium">{k}:</span> {JSON.stringify(v)}</div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Resolved info */}
                {selectedAlarm.resolved_at && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Resolved At</label>
                      <div className="mt-1 text-sm text-slate-800">{new Date(selectedAlarm.resolved_at).toLocaleString()}</div>
                    </div>
                    <div>
                      <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Resolved By</label>
                      <div className="mt-1 text-sm text-slate-800">{selectedAlarm.resolved_by || '-'}</div>
                    </div>
                  </div>
                )}

                {/* Notes */}
                <div>
                  <label className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1 block">Notes</label>
                  <textarea
                    className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-[#3374D0]/30 focus:border-[#3374D0]"
                    rows={3}
                    value={noteText}
                    onChange={e => setNoteText(e.target.value)}
                    placeholder="Add notes about this alarm..."
                  />
                </div>

                {/* Actions */}
                <div className="flex items-center gap-3 pt-2">
                  {selectedAlarm.status === 'unresolved' && (
                    <>
                      <button
                        onClick={() => updateAlarm(selectedAlarm.id, 'resolved', noteText || null)}
                        className="px-4 py-2 rounded-lg text-sm font-medium bg-green-600 text-white hover:bg-green-700 transition-colors"
                      >
                        Resolve
                      </button>
                      <button
                        onClick={() => updateAlarm(selectedAlarm.id, 'false_alarm', noteText || null)}
                        className="px-4 py-2 rounded-lg text-sm font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 transition-colors"
                      >
                        False Alarm
                      </button>
                    </>
                  )}
                  {noteText !== (selectedAlarm.notes || '') && (
                    <button
                      onClick={() => updateAlarm(selectedAlarm.id, null, noteText)}
                      className="px-4 py-2 rounded-lg text-sm font-medium bg-[#3374D0] text-white hover:bg-[#2860B0] transition-colors"
                    >
                      Save Notes
                    </button>
                  )}
                  <button
                    onClick={() => setSelectedAlarm(null)}
                    className="px-4 py-2 rounded-lg text-sm font-medium text-gray-600 hover:bg-gray-100 transition-colors ml-auto"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default AlarmManagement;
