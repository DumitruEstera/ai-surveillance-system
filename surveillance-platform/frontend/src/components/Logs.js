import React, { useState, useEffect, useCallback } from 'react';
import { FileText, Download, ChevronLeft, ChevronRight, X, Search, RefreshCw } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const getAuthHeaders = () => {
  const token = localStorage.getItem('auth_token');
  return token ? { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' } : {};
};

const typeConfig = {
  face:   { label: 'Face',   bg: 'bg-blue-50',   text: 'text-blue-700',   border: 'border-blue-200' },
  plate:  { label: 'Plate',  bg: 'bg-amber-50',  text: 'text-amber-700',  border: 'border-amber-200' },
  fire:   { label: 'Fire',   bg: 'bg-red-50',    text: 'text-red-700',    border: 'border-red-200' },
  har:    { label: 'Action', bg: 'bg-purple-50', text: 'text-purple-700', border: 'border-purple-200' },
  weapon: { label: 'Weapon', bg: 'bg-rose-50',   text: 'text-rose-700',   border: 'border-rose-200' },
};

const severityConfig = {
  critical: { bg: 'bg-red-50',    text: 'text-red-700',    border: 'border-red-200',    dot: 'bg-red-500' },
  high:     { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200', dot: 'bg-orange-500' },
  medium:   { bg: 'bg-amber-50',  text: 'text-amber-700',  border: 'border-amber-200',  dot: 'bg-amber-500' },
  low:      { bg: 'bg-blue-50',   text: 'text-blue-700',   border: 'border-blue-200',   dot: 'bg-blue-500' },
};

const statusBadge = (log) => {
  const s = log.status;
  if (!s) return <span className="text-gray-400">-</span>;
  const map = {
    unknown:      { cls: 'bg-red-50 text-red-700 border-red-200',       label: 'Unknown' },
    recognized:   { cls: 'bg-green-50 text-green-700 border-green-200', label: 'Recognized' },
    authorized:   { cls: 'bg-green-50 text-green-700 border-green-200', label: 'Authorized' },
    unauthorized: { cls: 'bg-red-50 text-red-700 border-red-200',       label: 'Unauthorized' },
    alert:        { cls: 'bg-red-50 text-red-700 border-red-200',       label: 'ALERT' },
    detected:     { cls: 'bg-amber-50 text-amber-700 border-amber-200', label: 'Detected' },
    threat:       { cls: 'bg-red-50 text-red-700 border-red-200',       label: 'THREAT' },
    critical:     { cls: 'bg-red-50 text-red-700 border-red-200',       label: 'CRITICAL' },
    high:         { cls: 'bg-orange-50 text-orange-700 border-orange-200', label: 'HIGH' },
    medium:       { cls: 'bg-amber-50 text-amber-700 border-amber-200', label: 'MEDIUM' },
  };
  const cfg = map[s] || { cls: 'bg-gray-50 text-gray-700 border-gray-200', label: s };
  return <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${cfg.cls}`}>{cfg.label}</span>;
};

const parseDetails = (d) => {
  if (!d) return {};
  if (typeof d === 'string') {
    try { return JSON.parse(d); } catch { return {}; }
  }
  return d;
};

const Logs = () => {
  const [logs, setLogs] = useState([]);
  const [total, setTotal] = useState(0);
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState(null);
  const limit = 20;

  const [filterType, setFilterType] = useState('');
  const [filterCamera, setFilterCamera] = useState('');
  const [filterStatus, setFilterStatus] = useState('');
  const [search, setSearch] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');

  // Debounced copies for text inputs — avoid fetching on every keystroke
  const [debouncedCamera, setDebouncedCamera] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');

  useEffect(() => {
    const t = setTimeout(() => { setDebouncedCamera(filterCamera); setPage(0); }, 300);
    return () => clearTimeout(t);
  }, [filterCamera]);

  useEffect(() => {
    const t = setTimeout(() => { setDebouncedSearch(search); setPage(0); }, 300);
    return () => clearTimeout(t);
  }, [search]);

  const buildParams = useCallback((extra = {}) => {
    const params = new URLSearchParams({ limit, offset: page * limit, ...extra });
    if (filterType) params.append('type', filterType);
    if (debouncedCamera) params.append('camera_id', debouncedCamera);
    if (filterStatus) params.append('status', filterStatus);
    if (debouncedSearch) params.append('search', debouncedSearch);
    if (dateFrom) params.append('date_from', dateFrom);
    if (dateTo) params.append('date_to', dateTo);
    return params;
  }, [page, filterType, debouncedCamera, filterStatus, debouncedSearch, dateFrom, dateTo]);

  const fetchLogs = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/logs?${buildParams()}`, { headers: getAuthHeaders() });
      const data = await res.json();
      setLogs(data.logs || []);
      setTotal(data.total || 0);
    } catch (err) {
      console.error('Error fetching logs:', err);
    } finally {
      setLoading(false);
    }
  }, [buildParams]);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/logs/stats`, { headers: getAuthHeaders() });
      const data = await res.json();
      setStats(data);
    } catch (err) {
      console.error('Error fetching log stats:', err);
    }
  }, []);

  useEffect(() => {
    fetchLogs();
    fetchStats();
  }, [fetchLogs, fetchStats]);

  useEffect(() => {
    const interval = setInterval(() => {
      fetchLogs();
      fetchStats();
    }, 10000);
    return () => clearInterval(interval);
  }, [fetchLogs, fetchStats]);

  const handleExport = async () => {
    try {
      const params = new URLSearchParams();
      if (filterType) params.append('type', filterType);
      if (debouncedCamera) params.append('camera_id', debouncedCamera);
      if (filterStatus) params.append('status', filterStatus);
      if (debouncedSearch) params.append('search', debouncedSearch);
      if (dateFrom) params.append('date_from', dateFrom);
      if (dateTo) params.append('date_to', dateTo);
      const res = await fetch(`${API_BASE}/api/logs/export?${params}`, { headers: getAuthHeaders() });
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `detection_logs_${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error exporting logs:', err);
    }
  };

  const resetPageAnd = (setter) => (v) => { setter(v); setPage(0); };
  const totalPages = Math.ceil(total / limit);

  const renderSubject = (log) => {
    const d = parseDetails(log.details);
    if (log.type === 'face') {
      return (
        <div>
          <span className={`font-medium ${log.status === 'unknown' ? 'text-red-600' : 'text-slate-800'}`}>
            {log.subject || 'Unknown'}
          </span>
          {d.employee_id && <span className="text-gray-500 ml-1">({d.employee_id})</span>}
          {log.confidence > 0 && (
            <span className="ml-2 px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded text-xs">
              {(log.confidence * 100).toFixed(1)}%
            </span>
          )}
        </div>
      );
    }
    if (log.type === 'plate') {
      return (
        <div>
          <span className="font-medium text-slate-800">{log.subject || 'Unknown'}</span>
          {d.owner && <span className="text-gray-500 ml-1">- {d.owner}</span>}
          {d.vehicle_type && <span className="text-gray-500 ml-1">({d.vehicle_type})</span>}
        </div>
      );
    }
    return (
      <div>
        <span className="font-medium text-slate-800">
          {(log.subject || '').toString().toUpperCase() || '-'}
        </span>
        {log.confidence > 0 && (
          <span className="ml-2 px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded text-xs">
            {(log.confidence * 100).toFixed(1)}%
          </span>
        )}
      </div>
    );
  };

  const renderDemographics = (log) => {
    if (log.type !== 'face') return <span className="text-gray-400">-</span>;
    const d = parseDetails(log.details);
    const parts = [];
    if (d.age) parts.push(`Age: ${d.age}`);
    if (d.gender) parts.push(d.gender);
    if (d.emotion) parts.push(d.emotion);
    return parts.length ? parts.join(', ') : <span className="text-gray-400">-</span>;
  };

  return (
    <div>
      {/* Header */}
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <FileText className="w-6 h-6 text-[#3374D0]" />
            <h2 className="text-xl font-semibold text-slate-800">Activity Logs</h2>
          </div>
          <p className="text-sm text-gray-500">Historical record of every detection across all systems</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => { fetchLogs(); fetchStats(); }}
            className="px-3 py-2 rounded-lg text-sm font-medium bg-white text-gray-600 border border-gray-200 hover:bg-gray-50 transition-colors flex items-center gap-2"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={handleExport}
            className="px-3 py-2 rounded-lg text-sm font-medium bg-[#3374D0] text-white hover:bg-[#2860B0] transition-colors flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
        <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
          <div className="text-2xl font-semibold text-slate-800">{stats.total_recent || 0}</div>
          <div className="text-xs text-gray-500 mt-1">Last {stats.window_hours || 24}h</div>
        </div>
        {Object.entries(typeConfig).map(([key, cfg]) => (
          <div key={key} className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
            <div className={`text-2xl font-semibold ${cfg.text}`}>
              {(stats.by_type && stats.by_type[key]) || 0}
            </div>
            <div className="text-xs text-gray-500 mt-1">{cfg.label}</div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-4">
        <select
          className="px-3 py-2 rounded-lg border border-gray-200 text-sm bg-white"
          value={filterType}
          onChange={e => resetPageAnd(setFilterType)(e.target.value)}
        >
          <option value="">All Types</option>
          <option value="face">Face</option>
          <option value="plate">Plate</option>
          <option value="fire">Fire</option>
          <option value="har">Action</option>
          <option value="weapon">Weapon</option>
        </select>
        <select
          className="px-3 py-2 rounded-lg border border-gray-200 text-sm bg-white"
          value={filterStatus}
          onChange={e => resetPageAnd(setFilterStatus)(e.target.value)}
        >
          <option value="">All Statuses</option>
          <option value="unknown">Unknown</option>
          <option value="recognized">Recognized</option>
          <option value="authorized">Authorized</option>
          <option value="unauthorized">Unauthorized</option>
          <option value="alert">Alert</option>
          <option value="threat">Threat</option>
          <option value="critical">Critical</option>
        </select>
        <input
          type="text"
          placeholder="Camera ID"
          className="px-3 py-2 rounded-lg border border-gray-200 text-sm bg-white w-32"
          value={filterCamera}
          onChange={e => setFilterCamera(e.target.value)}
        />
        <div className="relative">
          <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search subject..."
            className="pl-9 pr-3 py-2 rounded-lg border border-gray-200 text-sm bg-white"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
        <input
          type="datetime-local"
          className="px-3 py-2 rounded-lg border border-gray-200 text-sm bg-white"
          value={dateFrom}
          onChange={e => resetPageAnd(setDateFrom)(e.target.value)}
          title="From"
        />
        <input
          type="datetime-local"
          className="px-3 py-2 rounded-lg border border-gray-200 text-sm bg-white"
          value={dateTo}
          onChange={e => resetPageAnd(setDateTo)(e.target.value)}
          title="To"
        />
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="grid grid-cols-12 gap-4 px-6 py-3 border-b border-gray-200 bg-gray-50 text-xs font-medium text-gray-500 uppercase tracking-wider">
          <div className="col-span-2">Time</div>
          <div className="col-span-1">Type</div>
          <div className="col-span-1">Camera</div>
          <div className="col-span-3">Details</div>
          <div className="col-span-2">Demographics</div>
          <div className="col-span-1">Severity</div>
          <div className="col-span-2">Status</div>
        </div>

        <div className="divide-y divide-gray-100 max-h-[500px] overflow-y-auto">
          {loading ? (
            <div className="px-6 py-12 text-center text-gray-500 text-sm">Loading logs...</div>
          ) : logs.length === 0 ? (
            <div className="px-6 py-12 text-center text-gray-500 text-sm">No logs found</div>
          ) : (
            logs.map(log => {
              const tcfg = typeConfig[log.type] || { label: log.type, bg: 'bg-gray-50', text: 'text-gray-700', border: 'border-gray-200' };
              const scfg = severityConfig[log.severity];
              return (
                <div
                  key={log.id}
                  className="grid grid-cols-12 gap-4 px-6 py-3 items-center text-sm hover:bg-gray-50 transition-colors cursor-pointer"
                  onClick={() => setSelected(log)}
                >
                  <div className="col-span-2 text-gray-600 text-xs">
                    {new Date(log.created_at).toLocaleString()}
                  </div>
                  <div className="col-span-1">
                    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium border ${tcfg.bg} ${tcfg.text} ${tcfg.border}`}>
                      {tcfg.label}
                    </span>
                  </div>
                  <div className="col-span-1 text-gray-500 text-xs">{log.camera_id}</div>
                  <div className="col-span-3 text-slate-700">{renderSubject(log)}</div>
                  <div className="col-span-2 text-gray-500 text-xs">{renderDemographics(log)}</div>
                  <div className="col-span-1">
                    {scfg ? (
                      <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium border ${scfg.bg} ${scfg.text} ${scfg.border}`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${scfg.dot}`}></span>
                        {log.severity}
                      </span>
                    ) : <span className="text-gray-400 text-xs">-</span>}
                  </div>
                  <div className="col-span-2">{statusBadge(log)}</div>
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

      {/* Detail Modal */}
      {selected && (
        <>
          <div className="fixed inset-0 bg-black/30 backdrop-blur-sm z-50" onClick={() => setSelected(null)} />
          <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
              <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5 text-[#3374D0]" />
                  <h3 className="text-lg font-semibold text-slate-800">Detection Details</h3>
                  <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${
                    (typeConfig[selected.type] || {}).bg
                  } ${(typeConfig[selected.type] || {}).text} ${(typeConfig[selected.type] || {}).border}`}>
                    {(typeConfig[selected.type] || {}).label || selected.type}
                  </span>
                </div>
                <button onClick={() => setSelected(null)} className="p-2 hover:bg-gray-100 rounded-lg">
                  <X className="w-5 h-5 text-gray-500" />
                </button>
              </div>
              <div className="p-6 space-y-5">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Time</label>
                    <div className="mt-1 text-sm text-slate-800">{new Date(selected.created_at).toLocaleString()}</div>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Camera</label>
                    <div className="mt-1 text-sm text-slate-800">{selected.camera_id}</div>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Subject</label>
                    <div className="mt-1 text-sm text-slate-800">{selected.subject || '-'}</div>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</label>
                    <div className="mt-1 text-sm text-slate-800">
                      {selected.confidence != null ? `${(selected.confidence * 100).toFixed(1)}%` : '-'}
                    </div>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Severity</label>
                    <div className="mt-1 text-sm text-slate-800">{selected.severity || '-'}</div>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Status</label>
                    <div className="mt-1">{statusBadge(selected)}</div>
                  </div>
                </div>
                {selected.details && (
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Detection Data</label>
                    <div className="mt-1 bg-gray-50 rounded-lg p-3 text-xs font-mono text-gray-600">
                      {Object.entries(parseDetails(selected.details)).map(([k, v]) => (
                        <div key={k}><span className="text-gray-800 font-medium">{k}:</span> {JSON.stringify(v)}</div>
                      ))}
                    </div>
                  </div>
                )}
                <div className="flex justify-end">
                  <button
                    onClick={() => setSelected(null)}
                    className="px-4 py-2 rounded-lg text-sm font-medium text-gray-600 hover:bg-gray-100 transition-colors"
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

export default Logs;
