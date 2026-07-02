import React, { useEffect, useState, useCallback, useMemo } from 'react';
import {
  BarChart3, Users, Car, Shield, Video, Globe, Bell, AlertTriangle,
  Flame, Eye, Activity, Camera
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const getAuthHeaders = () => {
  const token = localStorage.getItem('auth_token');
  return token ? { 'Authorization': `Bearer ${token}` } : {};
};

const WINDOWS = [
  { label: '24h', hours: 24 },
  { label: '7d', hours: 24 * 7 },
  { label: '30d', hours: 24 * 30 },
];

const TYPE_COLORS = {
  face: '#3b82f6',
  plate: '#f59e0b',
  fire: '#ef4444',
  weapon: '#8b5cf6',
  har: '#10b981',
  demographics: '#ec4899',
  unknown: '#64748b',
};
const colorFor = (t) => TYPE_COLORS[t] || '#64748b';

const SEVERITY_COLORS = {
  critical: '#dc2626',
  high: '#f97316',
  warning: '#f59e0b',
  medium: '#eab308',
  info: '#3b82f6',
  low: '#64748b',
};
const severityColor = (s) => SEVERITY_COLORS[s?.toLowerCase()] || '#64748b';

// ── Simple inline SVG charts ────────────────────────────────────────────

const LineChart = ({ series, bucket }) => {
  const width = 640, height = 200, pad = { l: 36, r: 12, t: 12, b: 28 };
  if (!series || series.length === 0) {
    return <div className="text-sm text-gray-500 p-8 text-center">No data in this window.</div>;
  }
  const max = Math.max(1, ...series.map(p => p.total));
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const stepX = series.length > 1 ? innerW / (series.length - 1) : 0;
  const points = series.map((p, i) => {
    const x = pad.l + i * stepX;
    const y = pad.t + innerH - (p.total / max) * innerH;
    return [x, y, p];
  });
  const path = points.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const area = `${path} L${points[points.length - 1][0].toFixed(1)},${pad.t + innerH} L${points[0][0].toFixed(1)},${pad.t + innerH} Z`;

  const fmt = (iso) => {
    const d = new Date(iso);
    return bucket === 'hour'
      ? d.toLocaleTimeString([], { hour: '2-digit' })
      : d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  };
  const tickIdx = series.length <= 8
    ? series.map((_, i) => i)
    : [0, Math.floor(series.length / 4), Math.floor(series.length / 2), Math.floor(3 * series.length / 4), series.length - 1];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
      <defs>
        <linearGradient id="lc-grad" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#3374D0" stopOpacity="0.25" />
          <stop offset="100%" stopColor="#3374D0" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 0.25, 0.5, 0.75, 1].map((f, i) => {
        const y = pad.t + innerH * (1 - f);
        return (
          <g key={i}>
            <line x1={pad.l} x2={width - pad.r} y1={y} y2={y} stroke="#e5e7eb" strokeDasharray="2 3" />
            <text x={pad.l - 6} y={y + 3} textAnchor="end" fontSize="10" fill="#94a3b8">
              {Math.round(max * f)}
            </text>
          </g>
        );
      })}
      <path d={area} fill="url(#lc-grad)" />
      <path d={path} fill="none" stroke="#3374D0" strokeWidth="2" />
      {points.map(([x, y, p], i) => (
        <circle key={i} cx={x} cy={y} r="2.5" fill="#3374D0">
          <title>{`${fmt(p.ts)} · ${p.total}`}</title>
        </circle>
      ))}
      {tickIdx.map(i => {
        const [x] = points[i];
        return (
          <text key={i} x={x} y={height - 8} textAnchor="middle" fontSize="10" fill="#64748b">
            {fmt(series[i].ts)}
          </text>
        );
      })}
    </svg>
  );
};

const BarList = ({ items, keyField, color }) => {
  if (!items || items.length === 0) {
    return <div className="text-sm text-gray-500 p-4 text-center">No data.</div>;
  }
  const max = Math.max(1, ...items.map(i => i.count));
  return (
    <div className="space-y-2">
      {items.map((item, i) => (
        <div key={i} className="flex items-center gap-3">
          <div className="w-24 text-xs text-slate-700 truncate capitalize">{item[keyField] || 'unknown'}</div>
          <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                width: `${(item.count / max) * 100}%`,
                backgroundColor: typeof color === 'function' ? color(item[keyField]) : color,
              }}
            />
          </div>
          <div className="w-10 text-right text-xs font-semibold text-slate-700 tabular-nums">{item.count}</div>
        </div>
      ))}
    </div>
  );
};

const Donut = ({ items, keyField, colorFn }) => {
  const size = 160, stroke = 24, r = (size - stroke) / 2, c = 2 * Math.PI * r;
  const total = items.reduce((s, i) => s + i.count, 0);
  if (total === 0) {
    return <div className="text-sm text-gray-500 p-4 text-center">No data.</div>;
  }
  let offset = 0;
  return (
    <div className="flex items-center gap-6">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#f1f5f9" strokeWidth={stroke} />
        {items.map((item, i) => {
          const frac = item.count / total;
          const len = frac * c;
          const dash = `${len} ${c - len}`;
          const el = (
            <circle
              key={i}
              cx={size / 2} cy={size / 2} r={r}
              fill="none"
              stroke={colorFn(item[keyField])}
              strokeWidth={stroke}
              strokeDasharray={dash}
              strokeDashoffset={-offset}
              transform={`rotate(-90 ${size / 2} ${size / 2})`}
            >
              <title>{`${item[keyField]}: ${item.count}`}</title>
            </circle>
          );
          offset += len;
          return el;
        })}
        <text x={size / 2} y={size / 2 - 4} textAnchor="middle" fontSize="20" fontWeight="600" fill="#0f172a">{total}</text>
        <text x={size / 2} y={size / 2 + 14} textAnchor="middle" fontSize="10" fill="#64748b">total</text>
      </svg>
      <div className="flex-1 space-y-1.5">
        {items.map((item, i) => (
          <div key={i} className="flex items-center gap-2 text-xs">
            <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorFn(item[keyField]) }} />
            <span className="capitalize text-slate-700 flex-1">{item[keyField]}</span>
            <span className="tabular-nums font-semibold text-slate-700">{item.count}</span>
            <span className="text-gray-400">{((item.count / total) * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// ── Main component ─────────────────────────────────────────────────────

const Statistics = ({ systemStatus }) => {
  const [hours, setHours] = useState(24);
  const [logStats, setLogStats] = useState(null);
  const [alarmStats, setAlarmStats] = useState(null);
  const [timeseries, setTimeseries] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [registry, setRegistry] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const headers = getAuthHeaders();
      const [r1, r2, r3, r4, r5] = await Promise.all([
        fetch(`${API_BASE}/api/logs/stats?hours=${hours}`, { headers }),
        fetch(`${API_BASE}/api/alarms/stats`, { headers }),
        fetch(`${API_BASE}/api/logs/timeseries?hours=${hours}`, { headers }),
        fetch(`${API_BASE}/api/logs/breakdown?hours=${hours}`, { headers }),
        fetch(`${API_BASE}/api/stats`, { headers }),
      ]);
      if (!r1.ok || !r2.ok || !r3.ok || !r4.ok || !r5.ok) {
        throw new Error('One or more requests failed');
      }
      setLogStats(await r1.json());
      setAlarmStats(await r2.json());
      setTimeseries(await r3.json());
      setBreakdown(await r4.json());
      setRegistry(await r5.json());
    } catch (e) {
      setError(e.message || 'Failed to load statistics');
    } finally {
      setLoading(false);
    }
  }, [hours]);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 15000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  const activityKpis = useMemo(() => {
    const by = logStats?.by_type || {};
    return [
      { label: 'Total Detections', value: logStats?.total_recent || 0, Icon: Activity, color: 'text-blue-600', bg: 'bg-blue-50' },
      { label: 'Face', value: by.face || 0, Icon: Eye, color: 'text-blue-600', bg: 'bg-blue-50' },
      { label: 'License Plate', value: by.plate || 0, Icon: Car, color: 'text-amber-600', bg: 'bg-amber-50' },
      { label: 'Fire / Smoke', value: by.fire || 0, Icon: Flame, color: 'text-red-600', bg: 'bg-red-50' },
      { label: 'Weapon', value: by.weapon || 0, Icon: Shield, color: 'text-purple-600', bg: 'bg-purple-50' },
      { label: 'Activity (HAR)', value: by.har || 0, Icon: Activity, color: 'text-emerald-600', bg: 'bg-emerald-50' },
    ];
  }, [logStats]);

  const alarmKpis = useMemo(() => ([
    { label: 'Unresolved', value: alarmStats?.unresolved || 0, Icon: Bell, color: 'text-indigo-600', bg: 'bg-indigo-50' },
    { label: 'Critical Unresolved', value: alarmStats?.critical_unresolved || 0, Icon: AlertTriangle, color: 'text-rose-600', bg: 'bg-rose-50' },
    { label: 'Resolved (all time)', value: alarmStats?.resolved || 0, Icon: Shield, color: 'text-green-600', bg: 'bg-green-50' },
    { label: 'False Alarms', value: alarmStats?.false_alarm || 0, Icon: AlertTriangle, color: 'text-gray-600', bg: 'bg-gray-50' },
  ]), [alarmStats]);

  const registryItems = useMemo(() => ([
    { label: 'Persons Registered', value: registry?.total_persons || 0, Icon: Users, color: 'text-blue-600', bg: 'bg-blue-50' },
    { label: 'Plates Registered', value: registry?.total_plates || 0, Icon: Car, color: 'text-amber-600', bg: 'bg-amber-50' },
    { label: 'Authorized Plates', value: registry?.authorized_plates || 0, Icon: Shield, color: 'text-emerald-600', bg: 'bg-emerald-50' },
    { label: 'Unauthorized Accesses', value: registry?.unauthorized_vehicle_accesses || 0, Icon: AlertTriangle, color: 'text-red-600', bg: 'bg-red-50' },
  ]), [registry]);

  const enabledDetectors = useMemo(() => {
    if (!systemStatus) return [];
    return [
      { label: 'Face', on: systemStatus.face_detection_enabled },
      { label: 'Plate', on: systemStatus.plate_detection_enabled },
      { label: 'Demographics', on: systemStatus.demographics_enabled },
      { label: 'Fire', on: systemStatus.fire_detection_enabled },
      { label: 'HAR', on: systemStatus.har_enabled },
      { label: 'Weapon', on: systemStatus.weapon_detection_enabled },
    ];
  }, [systemStatus]);

  return (
    <div>
      <div className="mb-6 flex items-start justify-between gap-4 flex-wrap">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <BarChart3 className="w-6 h-6 text-[#3374D0]" />
            <h2 className="text-xl font-semibold text-slate-800">System Statistics</h2>
          </div>
          <p className="text-sm text-gray-500">Detection activity, alarms, and system health</p>
        </div>
        <div className="inline-flex rounded-lg border border-gray-200 bg-white p-1 shadow-sm">
          {WINDOWS.map(w => (
            <button
              key={w.hours}
              onClick={() => setHours(w.hours)}
              className={`px-3 py-1.5 text-sm font-medium rounded-md transition ${
                hours === w.hours
                  ? 'bg-[#3374D0] text-white'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              {w.label}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Activity KPIs */}
      <h3 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-2">
        Activity · Last {WINDOWS.find(w => w.hours === hours)?.label}
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
        {activityKpis.map((s, i) => {
          const Icon = s.Icon;
          return (
            <div key={i} className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
              <div className={`w-9 h-9 ${s.bg} rounded-lg flex items-center justify-center mb-2`}>
                <Icon className={`w-4 h-4 ${s.color}`} />
              </div>
              <div className="text-xl font-semibold text-slate-800">{s.value.toLocaleString()}</div>
              <div className="text-xs text-gray-500 mt-0.5">{s.label}</div>
            </div>
          );
        })}
      </div>

      {/* Timeseries + breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        <div className="lg:col-span-2 bg-white rounded-xl border border-gray-200 shadow-sm p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-slate-700">Detections over time</h3>
            <span className="text-xs text-gray-500">
              {timeseries?.bucket === 'hour' ? 'Hourly' : 'Daily'} buckets
            </span>
          </div>
          {loading && !timeseries ? (
            <div className="text-sm text-gray-500 p-8 text-center">Loading…</div>
          ) : (
            <LineChart series={timeseries?.series} bucket={timeseries?.bucket} />
          )}
        </div>
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-3">By type</h3>
          <Donut
            items={breakdown?.by_type || []}
            keyField="type"
            colorFn={colorFor}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-8">
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
            <Camera className="w-4 h-4 text-gray-500" /> Detections per camera
          </h3>
          <BarList items={breakdown?.by_camera || []} keyField="camera_id" color="#3374D0" />
        </div>
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-gray-500" /> Detections by severity
          </h3>
          <BarList items={breakdown?.by_severity || []} keyField="severity" color={severityColor} />
        </div>
      </div>

      {/* Alarms */}
      <h3 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-2">Alarms</h3>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
        {alarmKpis.map((s, i) => {
          const Icon = s.Icon;
          return (
            <div key={i} className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 ${s.bg} rounded-lg flex items-center justify-center`}>
                  <Icon className={`w-5 h-5 ${s.color}`} />
                </div>
                <div>
                  <div className="text-2xl font-semibold text-slate-800">{s.value.toLocaleString()}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{s.label}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Registry */}
      <h3 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-2">Registry (all time)</h3>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
        {registryItems.map((s, i) => {
          const Icon = s.Icon;
          return (
            <div key={i} className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 ${s.bg} rounded-lg flex items-center justify-center`}>
                  <Icon className={`w-5 h-5 ${s.color}`} />
                </div>
                <div>
                  <div className="text-2xl font-semibold text-slate-800">{s.value.toLocaleString()}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{s.label}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* System health */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-slate-800">System Health</h3>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-6">
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Video className="w-4 h-4 text-gray-500" />
              <span className="text-xs text-gray-500 font-medium">CAM-01 (Local)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${systemStatus?.cam01_streaming ? 'bg-green-500' : 'bg-gray-400'}`}></span>
              <span className="text-sm font-semibold text-slate-800">
                {systemStatus?.cam01_streaming ? 'Streaming' : 'Offline'}
              </span>
            </div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Video className="w-4 h-4 text-gray-500" />
              <span className="text-xs text-gray-500 font-medium">CAM-02 (IP)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${systemStatus?.cam02_streaming ? 'bg-green-500' : 'bg-gray-400'}`}></span>
              <span className="text-sm font-semibold text-slate-800">
                {systemStatus?.cam02_streaming ? 'Streaming' : 'Offline'}
              </span>
            </div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Globe className="w-4 h-4 text-gray-500" />
              <span className="text-xs text-gray-500 font-medium">API</span>
            </div>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${systemStatus?.status === 'online' ? 'bg-green-500' : 'bg-red-500'}`}></span>
              <span className="text-sm font-semibold text-slate-800">
                {systemStatus?.status === 'online' ? 'Online' : 'Unreachable'}
              </span>
            </div>
          </div>
        </div>
        <div className="px-6 pb-6">
          <div className="text-xs text-gray-500 font-medium mb-2">Enabled detectors</div>
          <div className="flex flex-wrap gap-2">
            {enabledDetectors.map(d => (
              <span
                key={d.label}
                className={`text-xs px-2.5 py-1 rounded-full border ${
                  d.on
                    ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
                    : 'bg-gray-50 border-gray-200 text-gray-400'
                }`}
              >
                {d.label}: {d.on ? 'on' : 'off'}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Statistics;
