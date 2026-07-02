import React, { useState, useEffect, useCallback } from 'react';
import { Car, Plus, Trash2, Edit3, X, Check, Search, Eye, ChevronLeft, ChevronRight, Clock, MapPin, Shield, ShieldOff, AlertTriangle } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const getAuthHeaders = () => {
  const token = localStorage.getItem('auth_token');
  return token ? { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' };
};

const VEHICLE_TYPE_OPTIONS = ['car', 'motorcycle', 'truck', 'bus', 'van'];

const PlateManagement = () => {
  const [plates, setPlates] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [vehicleTypeFilter, setVehicleTypeFilter] = useState('');
  const [authorizedFilter, setAuthorizedFilter] = useState('');
  const [vehicleTypes, setVehicleTypes] = useState([]);
  const [showRegisterForm, setShowRegisterForm] = useState(false);
  const [editingPlate, setEditingPlate] = useState(null);
  const [selectedPlate, setSelectedPlate] = useState(null);
  const [message, setMessage] = useState({ text: '', type: '' });
  const [loading, setLoading] = useState(false);

  const LIMIT = 10;

  // Register form state
  const [regPlateNumber, setRegPlateNumber] = useState('');
  const [regVehicleType, setRegVehicleType] = useState('car');
  const [regOwnerName, setRegOwnerName] = useState('');
  const [regOwnerId, setRegOwnerId] = useState('');
  const [regIsAuthorized, setRegIsAuthorized] = useState(true);
  const [regExpiryDate, setRegExpiryDate] = useState('');
  const [regNotes, setRegNotes] = useState('');

  // Edit form state
  const [editOwnerName, setEditOwnerName] = useState('');
  const [editOwnerId, setEditOwnerId] = useState('');
  const [editVehicleType, setEditVehicleType] = useState('');
  const [editIsAuthorized, setEditIsAuthorized] = useState(true);
  const [editExpiryDate, setEditExpiryDate] = useState('');
  const [editNotes, setEditNotes] = useState('');

  // Detail view state
  const [plateDetail, setPlateDetail] = useState(null);
  const [accessHistory, setAccessHistory] = useState([]);

  const showMessage = (text, type = 'success') => {
    setMessage({ text, type });
    setTimeout(() => setMessage({ text: '', type: '' }), 4000);
  };

  const fetchPlates = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: LIMIT, offset: page * LIMIT });
      if (searchQuery) params.append('search', searchQuery);
      if (vehicleTypeFilter) params.append('vehicle_type', vehicleTypeFilter);
      if (authorizedFilter !== '') params.append('is_authorized', authorizedFilter);

      const response = await fetch(`${API_BASE}/api/plates?${params}`, {
        headers: getAuthHeaders()
      });
      if (response.ok) {
        const data = await response.json();
        setPlates(data.plates || []);
        setTotal(data.total || 0);
      }
    } catch (error) {
      console.error('Error fetching plates:', error);
    } finally {
      setLoading(false);
    }
  }, [page, searchQuery, vehicleTypeFilter, authorizedFilter]);

  const fetchVehicleTypes = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/plates/vehicle-types`, {
        headers: getAuthHeaders()
      });
      if (response.ok) {
        const data = await response.json();
        setVehicleTypes(data.vehicle_types || []);
      }
    } catch (error) {
      console.error('Error fetching vehicle types:', error);
    }
  }, []);

  useEffect(() => {
    fetchPlates();
  }, [fetchPlates]);

  useEffect(() => {
    fetchVehicleTypes();
  }, [fetchVehicleTypes]);

  useEffect(() => {
    setPage(0);
  }, [searchQuery, vehicleTypeFilter, authorizedFilter]);

  const handleRegister = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(`${API_BASE}/api/plates/register`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          plate_number: regPlateNumber.toUpperCase(),
          vehicle_type: regVehicleType,
          owner_name: regOwnerName || null,
          owner_id: regOwnerId || null,
          is_authorized: regIsAuthorized,
          notes: regNotes || null
        })
      });
      const data = await response.json();
      if (response.ok) {
        showMessage(`Plate '${regPlateNumber.toUpperCase()}' registered successfully.`);
        setShowRegisterForm(false);
        setRegPlateNumber('');
        setRegVehicleType('car');
        setRegOwnerName('');
        setRegOwnerId('');
        setRegIsAuthorized(true);
        setRegExpiryDate('');
        setRegNotes('');
        fetchPlates();
        fetchVehicleTypes();
        // If expiry date was set, update it separately since register endpoint doesn't support it
        if (regExpiryDate) {
          await fetch(`${API_BASE}/api/plates/${regPlateNumber.toUpperCase()}`, {
            method: 'PUT',
            headers: getAuthHeaders(),
            body: JSON.stringify({ expiry_date: regExpiryDate })
          });
        }
      } else {
        showMessage(data.detail || 'Failed to register plate', 'error');
      }
    } catch (error) {
      showMessage('Error registering plate', 'error');
    }
  };

  const handleUpdate = async (plateNumber) => {
    try {
      const body = {};
      if (editOwnerName !== undefined) body.owner_name = editOwnerName;
      if (editOwnerId !== undefined) body.owner_id = editOwnerId;
      if (editVehicleType) body.vehicle_type = editVehicleType;
      body.is_authorized = editIsAuthorized;
      if (editExpiryDate !== undefined) body.expiry_date = editExpiryDate || '';
      if (editNotes !== undefined) body.notes = editNotes;

      const response = await fetch(`${API_BASE}/api/plates/${plateNumber}`, {
        method: 'PUT',
        headers: getAuthHeaders(),
        body: JSON.stringify(body)
      });
      if (response.ok) {
        showMessage('Plate updated successfully');
        setEditingPlate(null);
        fetchPlates();
        fetchVehicleTypes();
      } else {
        const data = await response.json();
        showMessage(data.detail || 'Failed to update plate', 'error');
      }
    } catch (error) {
      showMessage('Error updating plate', 'error');
    }
  };

  const handleDelete = async (plateNumber) => {
    if (!window.confirm(`Are you sure you want to delete plate "${plateNumber}"? This will also remove all access logs for this plate.`)) return;
    try {
      const response = await fetch(`${API_BASE}/api/plates/${plateNumber}`, {
        method: 'DELETE',
        headers: getAuthHeaders()
      });
      if (response.ok) {
        showMessage(`Plate '${plateNumber}' deleted successfully`);
        fetchPlates();
        fetchVehicleTypes();
        if (selectedPlate === plateNumber) setSelectedPlate(null);
      } else {
        const data = await response.json();
        showMessage(data.detail || 'Failed to delete plate', 'error');
      }
    } catch (error) {
      showMessage('Error deleting plate', 'error');
    }
  };

  const viewPlateDetail = async (plateNumber) => {
    if (selectedPlate === plateNumber) {
      setSelectedPlate(null);
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/api/plates/${plateNumber}`, {
        headers: getAuthHeaders()
      });
      if (response.ok) {
        const data = await response.json();
        setPlateDetail(data.plate);
        setAccessHistory(data.access_history || []);
        setSelectedPlate(plateNumber);
      }
    } catch (error) {
      console.error('Error fetching plate detail:', error);
    }
  };

  const startEdit = (plate) => {
    setEditingPlate(plate.plate_number);
    setEditOwnerName(plate.owner_name || '');
    setEditOwnerId(plate.owner_id || '');
    setEditVehicleType(plate.vehicle_type || 'car');
    setEditIsAuthorized(plate.is_authorized);
    setEditExpiryDate(plate.expiry_date ? plate.expiry_date.split('T')[0] : '');
    setEditNotes(plate.notes || '');
  };

  const isExpired = (expiryDate) => {
    if (!expiryDate) return false;
    return new Date(expiryDate) < new Date();
  };

  const totalPages = Math.ceil(total / LIMIT);

  return (
    <div className="max-w-[1200px] mx-auto">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <Car className="w-6 h-6 text-[#3374D0]" />
            <div>
              <h2 className="text-xl font-semibold text-slate-800">Plate Management</h2>
              <p className="text-sm text-gray-500">Manage registered vehicles in the license plate recognition system</p>
            </div>
          </div>
          <button
            onClick={() => setShowRegisterForm(!showRegisterForm)}
            className="flex items-center gap-2 px-4 py-2 bg-[#3374D0] text-white rounded-lg hover:bg-[#2861B0] transition-colors text-sm font-medium"
          >
            <Plus className="w-4 h-4" />
            Register Plate
          </button>
        </div>
      </div>

      {/* Messages */}
      {message.text && (
        <div className={`mb-4 p-3 rounded-lg text-sm ${
          message.type === 'error'
            ? 'bg-red-50 border border-red-200 text-red-700'
            : message.type === 'warning'
            ? 'bg-yellow-50 border border-yellow-200 text-yellow-700'
            : 'bg-green-50 border border-green-200 text-green-700'
        }`}>
          {message.text}
        </div>
      )}

      {/* Register Form */}
      {showRegisterForm && (
        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm mb-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Register New Plate</h3>
          <form onSubmit={handleRegister} className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Plate Number *</label>
              <input
                type="text"
                value={regPlateNumber}
                onChange={(e) => setRegPlateNumber(e.target.value)}
                placeholder="e.g., B 123 ABC"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm uppercase"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Vehicle Type</label>
              <select
                value={regVehicleType}
                onChange={(e) => setRegVehicleType(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm"
              >
                {VEHICLE_TYPE_OPTIONS.map(type => (
                  <option key={type} value={type}>{type.charAt(0).toUpperCase() + type.slice(1)}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Owner Name</label>
              <input
                type="text"
                value={regOwnerName}
                onChange={(e) => setRegOwnerName(e.target.value)}
                placeholder="Enter owner name"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Owner ID</label>
              <input
                type="text"
                value={regOwnerId}
                onChange={(e) => setRegOwnerId(e.target.value)}
                placeholder="Enter owner/employee ID"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Expiry Date</label>
              <input
                type="date"
                value={regExpiryDate}
                onChange={(e) => setRegExpiryDate(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm"
              />
            </div>
            <div className="flex items-end">
              <div className="flex items-center gap-3 pb-2">
                <button
                  type="button"
                  onClick={() => setRegIsAuthorized(!regIsAuthorized)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                    regIsAuthorized ? 'bg-[#3374D0]' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      regIsAuthorized ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
                <span className="text-sm text-slate-700 font-medium">Authorized</span>
              </div>
            </div>
            <div className="sm:col-span-3">
              <label className="block text-sm font-medium text-gray-700 mb-1">Notes</label>
              <textarea
                value={regNotes}
                onChange={(e) => setRegNotes(e.target.value)}
                placeholder="Additional notes (optional)"
                rows="2"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm resize-none"
              />
            </div>
            <div className="sm:col-span-3 flex gap-3">
              <button
                type="submit"
                className="px-4 py-2 bg-[#3374D0] text-white rounded-lg hover:bg-[#2861B0] transition-colors text-sm font-medium"
              >
                Register Plate
              </button>
              <button
                type="button"
                onClick={() => setShowRegisterForm(false)}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Search & Filters */}
      <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm mb-4">
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by plate number or owner name..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm"
            />
          </div>
          <select
            value={vehicleTypeFilter}
            onChange={(e) => setVehicleTypeFilter(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm min-w-[150px]"
          >
            <option value="">All Types</option>
            {vehicleTypes.map(type => (
              <option key={type} value={type}>{type.charAt(0).toUpperCase() + type.slice(1)}</option>
            ))}
          </select>
          <select
            value={authorizedFilter}
            onChange={(e) => setAuthorizedFilter(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm min-w-[150px]"
          >
            <option value="">All Status</option>
            <option value="true">Authorized</option>
            <option value="false">Unauthorized</option>
          </select>
          <div className="text-sm text-gray-500 flex items-center px-2">
            {total} plate{total !== 1 ? 's' : ''} found
          </div>
        </div>
      </div>

      {/* Plates Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Plate Number</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Owner</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Vehicle Type</th>
              <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Detections</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Seen</th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {plates.map((plate) => (
              <React.Fragment key={plate.plate_number}>
                <tr className={`hover:bg-gray-50 ${selectedPlate === plate.plate_number ? 'bg-blue-50' : ''}`}>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-mono font-semibold text-slate-800">{plate.plate_number}</span>
                      {isExpired(plate.expiry_date) && (
                        <AlertTriangle className="w-3.5 h-3.5 text-amber-500" title="Expired" />
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    {editingPlate === plate.plate_number ? (
                      <div className="space-y-1">
                        <input
                          type="text"
                          value={editOwnerName}
                          onChange={(e) => setEditOwnerName(e.target.value)}
                          placeholder="Owner name"
                          className="px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0] w-full max-w-[160px]"
                        />
                        <input
                          type="text"
                          value={editOwnerId}
                          onChange={(e) => setEditOwnerId(e.target.value)}
                          placeholder="Owner ID"
                          className="px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0] w-full max-w-[160px]"
                        />
                      </div>
                    ) : (
                      <div>
                        <div className="text-sm font-medium text-slate-800">{plate.owner_name || '-'}</div>
                        {plate.owner_id && <div className="text-xs text-gray-400 font-mono">{plate.owner_id}</div>}
                      </div>
                    )}
                  </td>
                  <td className="px-6 py-4">
                    {editingPlate === plate.plate_number ? (
                      <select
                        value={editVehicleType}
                        onChange={(e) => setEditVehicleType(e.target.value)}
                        className="px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0]"
                      >
                        {VEHICLE_TYPE_OPTIONS.map(type => (
                          <option key={type} value={type}>{type.charAt(0).toUpperCase() + type.slice(1)}</option>
                        ))}
                      </select>
                    ) : (
                      <span className="text-sm text-gray-600 capitalize">{plate.vehicle_type || '-'}</span>
                    )}
                  </td>
                  <td className="px-6 py-4 text-center">
                    {editingPlate === plate.plate_number ? (
                      <button
                        type="button"
                        onClick={() => setEditIsAuthorized(!editIsAuthorized)}
                        className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none ${
                          editIsAuthorized ? 'bg-[#3374D0]' : 'bg-gray-200'
                        }`}
                      >
                        <span
                          className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                            editIsAuthorized ? 'translate-x-4.5' : 'translate-x-0.5'
                          }`}
                        />
                      </button>
                    ) : (
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                        plate.is_authorized
                          ? 'bg-green-50 text-green-700 border border-green-200'
                          : 'bg-red-50 text-red-700 border border-red-200'
                      }`}>
                        {plate.is_authorized ? <Shield className="w-3 h-3" /> : <ShieldOff className="w-3 h-3" />}
                        {plate.is_authorized ? 'Authorized' : 'Unauthorized'}
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4 text-center">
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-slate-50 text-slate-600 border border-slate-200">
                      {plate.detection_count || 0}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    {plate.last_seen
                      ? new Date(plate.last_seen).toLocaleString()
                      : <span className="text-gray-400">Never</span>
                    }
                  </td>
                  <td className="px-6 py-4 text-right">
                    {editingPlate === plate.plate_number ? (
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={() => handleUpdate(plate.plate_number)}
                          className="p-1.5 text-green-600 hover:bg-green-50 rounded transition-colors"
                          title="Save"
                        >
                          <Check className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setEditingPlate(null)}
                          className="p-1.5 text-gray-500 hover:bg-gray-100 rounded transition-colors"
                          title="Cancel"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ) : (
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={() => viewPlateDetail(plate.plate_number)}
                          className={`p-1.5 rounded transition-colors ${
                            selectedPlate === plate.plate_number
                              ? 'text-[#3374D0] bg-blue-50'
                              : 'text-slate-500 hover:bg-slate-100'
                          }`}
                          title="View details"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => startEdit(plate)}
                          className="p-1.5 text-slate-500 hover:bg-slate-100 rounded transition-colors"
                          title="Edit plate"
                        >
                          <Edit3 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDelete(plate.plate_number)}
                          className="p-1.5 text-red-500 hover:bg-red-50 rounded transition-colors"
                          title="Delete plate"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    )}
                  </td>
                </tr>

                {/* Detail Row */}
                {selectedPlate === plate.plate_number && plateDetail && (
                  <tr>
                    <td colSpan="7" className="px-6 py-4 bg-slate-50 border-t border-blue-100">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Plate Info */}
                        <div>
                          <h4 className="text-sm font-semibold text-slate-700 mb-3">Plate Details</h4>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-500">Plate Number</span>
                              <span className="text-slate-700 font-mono font-semibold">{plateDetail.plate_number}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Owner</span>
                              <span className="text-slate-700 font-medium">{plateDetail.owner_name || '-'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Owner ID</span>
                              <span className="text-slate-700 font-mono">{plateDetail.owner_id || '-'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Vehicle Type</span>
                              <span className="text-slate-700 capitalize">{plateDetail.vehicle_type || '-'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Status</span>
                              <span className={`font-medium ${plateDetail.is_authorized ? 'text-green-600' : 'text-red-600'}`}>
                                {plateDetail.is_authorized ? 'Authorized' : 'Unauthorized'}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Registered</span>
                              <span className="text-slate-700">{new Date(plateDetail.registration_date).toLocaleDateString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Expiry Date</span>
                              <span className={`${isExpired(plateDetail.expiry_date) ? 'text-red-600 font-medium' : 'text-slate-700'}`}>
                                {plateDetail.expiry_date
                                  ? new Date(plateDetail.expiry_date).toLocaleDateString() + (isExpired(plateDetail.expiry_date) ? ' (Expired)' : '')
                                  : 'No expiry'}
                              </span>
                            </div>
                            {plateDetail.notes && (
                              <div className="pt-2 border-t border-gray-200">
                                <span className="text-gray-500 block mb-1">Notes</span>
                                <span className="text-slate-700 text-xs">{plateDetail.notes}</span>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Access History */}
                        <div>
                          <h4 className="text-sm font-semibold text-slate-700 mb-3">Recent Access History</h4>
                          {accessHistory.length > 0 ? (
                            <div className="space-y-2 max-h-[200px] overflow-y-auto">
                              {accessHistory.map((entry, idx) => (
                                <div key={idx} className="flex items-center gap-3 text-sm p-2 bg-white rounded-lg border border-gray-100">
                                  <Clock className="w-3.5 h-3.5 text-gray-400 shrink-0" />
                                  <span className="text-gray-600">
                                    {new Date(entry.detected_at).toLocaleString()}
                                  </span>
                                  <MapPin className="w-3.5 h-3.5 text-gray-400 shrink-0 ml-auto" />
                                  <span className="text-slate-700 font-mono text-xs">{entry.camera_id}</span>
                                  <span className="text-xs text-gray-400">
                                    {(entry.confidence * 100).toFixed(0)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <p className="text-sm text-gray-400">No access records yet</p>
                          )}
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
            {!loading && plates.length === 0 && (
              <tr>
                <td colSpan="7" className="px-6 py-12 text-center text-gray-500">
                  <Car className="w-10 h-10 mx-auto mb-3 text-gray-300" />
                  <p className="text-sm">
                    {searchQuery || vehicleTypeFilter || authorizedFilter
                      ? 'No plates match your search criteria'
                      : 'No plates registered yet. Click "Register Plate" to add one.'}
                  </p>
                </td>
              </tr>
            )}
            {loading && (
              <tr>
                <td colSpan="7" className="px-6 py-8 text-center text-gray-400 text-sm">
                  Loading...
                </td>
              </tr>
            )}
          </tbody>
        </table>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-6 py-3 border-t border-gray-200 bg-gray-50">
            <span className="text-sm text-gray-500">
              Showing {page * LIMIT + 1}-{Math.min((page + 1) * LIMIT, total)} of {total}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                className="p-1.5 rounded border border-gray-300 hover:bg-gray-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-sm text-gray-600 px-2">
                Page {page + 1} of {totalPages}
              </span>
              <button
                onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="p-1.5 rounded border border-gray-300 hover:bg-gray-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PlateManagement;
