import React, { useState, useEffect, useCallback } from 'react';
import { Users, UserPlus, Trash2, Edit3, X, Check, Search, Upload, Eye, ChevronLeft, ChevronRight, Camera, Clock, MapPin, Lock } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const getAuthHeaders = () => {
  const token = localStorage.getItem('auth_token');
  return token ? { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' };
};

const getAuthHeadersMultipart = () => {
  const token = localStorage.getItem('auth_token');
  return token ? { 'Authorization': `Bearer ${token}` } : {};
};

const PersonManagement = () => {
  const [persons, setPersons] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [departmentFilter, setDepartmentFilter] = useState('');
  const [departments, setDepartments] = useState([]);
  const [showRegisterForm, setShowRegisterForm] = useState(false);
  const [editingPerson, setEditingPerson] = useState(null);
  const [selectedPerson, setSelectedPerson] = useState(null);
  const [uploadingFor, setUploadingFor] = useState(null);
  const [message, setMessage] = useState({ text: '', type: '' });
  const [loading, setLoading] = useState(false);

  const LIMIT = 10;

  // Register form state
  const [regName, setRegName] = useState('');
  const [regEmployeeId, setRegEmployeeId] = useState('');
  const [regDepartment, setRegDepartment] = useState('');
  const [regZones, setRegZones] = useState([]);
  const [regFiles, setRegFiles] = useState([]);

  // Edit form state
  const [editName, setEditName] = useState('');
  const [editDepartment, setEditDepartment] = useState('');
  const [editZones, setEditZones] = useState([]);

  // All zones available for assignment
  const [zones, setZones] = useState([]);

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

  const toggleZoneIn = (list, setter, zoneName) => {
    if (list.includes(zoneName)) {
      setter(list.filter(z => z !== zoneName));
    } else {
      setter([...list, zoneName]);
    }
  };

  // Detail view state
  const [personDetail, setPersonDetail] = useState(null);
  const [accessHistory, setAccessHistory] = useState([]);

  const showMessage = (text, type = 'success') => {
    setMessage({ text, type });
    setTimeout(() => setMessage({ text: '', type: '' }), 4000);
  };

  const fetchPersons = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: LIMIT, offset: page * LIMIT });
      if (searchQuery) params.append('search', searchQuery);
      if (departmentFilter) params.append('department', departmentFilter);

      const response = await fetch(`${API_BASE}/api/persons?${params}`, {
        headers: getAuthHeaders()
      });
      if (response.ok) {
        const data = await response.json();
        setPersons(data.persons || []);
        setTotal(data.total || 0);
      }
    } catch (error) {
      console.error('Error fetching persons:', error);
    } finally {
      setLoading(false);
    }
  }, [page, searchQuery, departmentFilter]);

  const fetchDepartments = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/persons/departments`, {
        headers: getAuthHeaders()
      });
      if (response.ok) {
        const data = await response.json();
        setDepartments(data.departments || []);
      }
    } catch (error) {
      console.error('Error fetching departments:', error);
    }
  }, []);

  useEffect(() => {
    fetchPersons();
  }, [fetchPersons]);

  useEffect(() => {
    fetchDepartments();
  }, [fetchDepartments]);

  // Reset page when search/filter changes
  useEffect(() => {
    setPage(0);
  }, [searchQuery, departmentFilter]);

  const handleRegister = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(`${API_BASE}/api/persons/register`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          name: regName,
          employee_id: regEmployeeId,
          department: regDepartment || null,
          authorized_zones: regZones
        })
      });
      const data = await response.json();
      if (response.ok) {
        // If face images were selected in the form, upload them now using the
        // person_id returned by the register endpoint.
        if (regFiles.length > 0 && data.person_id) {
          await handleFaceUpload(data.person_id, regFiles);
        } else {
          showMessage(`Person '${regName}' registered successfully. You can now upload face images.`);
        }
        setShowRegisterForm(false);
        setRegName('');
        setRegEmployeeId('');
        setRegDepartment('');
        setRegZones([]);
        setRegFiles([]);
        fetchPersons();
        fetchDepartments();
      } else {
        showMessage(data.detail || 'Failed to register person', 'error');
      }
    } catch (error) {
      showMessage('Error registering person', 'error');
    }
  };

  const handleUpdate = async (personId) => {
    try {
      const body = {};
      if (editName) body.name = editName;
      if (editDepartment !== undefined) body.department = editDepartment;
      body.authorized_zones = editZones;

      const response = await fetch(`${API_BASE}/api/persons/${personId}`, {
        method: 'PUT',
        headers: getAuthHeaders(),
        body: JSON.stringify(body)
      });
      if (response.ok) {
        showMessage('Person updated successfully');
        setEditingPerson(null);
        fetchPersons();
        fetchDepartments();
      } else {
        const data = await response.json();
        showMessage(data.detail || 'Failed to update person', 'error');
      }
    } catch (error) {
      showMessage('Error updating person', 'error');
    }
  };

  const handleDelete = async (personId, name) => {
    if (!window.confirm(`Are you sure you want to delete "${name}"? This will remove all their face data and they will no longer be recognized by the system.`)) return;
    try {
      const response = await fetch(`${API_BASE}/api/persons/${personId}`, {
        method: 'DELETE',
        headers: getAuthHeaders()
      });
      if (response.ok) {
        showMessage(`Person '${name}' deleted successfully`);
        fetchPersons();
        fetchDepartments();
        if (selectedPerson === personId) setSelectedPerson(null);
      } else {
        const data = await response.json();
        showMessage(data.detail || 'Failed to delete person', 'error');
      }
    } catch (error) {
      showMessage('Error deleting person', 'error');
    }
  };

  const handleFaceUpload = async (personId, files) => {
    if (!files || files.length === 0) return;

    const formData = new FormData();
    for (const file of files) {
      formData.append('files', file);
    }

    setUploadingFor(personId);
    try {
      const response = await fetch(`${API_BASE}/api/persons/${personId}/faces`, {
        method: 'POST',
        headers: getAuthHeadersMultipart(),
        body: formData
      });
      const data = await response.json();
      if (response.ok) {
        let msg = data.message;
        if (data.errors && data.errors.length > 0) {
          msg += ` (${data.errors.length} error(s): ${data.errors.join('; ')})`;
        }
        showMessage(msg, data.errors && data.errors.length > 0 ? 'warning' : 'success');
        fetchPersons();
      } else {
        showMessage(data.detail || 'Failed to upload face images', 'error');
      }
    } catch (error) {
      showMessage('Error uploading face images', 'error');
    } finally {
      setUploadingFor(null);
    }
  };

  const viewPersonDetail = async (personId) => {
    if (selectedPerson === personId) {
      setSelectedPerson(null);
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/api/persons/${personId}`, {
        headers: getAuthHeaders()
      });
      if (response.ok) {
        const data = await response.json();
        setPersonDetail(data.person);
        setAccessHistory(data.access_history || []);
        setSelectedPerson(personId);
      }
    } catch (error) {
      console.error('Error fetching person detail:', error);
    }
  };

  const startEdit = (person) => {
    setEditingPerson(person.id);
    setEditName(person.name);
    setEditDepartment(person.department || '');
    setEditZones(Array.isArray(person.authorized_zones) ? person.authorized_zones : []);
  };

  const totalPages = Math.ceil(total / LIMIT);

  return (
    <div className="max-w-[1200px] mx-auto">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <Users className="w-6 h-6 text-[#3374D0]" />
            <div>
              <h2 className="text-xl font-semibold text-slate-800">Person Management</h2>
              <p className="text-sm text-gray-500">Manage registered persons in the facial recognition system</p>
            </div>
          </div>
          <button
            onClick={() => setShowRegisterForm(!showRegisterForm)}
            className="flex items-center gap-2 px-4 py-2 bg-[#3374D0] text-white rounded-lg hover:bg-[#2861B0] transition-colors text-sm font-medium"
          >
            <UserPlus className="w-4 h-4" />
            Register Person
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
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Register New Person</h3>
          <form onSubmit={handleRegister} className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Full Name *</label>
              <input
                type="text"
                value={regName}
                onChange={(e) => setRegName(e.target.value)}
                placeholder="Enter full name"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Employee ID *</label>
              <input
                type="text"
                value={regEmployeeId}
                onChange={(e) => setRegEmployeeId(e.target.value)}
                placeholder="Enter employee ID"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Department</label>
              <input
                type="text"
                value={regDepartment}
                onChange={(e) => setRegDepartment(e.target.value)}
                placeholder="Enter department"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm"
              />
            </div>
            <div className="sm:col-span-3">
              <label className="block text-sm font-medium text-gray-700 mb-1">Authorized Zones</label>
              {zones.length === 0 ? (
                <p className="text-xs text-gray-500">No zones defined. Create zones first in Zone Management.</p>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {zones.map(z => {
                    const active = regZones.includes(z.name);
                    return (
                      <button
                        type="button"
                        key={z.id}
                        onClick={() => toggleZoneIn(regZones, setRegZones, z.name)}
                        className={`px-3 py-1.5 text-xs rounded-full border transition-colors inline-flex items-center gap-1 ${
                          active
                            ? 'bg-[#3374D0] text-white border-[#3374D0]'
                            : 'bg-white text-slate-700 border-gray-300 hover:bg-gray-50'
                        }`}
                      >
                        {z.is_restricted && <Lock className="w-3 h-3" />}
                        {z.name}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
            <div className="sm:col-span-3">
              <label className="block text-sm font-medium text-gray-700 mb-1">Face Images</label>
              <label className="flex items-center gap-2 px-4 py-2 border border-dashed border-gray-300 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors w-fit text-sm text-slate-600">
                <Upload className="w-4 h-4 text-[#3374D0]" />
                {regFiles.length > 0 ? `${regFiles.length} image(s) selected` : 'Choose face images to upload'}
                <input
                  type="file"
                  multiple
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => setRegFiles(Array.from(e.target.files))}
                />
              </label>
              <p className="text-xs text-gray-500 mt-1">Optional. Images are uploaded automatically after the person is registered so they can be recognized right away.</p>
            </div>
            <div className="sm:col-span-3 flex gap-3">
              <button
                type="submit"
                className="px-4 py-2 bg-[#3374D0] text-white rounded-lg hover:bg-[#2861B0] transition-colors text-sm font-medium"
              >
                Register Person
              </button>
              <button
                type="button"
                onClick={() => { setShowRegisterForm(false); setRegFiles([]); }}
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
              placeholder="Search by name or employee ID..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm"
            />
          </div>
          <select
            value={departmentFilter}
            onChange={(e) => setDepartmentFilter(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent text-sm min-w-[180px]"
          >
            <option value="">All Departments</option>
            {departments.map(dept => (
              <option key={dept} value={dept}>{dept}</option>
            ))}
          </select>
          <div className="text-sm text-gray-500 flex items-center px-2">
            {total} person{total !== 1 ? 's' : ''} found
          </div>
        </div>
      </div>

      {/* Persons Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Person</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Employee ID</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Department</th>
              <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Faces</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Seen</th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {persons.map((person) => (
              <React.Fragment key={person.id}>
                <tr className={`hover:bg-gray-50 ${selectedPerson === person.id ? 'bg-blue-50' : ''}`}>
                  <td className="px-6 py-4">
                    {editingPerson === person.id ? (
                      <input
                        type="text"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        className="px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0] w-full max-w-[200px]"
                      />
                    ) : (
                      <div className="text-sm font-medium text-slate-800">{person.name}</div>
                    )}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-600 font-mono">{person.employee_id}</td>
                  <td className="px-6 py-4">
                    {editingPerson === person.id ? (
                      <input
                        type="text"
                        value={editDepartment}
                        onChange={(e) => setEditDepartment(e.target.value)}
                        placeholder="Department"
                        className="px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-[#3374D0] w-full max-w-[160px]"
                      />
                    ) : (
                      <span className="text-sm text-gray-600">{person.department || '-'}</span>
                    )}
                  </td>
                  <td className="px-6 py-4 text-center">
                    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                      person.face_count > 0
                        ? 'bg-green-50 text-green-700 border border-green-200'
                        : 'bg-orange-50 text-orange-700 border border-orange-200'
                    }`}>
                      <Camera className="w-3 h-3" />
                      {person.face_count}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    {person.last_seen
                      ? new Date(person.last_seen).toLocaleString()
                      : <span className="text-gray-400">Never</span>
                    }
                  </td>
                  <td className="px-6 py-4 text-right">
                    {editingPerson === person.id ? (
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={() => handleUpdate(person.id)}
                          className="p-1.5 text-green-600 hover:bg-green-50 rounded transition-colors"
                          title="Save (including zones below)"
                        >
                          <Check className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setEditingPerson(null)}
                          className="p-1.5 text-gray-500 hover:bg-gray-100 rounded transition-colors"
                          title="Cancel"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ) : (
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={() => viewPersonDetail(person.id)}
                          className={`p-1.5 rounded transition-colors ${
                            selectedPerson === person.id
                              ? 'text-[#3374D0] bg-blue-50'
                              : 'text-slate-500 hover:bg-slate-100'
                          }`}
                          title="View details"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <label
                          className={`p-1.5 rounded transition-colors cursor-pointer ${
                            uploadingFor === person.id
                              ? 'text-blue-400 bg-blue-50'
                              : 'text-slate-500 hover:bg-slate-100'
                          }`}
                          title="Upload face images"
                        >
                          <Upload className="w-4 h-4" />
                          <input
                            type="file"
                            multiple
                            accept="image/*"
                            className="hidden"
                            disabled={uploadingFor === person.id}
                            onChange={(e) => handleFaceUpload(person.id, e.target.files)}
                          />
                        </label>
                        <button
                          onClick={() => startEdit(person)}
                          className="p-1.5 text-slate-500 hover:bg-slate-100 rounded transition-colors"
                          title="Edit person"
                        >
                          <Edit3 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDelete(person.id, person.name)}
                          className="p-1.5 text-red-500 hover:bg-red-50 rounded transition-colors"
                          title="Delete person"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    )}
                  </td>
                </tr>

                {/* Edit-zones row */}
                {editingPerson === person.id && (
                  <tr>
                    <td colSpan="6" className="px-6 py-3 bg-blue-50/40 border-t border-blue-100">
                      <div>
                        <div className="text-xs font-medium text-slate-700 mb-2">Authorized Zones</div>
                        {zones.length === 0 ? (
                          <p className="text-xs text-gray-500">No zones defined yet.</p>
                        ) : (
                          <div className="flex flex-wrap gap-2">
                            {zones.map(z => {
                              const active = editZones.includes(z.name);
                              return (
                                <button
                                  type="button"
                                  key={z.id}
                                  onClick={() => toggleZoneIn(editZones, setEditZones, z.name)}
                                  className={`px-3 py-1 text-xs rounded-full border transition-colors inline-flex items-center gap-1 ${
                                    active
                                      ? 'bg-[#3374D0] text-white border-[#3374D0]'
                                      : 'bg-white text-slate-700 border-gray-300 hover:bg-gray-50'
                                  }`}
                                >
                                  {z.is_restricted && <Lock className="w-3 h-3" />}
                                  {z.name}
                                </button>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    </td>
                  </tr>
                )}

                {/* Detail Row */}
                {selectedPerson === person.id && personDetail && (
                  <tr>
                    <td colSpan="6" className="px-6 py-4 bg-slate-50 border-t border-blue-100">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Person Info */}
                        <div>
                          <h4 className="text-sm font-semibold text-slate-700 mb-3">Person Details</h4>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-500">Full Name</span>
                              <span className="text-slate-700 font-medium">{personDetail.name}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Employee ID</span>
                              <span className="text-slate-700 font-mono">{personDetail.employee_id}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Department</span>
                              <span className="text-slate-700">{personDetail.department || '-'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Registered</span>
                              <span className="text-slate-700">{new Date(personDetail.created_at).toLocaleDateString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Face Images</span>
                              <span className={`font-medium ${person.face_count > 0 ? 'text-green-600' : 'text-orange-600'}`}>
                                {person.face_count} registered
                              </span>
                            </div>
                          </div>

                          <div className="mt-3">
                            <div className="text-xs font-medium text-slate-600 mb-1.5">Authorized Zones</div>
                            {(personDetail.authorized_zones && personDetail.authorized_zones.length > 0) ? (
                              <div className="flex flex-wrap gap-1.5">
                                {personDetail.authorized_zones.map((zn) => (
                                  <span key={zn} className="px-2 py-0.5 text-xs rounded-full bg-blue-50 text-blue-700 border border-blue-200 inline-flex items-center gap-1">
                                    <MapPin className="w-3 h-3" />{zn}
                                  </span>
                                ))}
                              </div>
                            ) : (
                              <span className="text-xs text-gray-400">None</span>
                            )}
                          </div>

                          {person.face_count === 0 && (
                            <div className="mt-3 p-3 bg-orange-50 border border-orange-200 rounded-lg text-sm text-orange-700">
                              No face images uploaded yet. Upload face photos so this person can be recognized by the system.
                            </div>
                          )}
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
            {!loading && persons.length === 0 && (
              <tr>
                <td colSpan="6" className="px-6 py-12 text-center text-gray-500">
                  <Users className="w-10 h-10 mx-auto mb-3 text-gray-300" />
                  <p className="text-sm">
                    {searchQuery || departmentFilter
                      ? 'No persons match your search criteria'
                      : 'No persons registered yet. Click "Register Person" to add one.'}
                  </p>
                </td>
              </tr>
            )}
            {loading && (
              <tr>
                <td colSpan="6" className="px-6 py-8 text-center text-gray-400 text-sm">
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

export default PersonManagement;
