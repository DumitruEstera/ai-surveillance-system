import React, { useState } from 'react';
import { Car } from 'lucide-react';

const PlateRegistration = ({ onRegister }) => {
  const [formData, setFormData] = useState({
    plate_number: '',
    vehicle_type: 'car',
    owner_name: '',
    owner_id: '',
    is_authorized: true,
    notes: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setMessage('');

    try {
      const result = await onRegister(formData);
      setMessage({ type: 'success', text: result.message || 'License plate registered successfully!' });
      setFormData({
        plate_number: '',
        vehicle_type: 'car',
        owner_name: '',
        owner_id: '',
        is_authorized: true,
        notes: ''
      });
    } catch (error) {
      setMessage({ type: 'error', text: 'Error registering plate: ' + error.message });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-6 py-5 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <Car className="w-6 h-6 text-[#3374D0]" />
            <div>
              <h2 className="text-xl font-semibold text-slate-800">Register License Plate</h2>
              <p className="text-sm text-gray-500 mt-1">Add a new vehicle to the license plate recognition system</p>
            </div>
          </div>
        </div>

        <div className="p-6">
          {message && (
            <div className={`mb-6 p-4 rounded-lg text-sm ${
              message.type === 'success'
                ? 'bg-green-50 border border-green-200 text-green-700'
                : 'bg-red-50 border border-red-200 text-red-700'
            }`}>
              {message.text}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label htmlFor="plate_number" className="block text-sm font-medium text-slate-700 mb-1.5">
                License Plate Number <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                id="plate_number"
                name="plate_number"
                value={formData.plate_number}
                onChange={handleInputChange}
                required
                placeholder="e.g., B 123 ABC"
                className="w-full px-4 py-2.5 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent transition-all text-sm uppercase"
              />
            </div>

            <div>
              <label htmlFor="vehicle_type" className="block text-sm font-medium text-slate-700 mb-1.5">
                Vehicle Type
              </label>
              <select
                id="vehicle_type"
                name="vehicle_type"
                value={formData.vehicle_type}
                onChange={handleInputChange}
                className="w-full px-4 py-2.5 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent transition-all text-sm"
              >
                <option value="car">Car</option>
                <option value="motorcycle">Motorcycle</option>
                <option value="truck">Truck</option>
                <option value="bus">Bus</option>
                <option value="van">Van</option>
              </select>
            </div>

            <div>
              <label htmlFor="owner_name" className="block text-sm font-medium text-slate-700 mb-1.5">
                Owner Name
              </label>
              <input
                type="text"
                id="owner_name"
                name="owner_name"
                value={formData.owner_name}
                onChange={handleInputChange}
                placeholder="Enter owner name"
                className="w-full px-4 py-2.5 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent transition-all text-sm"
              />
            </div>

            <div>
              <label htmlFor="owner_id" className="block text-sm font-medium text-slate-700 mb-1.5">
                Owner ID
              </label>
              <input
                type="text"
                id="owner_id"
                name="owner_id"
                value={formData.owner_id}
                onChange={handleInputChange}
                placeholder="Enter owner/employee ID"
                className="w-full px-4 py-2.5 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent transition-all text-sm"
              />
            </div>

            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => setFormData(prev => ({ ...prev, is_authorized: !prev.is_authorized }))}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                  formData.is_authorized ? 'bg-[#3374D0]' : 'bg-gray-200'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    formData.is_authorized ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
              <span className="text-sm text-slate-700 font-medium">Authorized Vehicle</span>
            </div>

            <div>
              <label htmlFor="notes" className="block text-sm font-medium text-slate-700 mb-1.5">
                Notes
              </label>
              <textarea
                id="notes"
                name="notes"
                value={formData.notes}
                onChange={handleInputChange}
                placeholder="Additional notes (optional)"
                rows="3"
                className="w-full px-4 py-2.5 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent transition-all text-sm resize-none"
              />
            </div>

            <button
              type="submit"
              disabled={isSubmitting || !formData.plate_number}
              className="w-full py-2.5 bg-[#3374D0] hover:bg-[#2861B0] disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors text-sm"
            >
              {isSubmitting ? 'Registering...' : 'Register Plate'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default PlateRegistration;
