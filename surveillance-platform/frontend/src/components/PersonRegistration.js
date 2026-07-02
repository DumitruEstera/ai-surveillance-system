import React, { useState } from 'react';
import { UserPlus, Info } from 'lucide-react';

const PersonRegistration = ({ onRegister }) => {
  const [formData, setFormData] = useState({
    name: '',
    employee_id: '',
    department: '',
    authorized_zones: []
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setMessage('');

    try {
      const result = await onRegister(formData);
      setMessage({ type: 'success', text: result.message || 'Person registered successfully!' });
      setFormData({ name: '', employee_id: '', department: '', authorized_zones: [] });
    } catch (error) {
      setMessage({ type: 'error', text: 'Error registering person: ' + error.message });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-6 py-5 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <UserPlus className="w-6 h-6 text-[#3374D0]" />
            <div>
              <h2 className="text-xl font-semibold text-slate-800">Register New Person</h2>
              <p className="text-sm text-gray-500 mt-1">Add a new person to the facial recognition system</p>
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
              <label htmlFor="name" className="block text-sm font-medium text-slate-700 mb-1.5">
                Full Name <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                required
                placeholder="Enter full name"
                className="w-full px-4 py-2.5 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent transition-all text-sm"
              />
            </div>

            <div>
              <label htmlFor="employee_id" className="block text-sm font-medium text-slate-700 mb-1.5">
                Employee ID <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                id="employee_id"
                name="employee_id"
                value={formData.employee_id}
                onChange={handleInputChange}
                required
                placeholder="Enter employee ID"
                className="w-full px-4 py-2.5 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent transition-all text-sm"
              />
            </div>

            <div>
              <label htmlFor="department" className="block text-sm font-medium text-slate-700 mb-1.5">
                Department
              </label>
              <input
                type="text"
                id="department"
                name="department"
                value={formData.department}
                onChange={handleInputChange}
                placeholder="Enter department"
                className="w-full px-4 py-2.5 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3374D0] focus:border-transparent transition-all text-sm"
              />
            </div>

            <button
              type="submit"
              disabled={isSubmitting || !formData.name || !formData.employee_id}
              className="w-full py-2.5 bg-[#3374D0] hover:bg-[#2861B0] disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors text-sm"
            >
              {isSubmitting ? 'Registering...' : 'Register Person'}
            </button>
          </form>

          <div className="mt-6 p-4 bg-blue-50 border border-blue-100 rounded-lg flex gap-3">
            <Info className="w-5 h-5 text-[#3374D0] shrink-0 mt-0.5" />
            <div className="text-sm text-slate-600">
              <p className="font-medium text-slate-700 mb-1">Next Steps</p>
              <p>After registering, capture face images using the registration script:</p>
              <code className="block mt-2 p-2 bg-white rounded border border-gray-200 text-xs font-mono">
                python example_usage.py --mode register --name "John Doe" --employee-id "EMP001"
              </code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PersonRegistration;
