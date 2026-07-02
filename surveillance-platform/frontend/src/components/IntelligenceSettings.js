import React from 'react';

const IntelligenceSettings = ({
  faceDetectionEnabled,
  plateDetectionEnabled,
  demographicsEnabled,
  fireDetectionEnabled,
  harEnabled,
  weaponDetectionEnabled,
  onToggleFaceDetection,
  onTogglePlateDetection,
  onToggleDemographics,
  onToggleFireDetection,
  onToggleHar,
  onToggleWeaponDetection,
  isAdmin = false
}) => {
  const settings = [
    { id: 'face', label: 'Face Detection', enabled: faceDetectionEnabled, onToggle: onToggleFaceDetection },
    { id: 'plate', label: 'License Plate', enabled: plateDetectionEnabled, onToggle: onTogglePlateDetection },
    { id: 'demo', label: 'Demographics', enabled: demographicsEnabled, onToggle: onToggleDemographics },
    { id: 'fire', label: 'Fire Detection', enabled: fireDetectionEnabled, onToggle: onToggleFireDetection },
    { id: 'action', label: 'Action Recognition', enabled: harEnabled, onToggle: onToggleHar },
    { id: 'weapon', label: 'Weapon Detection', enabled: weaponDetectionEnabled, onToggle: onToggleWeaponDetection },
  ];

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-slate-800 mb-4">Intelligence Settings</h3>
      <p className="text-xs text-gray-400 uppercase tracking-wider mb-6 font-medium">AI models</p>

      {!isAdmin && (
        <p className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 mb-4">
          Only admins can change detection settings
        </p>
      )}

      <div className="space-y-5">
        {settings.map((setting) => (
          <div key={setting.id} className="flex items-center justify-between">
            <span className="text-sm text-slate-600 font-medium">{setting.label}</span>
            {isAdmin ? (
              <button
                onClick={() => setting.onToggle && setting.onToggle(!setting.enabled)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                  setting.enabled ? 'bg-[#3374D0]' : 'bg-gray-200'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    setting.enabled ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            ) : (
              <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                setting.enabled
                  ? 'bg-green-50 text-green-700 border border-green-200'
                  : 'bg-gray-100 text-gray-500 border border-gray-200'
              }`}>
                {setting.enabled ? 'Active' : 'Inactive'}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default IntelligenceSettings;
