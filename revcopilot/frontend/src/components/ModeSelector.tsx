import React from 'react';

const ModeSelector: React.FC<{ onModeChange: (mode: string) => void }> = ({ onModeChange }) => {
    const modes = [
        { value: 'auto-solve', label: 'Auto-Solve' },
        { value: 'ai-assist', label: 'AI Assist' },
        { value: 'tutor', label: 'Tutor Mode' },
    ];

    return (
        <div className="mode-selector">
            <h2>Select Mode</h2>
            <select onChange={(e) => onModeChange(e.target.value)}>
                {modes.map((mode) => (
                    <option key={mode.value} value={mode.value}>
                        {mode.label}
                    </option>
                ))}
            </select>
        </div>
    );
};

export default ModeSelector;