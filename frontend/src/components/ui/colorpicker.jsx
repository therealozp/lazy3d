import React, { useState } from 'react';
import { SketchPicker } from 'react-color';

function ColorPickerInput({ id, value, onChange, className }) {
  const [showPicker, setShowPicker] = useState(false);

  const handleColorChange = (color) => {
    onChange?.({ target: { value: color.hex } });
  };

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <input
        id={id}
        type="text"
        readOnly
        onClick={() => setShowPicker(!showPicker)}
        style={{
          background: value,
          color: '#000',
          cursor: 'pointer',
        }}
        className={className}
      />
      {showPicker && (
        <div style={{ position: 'absolute', zIndex: 2 }}>
          <SketchPicker color={value} onChange={handleColorChange} />
        </div>
      )}
    </div>
  );
}

export default ColorPickerInput;
