const LABELS = {
  huc8: "HUC-8",
  huc10: "HUC-10",
  training_watersheds: "Training Watersheds",
};

export default function LayerSelector({ layers, active, onChange }) {
  return (
    <select
      className="bg-white/95 border border-gray-300 rounded px-2 py-1 text-sm shadow
                 focus:outline-none focus:ring-2 focus:ring-blue-400"
      value={active}
      onChange={(e) => onChange(e.target.value)}
    >
      {layers.map((k) => (
        <option key={k} value={k}>
          {LABELS[k] || k}
        </option>
      ))}
    </select>
  );
}
