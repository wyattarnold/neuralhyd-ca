import { useState, useCallback } from "react";
import WatershedMap from "./components/WatershedMap";
import SidePanel from "./components/SidePanel";
import LayerSelector from "./components/LayerSelector";
import TopBar from "./components/TopBar";
import InfoDrawer from "./components/InfoDrawer";

const LAYER_ORDER = ["huc8", "huc10", "training_watersheds"];
const COLOR_MODES = [
  { id: "lstm_nse", label: "LSTM NSE" },
  { id: "vic_nse", label: "VIC NSE" },
  { id: "tier", label: "Tier" },
];

export default function App() {
  const [activeLayer, setActiveLayer] = useState("training_watersheds");
  const [selected, setSelected] = useState(null);
  const [panelWidth, setPanelWidth] = useState(700);
  const [colorMode, setColorMode] = useState("lstm_nse");
  const [docsOpen, setDocsOpen] = useState(false);

  const panelOpen = selected !== null;

  const handleDragStart = useCallback(
    (e) => {
      e.preventDefault();
      const startX = e.clientX;
      const startW = panelWidth;
      const onMove = (ev) =>
        setPanelWidth(Math.min(1200, Math.max(280, startW + (ev.clientX - startX))));
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [panelWidth],
  );

  return (
    <div className="flex flex-col h-full w-full">
      <TopBar onInfoClick={() => setDocsOpen(true)} />
      <InfoDrawer open={docsOpen} onClose={() => setDocsOpen(false)} />

      <div className="flex flex-1 min-h-0">
      {/* Left side panel */}
      {panelOpen && (
        <>
          <div
            className="bg-white border-r border-gray-200 overflow-hidden shrink-0"
            style={{ width: panelWidth }}
          >
            <SidePanel
              layerKey={selected.layerKey}
              polygonId={selected.id}
              name={selected.name}
              props={selected.props}
              onClose={() => setSelected(null)}
              onSelectBasin={(id, name, props) => setSelected({ layerKey: "training_watersheds", id, name, props })}
            />
          </div>
          {/* Resize handle */}
          <div
            className="w-1.5 cursor-col-resize bg-gray-300 hover:bg-blue-400 transition-colors shrink-0"
            onMouseDown={handleDragStart}
          />
        </>
      )}

      {/* Map area */}
      <div className="flex-1 relative min-w-0">
        {/* Controls overlay */}
        <div className="absolute top-3 left-14 z-[1000] flex items-center gap-2">
          <LayerSelector
            layers={LAYER_ORDER}
            active={activeLayer}
            onChange={setActiveLayer}
          />
          {activeLayer === "training_watersheds" && (
            <select
              className="bg-white/95 border border-gray-300 rounded px-2 py-1 text-sm shadow
                         focus:outline-none focus:ring-2 focus:ring-blue-400"
              value={colorMode}
              onChange={(e) => setColorMode(e.target.value)}
            >
              {COLOR_MODES.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          )}
        </div>
        <WatershedMap
          layerKey={activeLayer}
          selectedId={selected?.id}
          onSelect={(layerKey, id, name, props) => setSelected({ layerKey, id, name, props })}
          colorMode={colorMode}
        />
      </div>
      </div>
    </div>
  );
}
