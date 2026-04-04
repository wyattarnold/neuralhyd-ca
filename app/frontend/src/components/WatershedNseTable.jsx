import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchLayerGeoJSON } from "../api/client";

/**
 * Table of all training (training_watersheds) basins sorted by LSTM NSE.
 * Highlights the currently selected basin.
 */
export default function WatershedNseTable({ selectedId, onSelectBasin }) {
  const { data: geojson } = useQuery({
    queryKey: ["geojson", "training_watersheds"],
    queryFn: () => fetchLayerGeoJSON("training_watersheds"),
    staleTime: 10 * 60 * 1000,
  });

  // Build a lookup from basin id → feature properties for click callback
  const featureLookup = useMemo(() => {
    if (!geojson?.features) return new Map();
    const m = new Map();
    for (const f of geojson.features) {
      const id = String(f.properties["Pour Point ID"]);
      m.set(id, f.properties);
    }
    return m;
  }, [geojson]);

  const rows = useMemo(() => {
    if (!geojson?.features) return [];
    return geojson.features
      .map((f) => {
        const p = f.properties;
        return {
          id: String(p["Pour Point ID"]),
          tier: p.tier != null ? Math.round(p.tier) : null,
          lstm_nse: p.lstm_nse != null ? Number(p.lstm_nse) : null,
          vic_nse: p.vic_nse != null ? Number(p.vic_nse) : null,
        };
      })
      .filter((r) => r.lstm_nse != null)
      .sort((a, b) => (b.lstm_nse ?? -Infinity) - (a.lstm_nse ?? -Infinity));
  }, [geojson]);

  if (!rows.length) {
    return <div className="p-3 text-sm text-gray-400 italic">No data</div>;
  }

  const tierLabel = { 1: "T1", 2: "T2", 3: "T3" };

  return (
    <div className="overflow-auto h-full text-xs">
      <table className="w-full">
        <thead className="sticky top-0 bg-gray-50 border-b border-gray-200">
          <tr>
            <th className="text-left px-2 py-1 font-medium text-gray-600">#</th>
            <th className="text-left px-2 py-1 font-medium text-gray-600">Basin</th>
            <th className="text-center px-2 py-1 font-medium text-gray-600">Tier</th>
            <th className="text-right px-2 py-1 font-medium text-gray-600">LSTM NSE</th>
            <th className="text-right px-2 py-1 font-medium text-gray-600">VIC NSE</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={r.id}
              className={`border-b border-gray-100 cursor-pointer ${
                r.id === selectedId ? "bg-blue-100 font-semibold" : "hover:bg-blue-50"
              }`}
              onClick={() => {
                if (onSelectBasin) {
                  const props = featureLookup.get(r.id) ?? {};
                  onSelectBasin(r.id, r.id, props);
                }
              }}
            >
              <td className="px-2 py-0.5 text-gray-400">{i + 1}</td>
              <td className="px-2 py-0.5 text-gray-700">{r.id}</td>
              <td className="px-2 py-0.5 text-center text-gray-500">{tierLabel[r.tier] ?? "—"}</td>
              <td className="px-2 py-0.5 text-right font-mono text-gray-900">
                {r.lstm_nse != null ? r.lstm_nse.toFixed(3) : "—"}
              </td>
              <td className="px-2 py-0.5 text-right font-mono text-gray-500">
                {r.vic_nse != null ? r.vic_nse.toFixed(3) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
