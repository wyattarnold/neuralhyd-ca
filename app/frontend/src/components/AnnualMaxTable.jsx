import { useMemo } from "react";

/**
 * Table of observed annual maxima daily flows, sorted largest first.
 * For HUC-8/10: uses VIC flows. For training_watersheds: uses observed flows.
 * Click a row → calls onSelectYear(waterYear) to jump Qdaily to that water year.
 */
export default function AnnualMaxTable({ data, layerKey, onSelectYear }) {
  const rows = useMemo(() => {
    if (!data?.dates) return [];
    // Prefer obs for training_watersheds, fall back to vic
    const seriesKey = layerKey === "training_watersheds" ? "obs" : "vic";
    const arr = data[seriesKey] ?? data.vic;
    if (!arr) return [];

    // Group by water year (Oct 1 → Sep 30)
    const wyMap = new Map(); // waterYear → { max, date }
    for (let i = 0; i < data.dates.length; i++) {
      const v = arr[i];
      if (v == null) continue;
      const d = new Date(data.dates[i]);
      const m = d.getMonth();
      const wy = m >= 9 ? d.getFullYear() + 1 : d.getFullYear();
      const cur = wyMap.get(wy);
      if (!cur || v > cur.max) {
        wyMap.set(wy, { max: v, date: data.dates[i] });
      }
    }

    return [...wyMap.entries()]
      .map(([wy, { max, date }]) => ({ wy, max, date }))
      .sort((a, b) => b.max - a.max);
  }, [data, layerKey]);

  if (!rows.length) {
    return <div className="p-3 text-sm text-gray-400 italic">No flow data</div>;
  }

  return (
    <div className="overflow-auto h-full text-xs">
      <table className="w-full">
        <thead className="sticky top-0 bg-gray-50 border-b border-gray-200">
          <tr>
            <th className="text-left px-2 py-1 font-medium text-gray-600">WY</th>
            <th className="text-right px-2 py-1 font-medium text-gray-600">Peak (CFS)</th>
            <th className="text-left px-2 py-1 font-medium text-gray-600">Date</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ wy, max, date }) => (
            <tr
              key={wy}
              className="hover:bg-blue-50 cursor-pointer border-b border-gray-100"
              onClick={() => onSelectYear(wy)}
            >
              <td className="px-2 py-0.5 text-gray-700">{wy}</td>
              <td className="px-2 py-0.5 text-right font-mono text-gray-900">{Math.round(max).toLocaleString()}</td>
              <td className="px-2 py-0.5 text-gray-500">{date}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
