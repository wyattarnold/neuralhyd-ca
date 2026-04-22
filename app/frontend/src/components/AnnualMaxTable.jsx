import { useMemo } from "react";

/** Compute per-water-year max from a parallel dates+values array. */
function annualMaxMap(dates, arr) {
  if (!arr) return new Map();
  const wyMap = new Map(); // waterYear → { max, date }
  for (let i = 0; i < dates.length; i++) {
    const v = arr[i];
    if (v == null) continue;
    const d = new Date(dates[i]);
    const m = d.getMonth();
    const wy = m >= 9 ? d.getFullYear() + 1 : d.getFullYear();
    const cur = wyMap.get(wy);
    if (!cur || v > cur.max) {
      wyMap.set(wy, { max: v, date: dates[i] });
    }
  }
  return wyMap;
}

/**
 * Table of annual maxima daily flows, sorted largest first.
 * For training_watersheds: obs peak (CFS) + date, single column.
 * For HUC-8/10: LSTM Dual peak (sorted), with LSTM Single and VIC columns.
 * Click a row → calls onSelectYear(waterYear) to jump Qdaily to that water year.
 */
export default function AnnualMaxTable({ data, layerKey, onSelectYear }) {
  const isTraining = layerKey === "training_watersheds";

  // Training watersheds: single-series table (observed)
  const trainingRows = useMemo(() => {
    if (!isTraining || !data?.dates) return [];
    const wyMap = annualMaxMap(data.dates, data.obs);
    return [...wyMap.entries()]
      .map(([wy, { max, date }]) => ({ wy, max, date }))
      .sort((a, b) => b.max - a.max);
  }, [isTraining, data]);

  // HUC layers: multi-series table sorted by LSTM Dual
  const hucRows = useMemo(() => {
    if (isTraining || !data?.dates) return [];
    const dualMap = annualMaxMap(data.dates, data.lstm_pred);
    const singleMap = annualMaxMap(data.dates, data.lstm_single_pred);
    const vicMap = annualMaxMap(data.dates, data.vic);

    const wySet = new Set([...dualMap.keys(), ...singleMap.keys(), ...vicMap.keys()]);
    return [...wySet]
      .map((wy) => ({
        wy,
        dual: dualMap.get(wy)?.max ?? null,
        dualDate: dualMap.get(wy)?.date ?? null,
        single: singleMap.get(wy)?.max ?? null,
        vic: vicMap.get(wy)?.max ?? null,
      }))
      .filter((r) => r.dual != null)
      .sort((a, b) => b.dual - a.dual);
  }, [isTraining, data]);

  if (isTraining) {
    if (!trainingRows.length) {
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
            {trainingRows.map(({ wy, max, date }) => (
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

  // HUC-8/10: LSTM Dual sorted, with Single + VIC columns
  if (!hucRows.length) {
    return <div className="p-3 text-sm text-gray-400 italic">No LSTM data</div>;
  }
  const fmt = (v) => v != null ? Math.round(v).toLocaleString() : "—";
  return (
    <div className="overflow-auto h-full text-xs">
      <table className="w-full">
        <thead className="sticky top-0 bg-gray-50 border-b border-gray-200">
          <tr>
            <th className="text-left px-2 py-1 font-medium text-gray-600">WY</th>
            <th className="text-right px-2 py-1 font-medium text-orange-600">LSTM Dual</th>
            <th className="text-right px-2 py-1 font-medium text-purple-600">Single</th>
            <th className="text-right px-2 py-1 font-medium text-blue-600">VIC</th>
          </tr>
        </thead>
        <tbody>
          {hucRows.map(({ wy, dual, single, vic }) => (
            <tr
              key={wy}
              className="hover:bg-blue-50 cursor-pointer border-b border-gray-100"
              onClick={() => onSelectYear(wy)}
            >
              <td className="px-2 py-0.5 text-gray-700">{wy}</td>
              <td className="px-2 py-0.5 text-right font-mono text-gray-900">{fmt(dual)}</td>
              <td className="px-2 py-0.5 text-right font-mono text-gray-500">{fmt(single)}</td>
              <td className="px-2 py-0.5 text-right font-mono text-gray-500">{fmt(vic)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
