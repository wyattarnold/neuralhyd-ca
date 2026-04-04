import { useMemo, useState, useCallback, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";
import { fetchTimeseries } from "../api/client";
import HexDensityPlot from "./HexDensityPlot";
import {
  CHART_SERIES, GRID_PROPS, XAXIS_PROPS, LEGEND_PROPS,
  JUMP_YEARS, DEFAULT_WINDOW_DAYS, makeDateTickFormatter, makeYAxisProps,
  makeYearTicks,
} from "../chartConfig";

// --- Draggable range slider component ---
function RangeSlider({ min, max, start, end, onChange }) {
  const trackRef = useRef(null);
  const dragging = useRef(null); // "start" | "end" | "middle"
  const dragOrigin = useRef({ x: 0, s: 0, e: 0 });

  const pct = useCallback(
    (v) => ((v - min) / (max - min)) * 100,
    [min, max],
  );

  const valFromX = useCallback(
    (clientX) => {
      const rect = trackRef.current.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      return Math.round(min + frac * (max - min));
    },
    [min, max],
  );

  const onPointerDown = useCallback(
    (e, handle) => {
      e.preventDefault();
      e.stopPropagation();
      dragging.current = handle;
      dragOrigin.current = { x: e.clientX, s: start, e: end };
      const onMove = (ev) => {
        if (!dragging.current) return;
        const h = dragging.current;
        if (h === "start") {
          const v = valFromX(ev.clientX);
          onChange([Math.min(v, end - 30), end]);
        } else if (h === "end") {
          const v = valFromX(ev.clientX);
          onChange([start, Math.max(v, start + 30)]);
        } else {
          // middle drag — slide the window
          const dx = ev.clientX - dragOrigin.current.x;
          const rect = trackRef.current.getBoundingClientRect();
          const dIdx = Math.round((dx / rect.width) * (max - min));
          let newS = dragOrigin.current.s + dIdx;
          let newE = dragOrigin.current.e + dIdx;
          const span = newE - newS;
          if (newS < min) { newS = min; newE = min + span; }
          if (newE > max) { newE = max; newS = max - span; }
          onChange([newS, newE]);
        }
      };
      const onUp = () => {
        dragging.current = null;
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    },
    [start, end, min, max, onChange, valFromX],
  );

  const leftPct = pct(start);
  const rightPct = pct(end);

  return (
    <div
      ref={trackRef}
      className="relative h-4 select-none touch-none"
      style={{ cursor: "default" }}
    >
      {/* Track background */}
      <div className="absolute top-1.5 left-0 right-0 h-1 rounded bg-gray-200" />
      {/* Active range */}
      <div
        className="absolute top-1.5 h-1 rounded bg-blue-400 cursor-grab active:cursor-grabbing"
        style={{ left: `${leftPct}%`, right: `${100 - rightPct}%` }}
        onPointerDown={(e) => onPointerDown(e, "middle")}
      />
      {/* Start thumb */}
      <div
        className="absolute top-0 w-3 h-3 -ml-1.5 rounded-full bg-blue-600 border-2 border-white shadow cursor-ew-resize"
        style={{ left: `${leftPct}%` }}
        onPointerDown={(e) => onPointerDown(e, "start")}
      />
      {/* End thumb */}
      <div
        className="absolute top-0 w-3 h-3 -ml-1.5 rounded-full bg-blue-600 border-2 border-white shadow cursor-ew-resize"
        style={{ left: `${rightPct}%` }}
        onPointerDown={(e) => onPointerDown(e, "end")}
      />
    </div>
  );
}

// --- Main panel ---
export default function TimeseriesPanel({ layerKey, polygonId, name, obsStart, obsEnd, onClose }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["ts", layerKey, polygonId],
    queryFn: () => fetchTimeseries(layerKey, polygonId),
    staleTime: 5 * 60 * 1000,
  });

  const dates = data?.dates ?? [];
  const totalLen = dates.length;

  // Time window [startIdx, endIdx] inclusive
  const [range, setRange] = useState(null);

  const effectiveRange = useMemo(() => {
    if (!totalLen) return [0, 0];
    if (range && range[1] < totalLen) return range;
    // Use obs record bounds when available (training_watersheds)
    if (obsStart && obsEnd && dates.length) {
      const si = dates.indexOf(obsStart);
      const ei = dates.indexOf(obsEnd);
      if (si !== -1 && ei !== -1 && si < ei) return [si, ei];
    }
    return [Math.max(0, totalLen - DEFAULT_WINDOW_DAYS), totalLen - 1];
  }, [totalLen, range, polygonId, obsStart, obsEnd, dates]);

  const dateTickFmt = useMemo(
    () => makeDateTickFormatter(effectiveRange[1] - effectiveRange[0]),
    [effectiveRange],
  );

  const spanDays = effectiveRange[1] - effectiveRange[0];

  const [logScale, setLogScale] = useState(false);
  const [hiddenSeries, setHiddenSeries] = useState(new Set());

  const handleLegendClick = useCallback((entry) => {
    setHiddenSeries((prev) => {
      const next = new Set(prev);
      if (next.has(entry.dataKey)) next.delete(entry.dataKey);
      else next.add(entry.dataKey);
      return next;
    });
  }, []);
  const activeSeries = useMemo(() => {
    if (!data) return [];
    return Object.keys(CHART_SERIES).filter(
      (k) => data[k] && data[k].some((v) => v != null),
    );
  }, [data]);

  // Build chart data for visible window
  const chartData = useMemo(() => {
    if (!dates.length) return [];
    const [s, e] = effectiveRange;
    const slice = [];
    const len = e - s + 1;
    const step = len > 3000 ? Math.ceil(len / 3000) : 1;
    for (let i = s; i <= e; i += step) {
      const pt = { date: dates[i] };
      for (const k of activeSeries) {
        pt[k] = data[k]?.[i] ?? null;
      }
      slice.push(pt);
    }
    return slice;
  }, [dates, effectiveRange, activeSeries, data]);

  // Hex density plot toggle
  const [hexTarget, setHexTarget] = useState("lstm_pred");
  const hasObs = data?.obs?.some((v) => v != null);
  const hasLstm = data?.lstm_pred?.some((v) => v != null);
  const hasVic = data?.vic?.some((v) => v != null);
  const showHex = hasObs && (hasLstm || hasVic);

  // Windowed obs/sim arrays for hex plot
  const hexObs = useMemo(() => {
    if (!data?.obs) return [];
    return data.obs.slice(effectiveRange[0], effectiveRange[1] + 1);
  }, [data, effectiveRange]);

  const hexSim = useMemo(() => {
    if (!data?.[hexTarget]) return [];
    return data[hexTarget].slice(effectiveRange[0], effectiveRange[1] + 1);
  }, [data, hexTarget, effectiveRange]);

  const hexLabel = hexTarget === "lstm_pred" ? "LSTM" : "VIC-Sim";

  const jumpTo = useCallback(
    (years) => {
      if (!totalLen) return;
      const end = totalLen - 1;
      setRange([Math.max(0, end - years * 365), end]);
    },
    [totalLen],
  );

  const hasData = activeSeries.length > 0;

  return (
    <div className="flex flex-col h-full">
      {/* Header bar */}
      <div className="flex items-center gap-3 px-4 py-1.5 border-b border-gray-200 bg-gray-50 shrink-0">
        <div className="min-w-0 flex-1">
          <span className="font-semibold text-sm">{name || polygonId}</span>
          <span className="text-xs text-gray-400 ml-2">
            {layerKey.toUpperCase()} · {polygonId}
          </span>
        </div>
        {hasData && (
          <div className="flex gap-1 text-xs">
            {JUMP_YEARS.map((y) => (
              <button
                key={y}
                onClick={() => jumpTo(y)}
                className="px-1.5 py-0.5 rounded bg-gray-200 hover:bg-blue-200 transition-colors"
              >
                {y}y
              </button>
            ))}
            <button
              onClick={() => setRange([0, totalLen - 1])}
              className="px-1.5 py-0.5 rounded bg-gray-200 hover:bg-blue-200 transition-colors"
            >
              All
            </button>
            <button
              onClick={() => setLogScale((s) => !s)}
              className={`px-1.5 py-0.5 rounded transition-colors ${
                logScale ? "bg-amber-100 text-amber-800 font-medium" : "bg-gray-200 hover:bg-gray-300"
              }`}
            >
              Log
            </button>
          </div>
        )}
        <button
          className="p-1 rounded hover:bg-gray-200 text-gray-500 text-sm leading-none"
          onClick={onClose}
          title="Close"
        >
          ✕
        </button>
      </div>

      {/* Main content: chart + hex plot side by side */}
      <div className="flex flex-1 min-h-0">
        {/* Timeseries chart */}
        <div className="flex-1 min-w-0 px-2 pt-1">
          {isLoading && <p className="text-sm text-gray-400 p-4">Loading…</p>}
          {error && <p className="text-sm text-red-500 p-4">{error.message}</p>}
          {!isLoading && !hasData && (
            <p className="text-sm text-gray-400 italic p-4">No timeseries data.</p>
          )}
          {hasData && (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 4, right: 12, bottom: 4, left: 4 }}>
                <CartesianGrid {...GRID_PROPS} />
                <XAxis
                  dataKey="date"
                  tickFormatter={dateTickFmt}
                  {...(spanDays > 365 * 2
                    ? { ticks: makeYearTicks(chartData) }
                    : { interval: Math.max(1, Math.floor(chartData.length / 10)) }
                  )}
                  {...XAXIS_PROPS}
                />
                <YAxis {...makeYAxisProps(logScale)} />
                <Tooltip
                  labelFormatter={(d) => d}
                  formatter={(v, seriesKey) => [
                    v != null ? Math.round(v).toLocaleString() : "—",
                    CHART_SERIES[seriesKey]?.label ?? seriesKey,
                  ]}
                />
                <Legend
                  {...LEGEND_PROPS}
                  onClick={handleLegendClick}
                  formatter={(value, entry) => (
                    <span style={{ opacity: hiddenSeries.has(entry.dataKey) ? 0.35 : 1, cursor: "pointer" }}>
                      {value}
                    </span>
                  )}
                />
                {activeSeries.map((k) => {
                  const cfg = CHART_SERIES[k];
                  return (
                    <Line
                      key={k}
                      type="monotone"
                      dataKey={k}
                      name={cfg.label}
                      stroke={cfg.color}
                      strokeWidth={cfg.width}
                      strokeOpacity={hiddenSeries.has(k) ? 0 : 1}
                      dot={false}
                      isAnimationActive={false}
                    />
                  );
                })}
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Hex density plot */}
        {showHex && (
          <div className="shrink-0 flex flex-col border-l border-gray-200" style={{ width: 240 }}>
            {/* Toggle LSTM / VIC */}
            <div className="flex items-center justify-center gap-1 py-1 text-xs bg-gray-50 border-b border-gray-100">
              {hasLstm && (
                <button
                  onClick={() => setHexTarget("lstm_pred")}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    hexTarget === "lstm_pred" ? "bg-red-100 text-red-700 font-medium" : "bg-gray-200 hover:bg-gray-300"
                  }`}
                >
                  vs LSTM
                </button>
              )}
              {hasVic && (
                <button
                  onClick={() => setHexTarget("vic")}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    hexTarget === "vic" ? "bg-blue-100 text-blue-700 font-medium" : "bg-gray-200 hover:bg-gray-300"
                  }`}
                >
                  vs VIC
                </button>
              )}
            </div>
            <div className="flex-1 min-h-0 p-1">
              <HexDensityPlot obs={hexObs} sim={hexSim} simLabel={hexLabel} />
            </div>
          </div>
        )}
      </div>

      {/* Time range slider */}
      {hasData && totalLen > 365 && (
        <div className="flex items-center gap-2 px-4 py-1 border-t border-gray-100 bg-gray-50 text-xs shrink-0">
          <span className="text-gray-500 w-20 tabular-nums">{dates[effectiveRange[0]]}</span>
          <div className="flex-1">
            <RangeSlider
              min={0}
              max={totalLen - 1}
              start={effectiveRange[0]}
              end={effectiveRange[1]}
              onChange={setRange}
            />
          </div>
          <span className="text-gray-500 w-20 text-right tabular-nums">{dates[effectiveRange[1]]}</span>
        </div>
      )}
    </div>
  );
}
