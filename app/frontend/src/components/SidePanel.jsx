import { useMemo, useState, useCallback, useRef, useEffect } from "react";
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
import { fetchTimeseries, fetchStaticAttrs } from "../api/client";
import QstatPlot from "./QstatPlot";
import AnnualMaxTable from "./AnnualMaxTable";
import HexDensityPlot from "./HexDensityPlot";
import WatershedNseTable from "./WatershedNseTable";
import {
  CHART_SERIES, GRID_PROPS, XAXIS_PROPS, LEGEND_PROPS,
  JUMP_YEARS, DEFAULT_WINDOW_DAYS, makeDateTickFormatter, makeYAxisProps,
  makeYearTicks, symlog, tooltipValFmt, aggregateMonthly, makeSymlogTicks,
} from "../chartConfig";

// --- Draggable range slider ---
function RangeSlider({ min, max, start, end, onChange }) {
  const trackRef = useRef(null);
  const dragging = useRef(null);
  const dragOrigin = useRef({ x: 0, s: 0, e: 0 });

  const pct = useCallback((v) => ((v - min) / (max - min)) * 100, [min, max]);
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
        const h = dragging.current;
        if (!h) return;
        if (h === "start") {
          onChange([Math.min(valFromX(ev.clientX), end - 30), end]);
        } else if (h === "end") {
          onChange([start, Math.max(valFromX(ev.clientX), start + 30)]);
        } else {
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
    <div ref={trackRef} className="relative h-4 select-none touch-none">
      <div className="absolute top-1.5 left-0 right-0 h-1 rounded bg-gray-200" />
      <div
        className="absolute top-1.5 h-1 rounded bg-blue-400 cursor-grab active:cursor-grabbing"
        style={{ left: `${leftPct}%`, right: `${100 - rightPct}%` }}
        onPointerDown={(e) => onPointerDown(e, "middle")}
      />
      <div
        className="absolute top-0 w-3 h-3 -ml-1.5 rounded-full bg-blue-600 border-2 border-white shadow cursor-ew-resize"
        style={{ left: `${leftPct}%` }}
        onPointerDown={(e) => onPointerDown(e, "start")}
      />
      <div
        className="absolute top-0 w-3 h-3 -ml-1.5 rounded-full bg-blue-600 border-2 border-white shadow cursor-ew-resize"
        style={{ left: `${rightPct}%` }}
        onPointerDown={(e) => onPointerDown(e, "end")}
      />
    </div>
  );
}

// --- Info Table ---
function InfoTable({ layerKey, polygonId, props, staticAttrs }) {
  const tier = props?.tier != null ? Math.round(props.tier) : null;
  const tierLabel = { 1: "Tier 1 (Rainfall)", 2: "Tier 2 (Transitional)", 3: "Tier 3 (Snow)" };

  const rows = [];
  if (tier) rows.push(["Tier", tierLabel[tier] || `Tier ${tier}`]);
  if (props?.obs_start) rows.push(["Obs Period", `${props.obs_start} → ${props.obs_end}`]);
  if (props?.lstm_nse != null) rows.push(["LSTM NSE", Number(props.lstm_nse).toFixed(3)]);
  if (props?.vic_nse != null) rows.push(["VIC NSE", Number(props.vic_nse).toFixed(3)]);

  // Static attributes
  if (staticAttrs) {
    for (const [col, { label, value, unit }] of Object.entries(staticAttrs)) {
      rows.push([label, `${value}${unit ? ` ${unit}` : ""}`]);
    }
  }

  return (
    <div className="overflow-auto text-xs">
      <table className="w-full">
        <tbody>
          {rows.map(([k, v], i) => (
            <tr key={i} className="border-b border-gray-100">
              <td className="px-2 py-0.5 font-medium text-gray-500 whitespace-nowrap">{k}</td>
              <td className="px-2 py-0.5 text-gray-800">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// --- Main SidePanel ---
export default function SidePanel({ layerKey, polygonId, name, props, onClose, onSelectBasin }) {
  const [tab, setTab] = useState("qdaily");
  const [range, setRange] = useState(null);
  const [logScale, setLogScale] = useState(false);
  const [monthly, setMonthly] = useState(false);
  const [hiddenSeries, setHiddenSeries] = useState(new Set());
  const [bottomTab, setBottomTab] = useState("peaks");

  // Reset range when toggling monthly
  useEffect(() => { setRange(null); }, [monthly]);

  const unit = monthly ? "AF" : "CFS";

  const handleLegendClick = useCallback((entry) => {
    setHiddenSeries((prev) => {
      const next = new Set(prev);
      if (next.has(entry.dataKey)) next.delete(entry.dataKey);
      else next.add(entry.dataKey);
      return next;
    });
  }, []);
  const { data, isLoading, error } = useQuery({
    queryKey: ["ts", layerKey, polygonId],
    queryFn: () => fetchTimeseries(layerKey, polygonId),
    staleTime: 5 * 60 * 1000,
  });

  // Fetch static attrs (only for training_watersheds)
  const { data: staticAttrs } = useQuery({
    queryKey: ["static", polygonId],
    queryFn: () => fetchStaticAttrs(polygonId),
    staleTime: Infinity,
    enabled: layerKey === "training_watersheds",
  });

  const dates = data?.dates ?? [];
  const totalLen = dates.length;
  const obsStart = props?.obs_start ?? null;
  const obsEnd = props?.obs_end ?? null;

  // Monthly aggregation
  const monthlyData = useMemo(() => {
    if (!data?.dates?.length) return null;
    return aggregateMonthly(data);
  }, [data]);

  // displayData switches between daily and monthly
  const displayData = monthly && monthlyData ? monthlyData : data;
  const displayDates = displayData?.dates ?? [];
  const displayLen = displayDates.length;
  const stepsPerYear = monthly ? 12 : 365;

  const effectiveRange = useMemo(() => {
    if (!displayLen) return [0, 0];
    if (range && range[1] < displayLen) return range;
    if (!monthly && obsStart && obsEnd && displayDates.length) {
      const si = displayDates.indexOf(obsStart);
      const ei = displayDates.indexOf(obsEnd);
      if (si !== -1 && ei !== -1 && si < ei) return [si, ei];
    }
    return [Math.max(0, displayLen - (monthly ? 12 * 5 : DEFAULT_WINDOW_DAYS)), displayLen - 1];
  }, [displayLen, range, obsStart, obsEnd, displayDates, monthly]);

  const dateTickFmt = useMemo(
    () => makeDateTickFormatter(monthly ? (effectiveRange[1] - effectiveRange[0]) * 30 : effectiveRange[1] - effectiveRange[0]),
    [effectiveRange, monthly],
  );

  const spanDays = monthly
    ? (effectiveRange[1] - effectiveRange[0]) * 30
    : effectiveRange[1] - effectiveRange[0];

  const activeSeries = useMemo(() => {
    if (!displayData) return [];
    return Object.keys(CHART_SERIES).filter(
      (k) => displayData[k] && displayData[k].some((v) => v != null),
    );
  }, [displayData]);

  const chartData = useMemo(() => {
    if (!displayDates.length) return [];
    const [s, e] = effectiveRange;
    const slice = [];
    const len = e - s + 1;
    const step = len > 3000 ? Math.ceil(len / 3000) : 1;
    for (let i = s; i <= e; i += step) {
      const pt = { date: displayDates[i] };
      for (const k of activeSeries) {
        const raw = displayData[k]?.[i] ?? null;
        pt[k] = logScale ? symlog(raw) : raw;
      }
      slice.push(pt);
    }
    return slice;
  }, [displayDates, effectiveRange, activeSeries, displayData, logScale]);

  // Dynamic symlog ticks based on visible data max
  const symlogTicks = useMemo(() => {
    if (!logScale) return undefined;
    let maxVal = 0;
    const [s, e] = effectiveRange;
    for (let i = s; i <= e; i++) {
      for (const k of activeSeries) {
        const v = displayData?.[k]?.[i];
        if (v != null && v > maxVal) maxVal = v;
      }
    }
    return makeSymlogTicks(maxVal);
  }, [logScale, effectiveRange, activeSeries, displayData]);

  // Hex density plot state
  const [hexTarget, setHexTarget] = useState("lstm_pred");
  const hasObs = displayData?.obs?.some((v) => v != null);
  const hasLstm = displayData?.lstm_pred?.some((v) => v != null);
  const hasVic = displayData?.vic?.some((v) => v != null);
  const showHexTab = hasObs && (hasLstm || hasVic);

  const hexObs = useMemo(() => {
    if (!displayData?.obs) return [];
    return displayData.obs;
  }, [displayData]);

  const hexSim = useMemo(() => {
    if (!displayData?.[hexTarget]) return [];
    return displayData[hexTarget];
  }, [displayData, hexTarget]);

  const hexLabel = hexTarget === "lstm_pred" ? "LSTM" : "VIC-Sim";

  const jumpTo = useCallback(
    (years) => {
      if (!displayLen) return;
      const end = displayLen - 1;
      setRange([Math.max(0, end - years * stepsPerYear), end]);
    },
    [displayLen, stepsPerYear],
  );

  // Jump to a water year (called from AnnualMaxTable)
  const jumpToWaterYear = useCallback(
    (wy) => {
      if (!dates.length) return;
      // Water year WY starts Oct 1 of WY-1, ends Sep 30 of WY
      const wyStart = `${wy - 1}-10-01`;
      const wyEnd = `${wy}-09-30`;
      let si = dates.findIndex((d) => d >= wyStart);
      let ei = dates.findIndex((d) => d > wyEnd);
      if (si === -1) si = 0;
      if (ei === -1) ei = dates.length - 1;
      else ei = ei - 1;
      setRange([si, Math.max(si + 30, ei)]);
      setTab("qdaily");
    },
    [dates],
  );

  const hasData = activeSeries.length > 0;
  const TABS = [
    { id: "qdaily", label: "Qdaily" },
    { id: "qstat", label: "Qstat" },
  ];
  if (showHexTab) TABS.push({ id: "r2", label: "R²" });

  return (
    <div className="flex flex-col h-full bg-white text-gray-900">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-200 bg-gray-50 shrink-0">
        <div className="min-w-0 flex-1">
          <span className="font-semibold text-sm">{name || polygonId}</span>
          <span className="text-xs text-gray-400 ml-2">{polygonId}</span>
        </div>
        <button
          className="p-1 rounded hover:bg-gray-200 text-gray-500 text-sm leading-none"
          onClick={onClose}
          title="Close"
        >
          ✕
        </button>
      </div>

      {/* Section A: Info table */}
      <div className="shrink-0 border-b border-gray-200 max-h-40 overflow-auto">
        <InfoTable layerKey={layerKey} polygonId={polygonId} props={props} staticAttrs={staticAttrs} />
      </div>

      {/* Section B: Tabbed plots */}
      <div className="flex-1 min-h-0 flex flex-col">
        {/* Tab bar */}
        <div className="flex items-center gap-1 px-2 py-1 border-b border-gray-200 bg-gray-50 shrink-0">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-2 py-0.5 rounded text-xs transition-colors ${
                tab === t.id
                  ? "bg-blue-100 text-blue-700 font-medium"
                  : "bg-gray-200 hover:bg-gray-300 text-gray-600"
              }`}
            >
              {t.label}
            </button>
          ))}

          {/* Time range quick buttons (qdaily only) */}
          {tab === "qdaily" && hasData && (
            <div className="ml-auto flex gap-1 text-xs">
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
                onClick={() => setRange([0, displayLen - 1])}
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
              <button
                onClick={() => setMonthly((m) => !m)}
                className={`px-1.5 py-0.5 rounded transition-colors ${
                  monthly ? "bg-green-100 text-green-800 font-medium" : "bg-gray-200 hover:bg-gray-300"
                }`}
              >
                Monthly
              </button>
            </div>
          )}
          {/* Monthly toggle for Qstat and R² tabs */}
          {(tab === "qstat" || tab === "r2") && (
            <div className="ml-auto flex gap-1 text-xs">
              <button
                onClick={() => setMonthly((m) => !m)}
                className={`px-1.5 py-0.5 rounded transition-colors ${
                  monthly ? "bg-green-100 text-green-800 font-medium" : "bg-gray-200 hover:bg-gray-300"
                }`}
              >
                Monthly
              </button>
            </div>
          )}
        </div>

        {/* Plot area */}
        <div className="flex-1 min-h-0">
          {isLoading && <p className="text-sm text-gray-400 p-4">Loading…</p>}
          {error && <p className="text-sm text-red-500 p-4">{error.message}</p>}

          {/* Qdaily tab */}
          {tab === "qdaily" && !isLoading && hasData && (
            <div className="flex flex-col h-full">
              <div className="flex-1 min-h-0 px-1">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
                    <CartesianGrid {...GRID_PROPS} />
                    <XAxis
                      dataKey="date"
                      tickFormatter={dateTickFmt}
                      {...(spanDays > 365 * 2
                        ? { ticks: makeYearTicks(chartData) }
                        : { interval: Math.max(1, Math.floor(chartData.length / 8)) }
                      )}
                      {...XAXIS_PROPS}
                    />
                    <YAxis {...makeYAxisProps(logScale, unit, symlogTicks)} />
                    <Tooltip
                      labelFormatter={(d) => d}
                      formatter={(v, seriesKey) => [
                        tooltipValFmt(v, logScale),
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
              </div>
              {/* Range slider */}
              {displayLen > (monthly ? 24 : 365) && (
                <div className="px-3 pb-1 shrink-0">
                  <RangeSlider
                    min={0}
                    max={displayLen - 1}
                    start={effectiveRange[0]}
                    end={effectiveRange[1]}
                    onChange={setRange}
                  />
                  <div className="flex justify-between text-[9px] text-gray-400 mt-0.5">
                    <span>{displayDates[effectiveRange[0]]}</span>
                    <span>{displayDates[effectiveRange[1]]}</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Qstat tab */}
          {tab === "qstat" && !isLoading && hasData && (
            <QstatPlot data={data} activeSeries={activeSeries} monthly={monthly} unit={unit} />
          )}

          {/* R² tab */}
          {tab === "r2" && !isLoading && showHexTab && (
            <div className="flex flex-col h-full">
              {/* VIC/LSTM toggle */}
              <div className="flex items-center justify-center gap-1 py-1 text-xs bg-gray-50 border-b border-gray-100 shrink-0">
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
              <div className="flex-1 min-h-0">
                <HexDensityPlot obs={hexObs} sim={hexSim} simLabel={hexLabel} unit={unit} />
              </div>
            </div>
          )}

          {!isLoading && !hasData && (
            <p className="text-sm text-gray-400 italic p-4">No timeseries data.</p>
          )}
        </div>
      </div>

      {/* Section C: Bottom tables with tabs */}
      <div className="shrink-0 border-t border-gray-200 max-h-48 overflow-hidden flex flex-col">
        <div className="flex items-center gap-1 px-2 py-1 bg-gray-50 border-b border-gray-100 shrink-0">
          <button
            onClick={() => setBottomTab("peaks")}
            className={`px-2 py-0.5 rounded text-xs transition-colors ${
              bottomTab === "peaks"
                ? "bg-blue-100 text-blue-700 font-medium"
                : "bg-gray-200 hover:bg-gray-300 text-gray-600"
            }`}
          >
            Annual Peaks
          </button>
          {layerKey === "training_watersheds" && (
            <button
              onClick={() => setBottomTab("nse")}
              className={`px-2 py-0.5 rounded text-xs transition-colors ${
                bottomTab === "nse"
                  ? "bg-blue-100 text-blue-700 font-medium"
                  : "bg-gray-200 hover:bg-gray-300 text-gray-600"
              }`}
            >
              All Watersheds
            </button>
          )}
        </div>
        <div className="flex-1 min-h-0 overflow-hidden">
          {bottomTab === "peaks" && (
            <AnnualMaxTable data={data} layerKey={layerKey} onSelectYear={jumpToWaterYear} />
          )}
          {bottomTab === "nse" && layerKey === "training_watersheds" && (
            <WatershedNseTable selectedId={polygonId} onSelectBasin={onSelectBasin} />
          )}
        </div>
      </div>
    </div>
  );
}
