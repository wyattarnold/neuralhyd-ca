import { useMemo, useState, useCallback, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ResponsiveContainer,
  LineChart,
  AreaChart,
  Area,
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
  CHART_SERIES, QDAILY_HIDDEN_DEFAULT, FLOW_SEP_SERIES, TOOLTIP_PROPS,
  GRID_PROPS, XAXIS_PROPS, LEGEND_PROPS,
  JUMP_YEARS, DEFAULT_WINDOW_DAYS, makeDateTickFormatter, makeYAxisProps,
  makeYearTicks, symlog, tooltipValFmt, aggregateMonthly, makeSymlogTicks,
} from "../chartConfig";

// --- Draggable range slider ---
// Scroll-wheel zooms around cursor; drag thumbs / middle bar to pan.
// When thumbs are close together, the selected bar grows taller as a grab target.
const MIN_SPAN = 30; // minimum index span
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

  const clampRange = useCallback(
    (s, e) => {
      let ns = Math.max(min, Math.min(s, max - MIN_SPAN));
      let ne = Math.min(max, Math.max(e, min + MIN_SPAN));
      if (ne - ns < MIN_SPAN) { ns = Math.max(min, ne - MIN_SPAN); }
      return [ns, ne];
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
          onChange(clampRange(valFromX(ev.clientX), end));
        } else if (h === "end") {
          onChange(clampRange(start, valFromX(ev.clientX)));
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
    [start, end, min, max, onChange, valFromX, clampRange],
  );

  // Scroll-wheel zoom: zoom in/out centred on cursor position
  // Must use non-passive addEventListener so preventDefault() works on trackpads
  const onWheel = useCallback(
    (e) => {
      e.preventDefault();
      const span = end - start;
      const zoomFactor = e.deltaY > 0 ? 1.15 : 1 / 1.15; // scroll down = zoom out
      const newSpan = Math.max(MIN_SPAN, Math.min(max - min, Math.round(span * zoomFactor)));
      // Centre around cursor position
      const rect = trackRef.current.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const anchor = min + frac * (max - min);
      const fracInRange = span > 0 ? (anchor - start) / span : 0.5;
      let newS = Math.round(anchor - fracInRange * newSpan);
      let newE = newS + newSpan;
      if (newS < min) { newS = min; newE = min + newSpan; }
      if (newE > max) { newE = max; newS = max - newSpan; }
      onChange([newS, newE]);
    },
    [start, end, min, max, onChange],
  );

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [onWheel]);

  const leftPct = pct(start);
  const rightPct = pct(end);
  // When thumbs are close (< 4% of track), make the bar taller so it's easier to grab
  const barNarrow = (rightPct - leftPct) < 4;

  return (
    <div
      ref={trackRef}
      className="relative h-5 select-none touch-none"
    >
      <div className="absolute top-2 left-0 right-0 h-1 rounded bg-gray-200" />
      <div
        className={`absolute rounded bg-blue-400 cursor-grab active:cursor-grabbing ${
          barNarrow ? "top-0.5 h-4" : "top-2 h-1"
        }`}
        style={{ left: `${leftPct}%`, right: `${100 - rightPct}%`, minWidth: "8px" }}
        onPointerDown={(e) => onPointerDown(e, "middle")}
      />
      <div
        className="absolute top-0.5 w-4 h-4 -ml-2 rounded-full bg-blue-600 border-2 border-white shadow cursor-ew-resize z-10"
        style={{ left: `${leftPct}%` }}
        onPointerDown={(e) => onPointerDown(e, "start")}
      />
      <div
        className="absolute top-0.5 w-4 h-4 -ml-2 rounded-full bg-blue-600 border-2 border-white shadow cursor-ew-resize z-10"
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
  if (props?.lstm_nse != null) rows.push(["LSTM Dual NSE", Number(props.lstm_nse).toFixed(3)]);
  if (props?.lstm_single_nse != null) rows.push(["LSTM Single NSE", Number(props.lstm_single_nse).toFixed(3)]);
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
  const [hiddenSeries, setHiddenSeries] = useState(new Set(QDAILY_HIDDEN_DEFAULT));
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
    for (let i = s; i <= e; i++) {
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
  const hasLstmSingle = displayData?.lstm_single_pred?.some((v) => v != null);
  const hasVic = displayData?.vic?.some((v) => v != null);
  const showHexTab = hasObs && (hasLstm || hasLstmSingle || hasVic);

  const hexObs = useMemo(() => {
    if (!displayData?.obs) return [];
    return displayData.obs;
  }, [displayData]);

  const hexSim = useMemo(() => {
    if (!displayData?.[hexTarget]) return [];
    return displayData[hexTarget];
  }, [displayData, hexTarget]);

  const hexLabel = hexTarget === "lstm_pred" ? "LSTM Dual"
    : hexTarget === "lstm_single_pred" ? "LSTM Single"
    : "VIC-Sim";

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
      // Stay on current tab if it's a timeseries view; otherwise switch to qdaily
      setTab((prev) => (prev === "flowsep" || prev === "qdaily") ? prev : "qdaily");
    },
    [dates],
  );

  const hasData = activeSeries.length > 0;

  // Flow separation data availability
  const hasObsBaseflow = data?.obs_baseflow?.some((v) => v != null);
  const hasLstmComponents = data?.lstm_fast?.some((v) => v != null) && data?.lstm_slow?.some((v) => v != null);
  const hasVicComponents = data?.vic_baseflow?.some((v) => v != null) && data?.vic_surface?.some((v) => v != null);
  const showFlowSep = hasObsBaseflow && layerKey === "training_watersheds";

  // Flow separation toggle groups
  const [flowSepGroups, setFlowSepGroups] = useState({ lstm: true, vic: false });
  const toggleFlowSepGroup = useCallback((group) => {
    setFlowSepGroups((prev) => ({ ...prev, [group]: !prev[group] }));
  }, []);

  // Flow separation chart data
  const flowSepData = useMemo(() => {
    if (!showFlowSep || !displayDates.length) return [];
    const [s, e] = effectiveRange;
    const slice = [];
    for (let i = s; i <= e; i++) {
      const pt = { date: displayDates[i] };

      // Observed components (always shown)
      const obsTotal = displayData?.obs?.[i] ?? null;
      const obsBf = displayData?.obs_baseflow?.[i] ?? null;
      pt.obs_baseflow = obsBf;
      pt.obs_quickflow = (obsTotal != null && obsBf != null) ? Math.max(0, obsTotal - obsBf) : null;

      // LSTM dual components
      pt.lstm_slow = displayData?.lstm_slow?.[i] ?? null;
      pt.lstm_fast = displayData?.lstm_fast?.[i] ?? null;

      // VIC components
      pt.vic_baseflow = displayData?.vic_baseflow?.[i] ?? null;
      pt.vic_surface = displayData?.vic_surface?.[i] ?? null;

      slice.push(pt);
    }
    return slice;
  }, [showFlowSep, displayDates, effectiveRange, displayData]);

  const TABS = [
    { id: "qdaily", label: "Qdaily" },
  ];
  if (showFlowSep) TABS.push({ id: "flowsep", label: "Flow Sep" });
  TABS.push({ id: "qstat", label: "Qstat" });
  if (showHexTab) TABS.push({ id: "r2", label: "R²" });

  return (
    <div className="flex flex-col h-full bg-paper text-gray-900">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-200 bg-paper shrink-0">
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
          {/* Flow Sep tab controls */}
          {tab === "flowsep" && hasData && (
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
              {hasLstmComponents && (
                <button
                  onClick={() => toggleFlowSepGroup("lstm")}
                  className={`px-1.5 py-0.5 rounded transition-colors ${
                    flowSepGroups.lstm ? "bg-orange-100 text-orange-800 font-medium" : "bg-gray-200 hover:bg-gray-300"
                  }`}
                >
                  LSTM
                </button>
              )}
              {hasVicComponents && (
                <button
                  onClick={() => toggleFlowSepGroup("vic")}
                  className={`px-1.5 py-0.5 rounded transition-colors ${
                    flowSepGroups.vic ? "bg-green-100 text-green-800 font-medium" : "bg-gray-200 hover:bg-gray-300"
                  }`}
                >
                  VIC
                </button>
              )}
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
                      {...TOOLTIP_PROPS}
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
                          type="linear"
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

          {/* Flow Separation tab */}
          {tab === "flowsep" && !isLoading && showFlowSep && (
            <div className="flex flex-col h-full">
              <div className="flex-1 min-h-0 px-1">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={flowSepData} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
                    <CartesianGrid {...GRID_PROPS} />
                    <XAxis
                      dataKey="date"
                      tickFormatter={dateTickFmt}
                      {...(spanDays > 365 * 2
                        ? { ticks: makeYearTicks(flowSepData) }
                        : { interval: Math.max(1, Math.floor(flowSepData.length / 8)) }
                      )}
                      {...XAXIS_PROPS}
                    />
                    <YAxis {...makeYAxisProps(false, unit)} />
                    <Tooltip
                      {...TOOLTIP_PROPS}
                      labelFormatter={(d) => d}
                      formatter={(v, seriesKey) => [
                        v != null ? Math.round(v).toLocaleString("en-US") : "—",
                        FLOW_SEP_SERIES[seriesKey]?.label ?? seriesKey,
                      ]}
                    />
                    <Legend
                      {...LEGEND_PROPS}
                    />
                    {/* Observed baseflow + quickflow (always visible, stacked) */}
                    <Area
                      type="linear"
                      dataKey="obs_baseflow"
                      name={FLOW_SEP_SERIES.obs_baseflow.label}
                      stackId="obs"
                      stroke={FLOW_SEP_SERIES.obs_baseflow.color}
                      fill={FLOW_SEP_SERIES.obs_baseflow.color}
                      fillOpacity={0.45}
                      strokeWidth={1.8}
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Area
                      type="linear"
                      dataKey="obs_quickflow"
                      name={FLOW_SEP_SERIES.obs_quickflow.label}
                      stackId="obs"
                      stroke={FLOW_SEP_SERIES.obs_quickflow.color}
                      fill={FLOW_SEP_SERIES.obs_quickflow.color}
                      fillOpacity={0.08}
                      strokeWidth={0.5}
                      dot={false}
                      isAnimationActive={false}
                    />
                    {/* LSTM dual components (toggleable, stacked) */}
                    {flowSepGroups.lstm && hasLstmComponents && (
                      <>
                        <Area
                          type="linear"
                          dataKey="lstm_slow"
                          name={FLOW_SEP_SERIES.lstm_slow.label}
                          stackId="lstm"
                          stroke={FLOW_SEP_SERIES.lstm_slow.color}
                          fill={FLOW_SEP_SERIES.lstm_slow.color}
                          fillOpacity={0.45}
                          strokeWidth={1.8}
                          dot={false}
                          isAnimationActive={false}
                        />
                        <Area
                          type="linear"
                          dataKey="lstm_fast"
                          name={FLOW_SEP_SERIES.lstm_fast.label}
                          stackId="lstm"
                          stroke={FLOW_SEP_SERIES.lstm_fast.color}
                          fill={FLOW_SEP_SERIES.lstm_fast.color}
                          fillOpacity={0.08}
                          strokeWidth={0.5}
                          dot={false}
                          isAnimationActive={false}
                        />
                      </>
                    )}
                    {/* VIC components (toggleable, stacked) */}
                    {flowSepGroups.vic && hasVicComponents && (
                      <>
                        <Area
                          type="linear"
                          dataKey="vic_baseflow"
                          name={FLOW_SEP_SERIES.vic_baseflow.label}
                          stackId="vic"
                          stroke={FLOW_SEP_SERIES.vic_baseflow.color}
                          fill={FLOW_SEP_SERIES.vic_baseflow.color}
                          fillOpacity={0.45}
                          strokeWidth={1.8}
                          dot={false}
                          isAnimationActive={false}
                        />
                        <Area
                          type="linear"
                          dataKey="vic_surface"
                          name={FLOW_SEP_SERIES.vic_surface.label}
                          stackId="vic"
                          stroke={FLOW_SEP_SERIES.vic_surface.color}
                          fill={FLOW_SEP_SERIES.vic_surface.color}
                          fillOpacity={0.08}
                          strokeWidth={0.5}
                          dot={false}
                          isAnimationActive={false}
                        />
                      </>
                    )}
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              {/* Range slider */}
              {displayLen > 365 && (
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
                      hexTarget === "lstm_pred" ? "bg-orange-100 text-orange-700 font-medium" : "bg-gray-200 hover:bg-gray-300"
                    }`}
                  >
                    vs LSTM Dual
                  </button>
                )}
                {hasLstmSingle && (
                  <button
                    onClick={() => setHexTarget("lstm_single_pred")}
                    className={`px-2 py-0.5 rounded transition-colors ${
                      hexTarget === "lstm_single_pred" ? "bg-purple-100 text-purple-700 font-medium" : "bg-gray-200 hover:bg-gray-300"
                    }`}
                  >
                    vs LSTM Single
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
