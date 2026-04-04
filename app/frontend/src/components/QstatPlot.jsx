import { useMemo, useState, useCallback } from "react";
import { SERIES_COLORS, SERIES_LABELS, GRID_PROPS, makeYAxisProps, symlog, tooltipValFmt, CFS_TO_AF_DAY, makeSymlogTicks } from "../chartConfig";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ComposedChart,
} from "recharts";

const MONTH_LABELS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"];
const DOY_TICKS = [0, 31, 61, 92, 122, 153, 182, 213, 244, 274, 305, 335];

/**
 * Water-year day-of-year statistics plot.
 * Shows median and togglable percentile bands for obs + sim (vic or lstm).
 */
export default function QstatPlot({ data, activeSeries, monthly = false, unit = "CFS" }) {
  const [show5_95, setShow5_95] = useState(true);
  const [show1_99, setShow1_99] = useState(false);
  const [logScale, setLogScale] = useState(false);
  const [hiddenSeries, setHiddenSeries] = useState(new Set());

  const handleLegendClick = useCallback((entry) => {
    // entry.dataKey is like "obs_med" — extract base key
    const base = entry.dataKey?.replace(/_med$/, "");
    if (!base) return;
    setHiddenSeries((prev) => {
      const next = new Set(prev);
      if (next.has(base)) next.delete(base);
      else next.add(base);
      return next;
    });
  }, []);

  // stats[seriesKey][bin] = { median, p5, p95, p1, p99 }
  // Daily mode: 366 bins (water-year DOY). Monthly mode: 12 bins (water-year month).
  const stats = useMemo(() => {
    if (!data?.dates) return {};
    const result = {};
    const keys = activeSeries.filter((k) => k !== "lstm_fast" && k !== "lstm_slow");

    for (const key of keys) {
      const arr = data[key];
      if (!arr) continue;

      if (monthly) {
        // Group daily values by (waterYear, waterMonth), sum to AF
        const wyMonthTotals = new Map(); // "WY-WM" → sum
        for (let i = 0; i < data.dates.length; i++) {
          const v = arr[i];
          if (v == null) continue;
          const d = new Date(data.dates[i]);
          const m = d.getMonth();
          const wy = m >= 9 ? d.getFullYear() + 1 : d.getFullYear();
          const wm = m >= 9 ? m - 9 : m + 3; // Oct=0 ... Sep=11
          const k2 = `${wy}-${wm}`;
          wyMonthTotals.set(k2, (wyMonthTotals.get(k2) || 0) + v * CFS_TO_AF_DAY);
        }
        // Group by water month across all years
        const buckets = Array.from({ length: 12 }, () => []);
        for (const [k2, total] of wyMonthTotals) {
          const wm = parseInt(k2.split("-")[1]);
          buckets[wm].push(total);
        }
        const monthStats = buckets.map((vals) => {
          const sorted = vals.sort((a, b) => a - b);
          const n = sorted.length;
          if (n === 0) return null;
          const pct = (p) => sorted[Math.min(Math.floor(p * n), n - 1)];
          return { median: pct(0.5), p5: pct(0.05), p95: pct(0.95), p1: pct(0.01), p99: pct(0.99) };
        });
        result[key] = monthStats;
      } else {
        // Daily: group by water-year DOY (Oct 1 = day 0)
        const buckets = Array.from({ length: 366 }, () => []);
        for (let i = 0; i < data.dates.length; i++) {
          const v = arr[i];
          if (v == null) continue;
          const d = new Date(data.dates[i]);
          const m = d.getMonth();
          let doy;
          if (m >= 9) {
            const octFirst = new Date(d.getFullYear(), 9, 1);
            doy = Math.floor((d - octFirst) / 86400000);
          } else {
            const octFirst = new Date(d.getFullYear() - 1, 9, 1);
            doy = Math.floor((d - octFirst) / 86400000);
          }
          if (doy >= 0 && doy < 366) buckets[doy].push(v);
        }

        const dayStats = [];
        for (let doy = 0; doy < 366; doy++) {
          const vals = buckets[doy].sort((a, b) => a - b);
          const n = vals.length;
          if (n === 0) { dayStats.push(null); continue; }
          const pct = (p) => vals[Math.min(Math.floor(p * n), n - 1)];
          dayStats.push({ median: pct(0.5), p5: pct(0.05), p95: pct(0.95), p1: pct(0.01), p99: pct(0.99) });
        }
        result[key] = dayStats;
      }
    }
    return result;
  }, [data, activeSeries, monthly]);

  const chartData = useMemo(() => {
    const seriesKeys = Object.keys(stats);
    if (!seriesKeys.length) return [];
    const nBins = monthly ? 12 : 366;
    const out = [];
    for (let b = 0; b < nBins; b++) {
      const pt = monthly ? { month: b } : { doy: b };
      for (const key of seriesKeys) {
        const s = stats[key]?.[b];
        if (s) {
          if (logScale) {
            pt[`${key}_med`] = symlog(s.median);
            pt[`${key}_5_95`] = [symlog(s.p5), symlog(s.p95)];
            pt[`${key}_1_99`] = [symlog(s.p1), symlog(s.p99)];
          } else {
            pt[`${key}_med`] = s.median;
            pt[`${key}_5_95`] = [s.p5, s.p95];
            pt[`${key}_1_99`] = [s.p1, s.p99];
          }
        }
      }
      out.push(pt);
    }
    return out;
  }, [stats, logScale, monthly]);

  const seriesKeys = Object.keys(stats);

  // Dynamic symlog ticks from data max
  const symlogTicks = useMemo(() => {
    if (!logScale) return undefined;
    let maxVal = 0;
    for (const key of seriesKeys) {
      const arr = stats[key];
      if (!arr) continue;
      for (const s of arr) {
        if (s) {
          if (s.p99 > maxVal) maxVal = s.p99;
          if (s.p95 > maxVal) maxVal = s.p95;
          if (s.median > maxVal) maxVal = s.median;
        }
      }
    }
    return makeSymlogTicks(maxVal);
  }, [logScale, stats, seriesKeys]);

  if (!chartData.length) {
    return <div className="flex items-center justify-center h-full text-sm text-gray-400 italic">No data</div>;
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toggles */}
      <div className="flex items-center gap-3 px-3 py-1 text-xs bg-gray-50 border-b border-gray-100 shrink-0">
        <label className="flex items-center gap-1 cursor-pointer">
          <input type="checkbox" checked={show5_95} onChange={(e) => setShow5_95(e.target.checked)} className="w-3 h-3" />
          5–95th %
        </label>
        <label className="flex items-center gap-1 cursor-pointer">
          <input type="checkbox" checked={show1_99} onChange={(e) => setShow1_99(e.target.checked)} className="w-3 h-3" />
          1–99th %
        </label>
        <label className="flex items-center gap-1 cursor-pointer ml-2">
          <input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)} className="w-3 h-3" />
          Log scale
        </label>
      </div>
      <div className="flex-1 min-h-0 px-1">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 8, right: 8, bottom: 4, left: 4 }}>
            <CartesianGrid {...GRID_PROPS} />
            <XAxis
              dataKey={monthly ? "month" : "doy"}
              ticks={monthly ? [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] : DOY_TICKS}
              tickFormatter={(d) => MONTH_LABELS[monthly ? d : DOY_TICKS.indexOf(d)] ?? ""}
              tick={{ fontSize: 10 }}
            />
            <YAxis {...makeYAxisProps(logScale, unit, symlogTicks)} />
            <Tooltip
              labelFormatter={(d) => {
                if (monthly) return MONTH_LABELS[d] ?? "";
                const idx = DOY_TICKS.findIndex((t, i) => (DOY_TICKS[i + 1] ?? 366) > d);
                return `${MONTH_LABELS[idx >= 0 ? idx : 11]} day ${d - (DOY_TICKS[idx >= 0 ? idx : 11] || 0) + 1}`;
              }}
              formatter={(v, name) => {
                if (Array.isArray(v)) {
                  const lo = tooltipValFmt(v[0], logScale);
                  const hi = tooltipValFmt(v[1], logScale);
                  return [`${lo} – ${hi}`, name];
                }
                return [tooltipValFmt(v, logScale), name];
              }}
            />
            <Legend
              verticalAlign="top"
              height={24}
              iconSize={10}
              wrapperStyle={{ fontSize: 11, cursor: "pointer" }}
              onClick={handleLegendClick}
              formatter={(value, entry) => (
                <span style={{ opacity: hiddenSeries.has(entry.dataKey?.replace(/_med$/, "")) ? 0.35 : 1 }}>
                  {value}
                </span>
              )}
            />
            {/* Percentile bands */}
            {seriesKeys.map((key) => {
              const color = SERIES_COLORS[key] || "#6366f1";
              const hidden = hiddenSeries.has(key);
              return [
                show1_99 && (
                  <Area
                    key={`${key}_1_99`}
                    dataKey={`${key}_1_99`}
                    stroke="none"
                    fill={color}
                    fillOpacity={hidden ? 0 : 0.06}
                    name={`${SERIES_LABELS[key] || key} 1-99%`}
                    isAnimationActive={false}
                    legendType="none"
                  />
                ),
                show5_95 && (
                  <Area
                    key={`${key}_5_95`}
                    dataKey={`${key}_5_95`}
                    stroke="none"
                    fill={color}
                    fillOpacity={hidden ? 0 : 0.13}
                    name={`${SERIES_LABELS[key] || key} 5-95%`}
                    isAnimationActive={false}
                    legendType="none"
                  />
                ),
              ];
            })}
            {/* Medians */}
            {seriesKeys.map((key) => (
              <Line
                key={`${key}_med`}
                type="monotone"
                dataKey={`${key}_med`}
                name={`${SERIES_LABELS[key] || key}`}
                stroke={SERIES_COLORS[key] || "#6366f1"}
                strokeWidth={1.5}
                strokeOpacity={hiddenSeries.has(key) ? 0 : 1}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
