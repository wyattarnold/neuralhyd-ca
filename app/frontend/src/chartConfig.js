/** Single source of truth for chart series labels, colors, and stroke widths. */
export const CHART_SERIES = {
  obs:       { label: "Observed", color: "#3b3b3d", width: 1.2 },
  vic:       { label: "VIC-Sim",  color: "#668bdc", width: 1.2 },
  lstm_pred: { label: "LSTM",     color: "#e89c0f", width: 1.2 },
};

/** Convenience maps derived from CHART_SERIES for components that need separate lookups. */
export const SERIES_COLORS = Object.fromEntries(
  Object.entries(CHART_SERIES).map(([k, v]) => [k, v.color])
);
export const SERIES_LABELS = Object.fromEntries(
  Object.entries(CHART_SERIES).map(([k, v]) => [k, v.label])
);

// ---------------------------------------------------------------------------
// Quick-jump year options shown above the time-range slider
// ---------------------------------------------------------------------------
export const JUMP_YEARS = [1, 5, 10, 20];

// ---------------------------------------------------------------------------
// Default time-window (days) when no obs bounds are available
// ---------------------------------------------------------------------------
export const DEFAULT_WINDOW_DAYS = 365 * 5;

// ---------------------------------------------------------------------------
// CFS → acre-feet conversion: 1 CFS flowing for 1 day = 1.9835 AF
// ---------------------------------------------------------------------------
export const CFS_TO_AF_DAY = 1.9835;

// ---------------------------------------------------------------------------
// Symlog transform — handles zeros gracefully for log-scale display.
// sign(x) * log10(1 + |x|)
// ---------------------------------------------------------------------------
export function symlog(x) {
  if (x == null) return null;
  return Math.sign(x) * Math.log10(1 + Math.abs(x));
}

export function symlogInv(y) {
  if (y == null) return null;
  return Math.sign(y) * (Math.pow(10, Math.abs(y)) - 1);
}

/**
 * Build symlog tick values dynamically from 0 up to the next power of 10
 * above dataMax.  Returns an array of symlog-transformed tick positions.
 */
export function makeSymlogTicks(dataMax) {
  const ticks = [0];
  if (dataMax > 0) {
    let v = 1;
    const ceil = Math.pow(10, Math.ceil(Math.log10(dataMax + 1)));
    while (v <= ceil) { ticks.push(symlog(v)); v *= 10; }
  }
  return ticks;
}

/** Format a number with commas, no decimals. */
function commaFmt(v) {
  return Math.round(v).toLocaleString("en-US");
}

// ---------------------------------------------------------------------------
// Shared Recharts axis / legend props
// ---------------------------------------------------------------------------
export const XAXIS_PROPS = {
  tick: { fontSize: 10 },
  minTickGap: 50,
};

export const GRID_PROPS = {
  strokeDasharray: "3 3",
  stroke: "#e5e7eb",
};

export const LEGEND_PROPS = {
  verticalAlign: "top",
  height: 20,
  iconSize: 10,
  wrapperStyle: { fontSize: 11 },
};

// ---------------------------------------------------------------------------
// YAxis props factory.
// When logScale=true uses symlog-transformed values — caller must also
// transform data values via symlog() before passing to the chart.
// ---------------------------------------------------------------------------
export function makeYAxisProps(logScale = false, unit = "CFS", symlogTicks) {
  const label = { value: unit, angle: -90, position: "insideLeft", fontSize: 10, offset: 8 };

  if (!logScale) {
    return {
      tick: { fontSize: 10 },
      tickFormatter: commaFmt,
      width: 56,
      label,
    };
  }
  return {
    tick: { fontSize: 10 },
    tickFormatter: (v) => commaFmt(symlogInv(v)),
    width: 56,
    label,
    ticks: symlogTicks ?? makeSymlogTicks(100_000),
    domain: [0, "dataMax"],
    type: "number",
  };
}

/**
 * Tooltip value formatter for symlog charts.
 * Pass logScale so it applies the inverse transform when needed.
 */
export function tooltipValFmt(v, logScale = false) {
  if (v == null) return "—";
  const real = logScale ? symlogInv(v) : v;
  return commaFmt(real);
}

// ---------------------------------------------------------------------------
// Monthly aggregation — CFS daily → AF monthly
// ---------------------------------------------------------------------------
const _ALL_SERIES = ["obs", "vic", "lstm_pred", "lstm_fast", "lstm_slow"];

export function aggregateMonthly(data) {
  if (!data?.dates?.length) return data;

  const months = new Map(); // "YYYY-MM" → { sums: {key: total}, counts: {key: n} }

  for (let i = 0; i < data.dates.length; i++) {
    const ym = data.dates[i].slice(0, 7);
    if (!months.has(ym)) months.set(ym, { sums: {}, counts: {} });
    const bucket = months.get(ym);
    for (const k of _ALL_SERIES) {
      const v = data[k]?.[i];
      if (v != null) {
        bucket.sums[k] = (bucket.sums[k] || 0) + v * CFS_TO_AF_DAY;
        bucket.counts[k] = (bucket.counts[k] || 0) + 1;
      }
    }
  }

  const result = { dates: [] };
  for (const k of _ALL_SERIES) result[k] = [];

  for (const [ym, bucket] of months) {
    result.dates.push(ym + "-15");
    for (const k of _ALL_SERIES) {
      result[k].push(bucket.sums[k] != null ? Math.round(bucket.sums[k] * 10) / 10 : null);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Dynamic X-axis date tick formatter
// Switches granularity based on the number of points visible:
//   > 2 years  →  "2010"
//   > 4 months →  "Jan '10"
//   ≤ 4 months →  "Jan 15"
// ---------------------------------------------------------------------------
const _MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

export function makeDateTickFormatter(spanDays) {
  if (spanDays > 365 * 2) {
    return (d) => d?.slice(0, 4) ?? "";
  }
  if (spanDays > 120) {
    return (d) => {
      if (!d) return "";
      const [y, m] = d.split("-");
      return `${_MONTHS[parseInt(m) - 1]} '${y.slice(2)}`;
    };
  }
  return (d) => {
    if (!d) return "";
    const [, m, day] = d.split("-");
    return `${_MONTHS[parseInt(m) - 1]} ${parseInt(day)}`;
  };
}

// ---------------------------------------------------------------------------
// Compute explicit tick date values — one per year (first data point of each
// calendar year). Pass the returned array to <XAxis ticks={...} /> when the
// visible span exceeds ~2 years to avoid duplicate year labels.
// ---------------------------------------------------------------------------
export function makeYearTicks(chartData) {
  const seen = new Set();
  const ticks = [];
  for (const pt of chartData) {
    const yr = pt.date?.slice(0, 4);
    if (yr && !seen.has(yr)) {
      seen.add(yr);
      ticks.push(pt.date);
    }
  }
  return ticks;
}