import { useRef, useEffect, useMemo } from "react";

/**
 * Log-log hex-binned density scatter of observed vs simulated.
 * Renders to a <canvas> for performance. Always square with 1:1 line.
 *
 * Props:
 *   obs       - array of observed values (CFS, same length as sim)
 *   sim       - array of simulated values (CFS)
 *   simLabel  - string label for toggle button display
 */

const HEX_RADIUS = 5;
const SQRT3 = Math.sqrt(3);
const PAD = { top: 8, right: 12, bottom: 32, left: 40 };

// Viridis-ish stops for density coloring
const COLORS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function densityColor(frac) {
  // frac in [0,1] → interpolate through COLORS
  const t = Math.max(0, Math.min(1, frac)) * (COLORS.length - 1);
  const i = Math.min(Math.floor(t), COLORS.length - 2);
  const f = t - i;
  const r = Math.round(lerp(COLORS[i][0], COLORS[i + 1][0], f));
  const g = Math.round(lerp(COLORS[i][1], COLORS[i + 1][1], f));
  const b = Math.round(lerp(COLORS[i][2], COLORS[i + 1][2], f));
  return `rgb(${r},${g},${b})`;
}

function hexBin(points, radius, plotW, plotH) {
  // Flat-top hex grid
  const dx = radius * 2;
  const dy = radius * SQRT3;
  const bins = new Map();

  for (const [px, py] of points) {
    // Hex grid col/row
    const col = Math.round(px / dx);
    const row = Math.round(py / dy - (col % 2 === 0 ? 0 : 0.5));
    const key = `${col},${row}`;
    bins.set(key, (bins.get(key) || 0) + 1);
  }

  // Convert back to pixel centres
  const result = [];
  for (const [key, count] of bins) {
    const [c, r] = key.split(",").map(Number);
    const cx = c * dx;
    const cy = (r + (c % 2 === 0 ? 0 : 0.5)) * dy;
    result.push({ cx, cy, count });
  }
  return result;
}

function drawHex(ctx, cx, cy, r) {
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
}

export default function HexDensityPlot({ obs, sim, simLabel, unit = "CFS" }) {
  const canvasRef = useRef(null);

  // Filter to pairs where both are non-negative (symlog handles zeros)
  const pairs = useMemo(() => {
    if (!obs || !sim) return [];
    const out = [];
    const len = Math.min(obs.length, sim.length);
    for (let i = 0; i < len; i++) {
      const o = obs[i];
      const s = sim[i];
      if (o != null && s != null && o >= 0 && s >= 0) {
        out.push([o, s]);
      }
    }
    return out;
  }, [obs, sim]);

  // Compute R² and NSE on the raw (non-log) values
  const metrics = useMemo(() => {
    if (pairs.length < 2) return null;
    const n = pairs.length;
    let sumO = 0, sumS = 0, sumOO = 0, sumSS = 0, sumOS = 0;
    for (const [o, s] of pairs) {
      sumO += o; sumS += s;
      sumOO += o * o; sumSS += s * s; sumOS += o * s;
    }
    const meanO = sumO / n;
    // R²
    const num = n * sumOS - sumO * sumS;
    const den = Math.sqrt((n * sumOO - sumO * sumO) * (n * sumSS - sumS * sumS));
    const r2 = den > 0 ? (num / den) ** 2 : NaN;
    // NSE
    let ssRes = 0, ssTot = 0;
    for (const [o, s] of pairs) {
      ssRes += (o - s) ** 2;
      ssTot += (o - meanO) ** 2;
    }
    const nse = ssTot > 0 ? 1 - ssRes / ssTot : NaN;
    return { r2, nse };
  }, [pairs]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;

    // Use the element's CSS size
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Clear
    ctx.clearRect(0, 0, w, h);

    const plotW = w - PAD.left - PAD.right;
    const plotH = h - PAD.top - PAD.bottom;

    if (pairs.length < 2 || plotW < 20 || plotH < 20) {
      ctx.fillStyle = "#9ca3af";
      ctx.font = "11px system-ui";
      ctx.textAlign = "center";
      ctx.fillText("No data", w / 2, h / 2);
      return;
    }

    // Compute symlog range (union so axes match for square)
    // symlog(x) = sign(x) * log10(1 + |x|)
    const sl = (v) => Math.sign(v) * Math.log10(1 + Math.abs(v));
    const allVals = pairs.flatMap(([o, s]) => [o, s]);
    const maxVal = Math.max(...allVals);
    const slMax = sl(maxVal);
    // Ticks at powers of 10: 0, 1, 10, 100, ...
    const tickValues = [0];
    { let tv = 1; while (tv <= maxVal * 1.5) { tickValues.push(tv); tv *= 10; } }

    const scale = (v) => sl(v) / slMax;
    const toX = (v) => PAD.left + scale(v) * plotW;
    const toY = (v) => PAD.top + plotH - scale(v) * plotH;

    // Map pairs to pixel coords
    const pixelPts = pairs.map(([o, s]) => [
      toX(o) - PAD.left,
      toY(s) - PAD.top,
    ]);

    // Hex-bin
    const bins = hexBin(pixelPts, HEX_RADIUS, plotW, plotH);
    const maxCount = Math.max(...bins.map((b) => b.count));
    const logMaxCount = Math.log(maxCount + 1);

    // Draw plot area background
    ctx.fillStyle = "#f9fafb";
    ctx.fillRect(PAD.left, PAD.top, plotW, plotH);

    // 1:1 line
    ctx.save();
    ctx.beginPath();
    ctx.rect(PAD.left, PAD.top, plotW, plotH);
    ctx.clip();
    ctx.strokeStyle = "#9ca3af";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top + plotH);
    ctx.lineTo(PAD.left + plotW, PAD.top);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Draw hexagons
    ctx.save();
    ctx.beginPath();
    ctx.rect(PAD.left, PAD.top, plotW, plotH);
    ctx.clip();
    for (const { cx, cy, count } of bins) {
      const frac = Math.log(count + 1) / logMaxCount;
      ctx.fillStyle = densityColor(frac);
      drawHex(ctx, PAD.left + cx, PAD.top + cy, HEX_RADIUS);
      ctx.fill();
    }
    ctx.restore();

    // Axes
    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top);
    ctx.lineTo(PAD.left, PAD.top + plotH);
    ctx.lineTo(PAD.left + plotW, PAD.top + plotH);
    ctx.stroke();

    // Tick labels
    ctx.fillStyle = "#6b7280";
    ctx.font = "9px system-ui";
    for (const tv of tickValues) {
      const frac = sl(tv) / slMax;
      const label = tv.toLocaleString("en-US");
      // X axis (Observed)
      const x = PAD.left + frac * plotW;
      ctx.textAlign = "center";
      ctx.fillText(label, x, PAD.top + plotH + 14);
      // Y axis (Simulated)
      const y = PAD.top + plotH - frac * plotH;
      ctx.textAlign = "right";
      ctx.fillText(label, PAD.left - 4, y + 3);
    }

    // Axis labels
    ctx.fillStyle = "#374151";
    ctx.font = "10px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(`Observed (${unit})`, PAD.left + plotW / 2, h - 2);
    ctx.save();
    ctx.translate(10, PAD.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`${simLabel} (${unit})`, 0, 0);
    ctx.restore();

    // N label + metrics
    ctx.fillStyle = "#9ca3af";
    ctx.font = "9px system-ui";
    ctx.textAlign = "right";
    ctx.fillText(`n=${pairs.length.toLocaleString()}`, PAD.left + plotW - 2, PAD.top + 12);

    if (metrics) {
      ctx.fillStyle = "#374151";
      ctx.font = "bold 9px system-ui";
      ctx.textAlign = "left";
      const r2Str = isNaN(metrics.r2) ? "R²=—" : `R²=${metrics.r2.toFixed(3)}`;
      const nseStr = isNaN(metrics.nse) ? "NSE=—" : `NSE=${metrics.nse.toFixed(3)}`;
      ctx.fillText(r2Str, PAD.left + 3, PAD.top + 12);
      ctx.fillText(nseStr, PAD.left + 3, PAD.top + 23);
    }
  }, [pairs, simLabel, metrics, unit]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full"
      style={{ display: "block" }}
    />
  );
}
