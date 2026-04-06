import { useEffect, useRef, useMemo, useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { MapContainer, TileLayer, GeoJSON, useMap, useMapEvents, Rectangle } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { fetchLayerGeoJSON, fetchCaOutline } from "../api/client";

// California center
const CA_CENTER = [37.5, -119.5];
const CA_ZOOM = 6;

// --- Color modes ---
const TIER_COLORS = {
  1: { stroke: "#c2410c", fill: "#fed7aa" },
  2: { stroke: "#15803d", fill: "#bbf7d0" },
  3: { stroke: "#1d4ed8", fill: "#bfdbfe" },
};
const TIER_DEFAULT = { stroke: "#6b7280", fill: "#e5e7eb" };

function nseColor(nse) {
  if (nse == null || isNaN(nse)) return { fill: "#d1d5db", stroke: "#9ca3af" };
  const clamped = Math.max(-0.5, Math.min(1.0, nse));
  const t = (clamped + 0.5) / 1.5;
  let r, g, b;
  if (t < 0.33) {
    const f = t / 0.33;
    r = 220; g = Math.round(60 + 160 * f); b = 60;
  } else if (t < 0.66) {
    const f = (t - 0.33) / 0.33;
    r = Math.round(220 - 180 * f); g = Math.round(220 - 40 * f); b = Math.round(60 + 60 * f);
  } else {
    const f = (t - 0.66) / 0.34;
    r = Math.round(40 - 20 * f); g = Math.round(180 - 100 * f); b = Math.round(120 + 115 * f);
  }
  const fill = `rgb(${r},${g},${b})`;
  const stroke = `rgb(${Math.max(0, r - 40)},${Math.max(0, g - 40)},${Math.max(0, b - 40)})`;
  return { fill, stroke };
}

const LAYER_COLORS = { huc8: "#0891b2", huc10: "#059669" };

function defaultStyle(layerKey, feature, colorMode) {
  if (layerKey === "training_watersheds" && feature) {
    if (colorMode === "lstm_nse" || colorMode === "lstm_single_nse" || colorMode === "vic_nse") {
      const nse = feature.properties?.[colorMode];
      const c = nseColor(nse);
      return { color: c.stroke, weight: 1.5, opacity: 0.8, fillOpacity: 0.55, fillColor: c.fill };
    }
    const t = Math.round(feature.properties?.tier);
    const c = TIER_COLORS[t] ?? TIER_DEFAULT;
    return { color: c.stroke, weight: 1.5, opacity: 0.8, fillOpacity: 0.18, fillColor: c.fill };
  }
  return {
    color: LAYER_COLORS[layerKey] || "#6366f1",
    weight: 1.5,
    opacity: 0.7,
    fillOpacity: 0.08,
    fillColor: LAYER_COLORS[layerKey] || "#6366f1",
  };
}

function selectedStyle() {
  return {
    color: "#dc2626",
    weight: 3,
    opacity: 1,
    fillOpacity: 0.4,
    fillColor: "#fca5a5",
  };
}

// --- KGE Legend ---
function NSELegend({ colorMode }) {
  const label = colorMode === "lstm_nse" ? "LSTM Dual NSE"
    : colorMode === "lstm_single_nse" ? "LSTM Single NSE"
    : "VIC NSE";
  const stops = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0];
  return (
    <div className="bg-white/95 rounded shadow px-2 py-1.5 text-xs border border-gray-200">
      <div className="font-medium text-gray-700 mb-1">{label}</div>
      <div className="flex items-center">
        {stops.map((v, i) => {
          if (i === stops.length - 1) return null;
          const c = nseColor((v + stops[i + 1]) / 2);
          return (
            <div key={v} className="flex flex-col items-center" style={{ width: 28 }}>
              <div style={{ width: 28, height: 12, background: c.fill }} />
              <span className="text-[8px] text-gray-500 mt-0.5">{v}</span>
            </div>
          );
        })}
        <span className="text-[8px] text-gray-500 -ml-1">{stops[stops.length - 1]}</span>
      </div>
      <div className="mt-0.5 flex items-center gap-1">
        <div style={{ width: 12, height: 12, background: "#d1d5db", border: "1px solid #9ca3af" }} />
        <span className="text-[8px] text-gray-500">No data</span>
      </div>
    </div>
  );
}

function TierLegend() {
  return (
    <div className="bg-white/95 rounded shadow px-2 py-1.5 text-xs border border-gray-200">
      <div className="font-medium text-gray-700 mb-1">Tier</div>
      {[[1, "Rainfall", TIER_COLORS[1]], [2, "Transitional", TIER_COLORS[2]], [3, "Snow", TIER_COLORS[3]]].map(
        ([t, label, c]) => (
          <div key={t} className="flex items-center gap-1.5 py-0.5">
            <div style={{ width: 14, height: 10, background: c.fill, border: `1.5px solid ${c.stroke}`, borderRadius: 2 }} />
            <span className="text-gray-600">{label}</span>
          </div>
        ),
      )}
    </div>
  );
}

// Legend is rendered outside MapContainer — see the bottom of the component.

/** Invalidate map size when container resizes. */
function ResizeWatcher() {
  const map = useMap();
  useEffect(() => {
    const ro = new ResizeObserver(() => map.invalidateSize());
    ro.observe(map.getContainer());
    return () => ro.disconnect();
  }, [map]);
  return null;
}

function ClickAway({ onDismiss }) {
  useMapEvents({
    click: () => onDismiss(),
  });
  return null;
}

// --- Overview minimap ---
// Center/zoom for minimap: must show all of CA (lat ~32.5–42, lon ~-124.5–-114)
const MINI_CENTER = [37.2, -119.5];
const MINI_ZOOM = 3;

function OverviewMinimap({ mainMap, caOutline }) {
  const [bounds, setBounds] = useState(null);

  useEffect(() => {
    if (!mainMap) return;
    const update = () => setBounds(mainMap.getBounds());
    update();
    mainMap.on("moveend zoomend", update);
    return () => mainMap.off("moveend zoomend", update);
  }, [mainMap]);

  return (
    <div className="w-20 h-25 border border-gray-300 rounded shadow bg-white overflow-hidden">
      <MapContainer
        center={MINI_CENTER}
        zoom={MINI_ZOOM}
        zoomControl={false}
        dragging={false}
        scrollWheelZoom={false}
        doubleClickZoom={false}
        touchZoom={false}
        attributionControl={false}
        className="h-full w-full"
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          maxZoom={19}
        />
        {caOutline && (
          <GeoJSON
            key="minimap-ca"
            data={caOutline}
            style={{ color: "#374151", weight: 2, opacity: 0.9, fillOpacity: 0 }}
            interactive={false}
          />
        )}
        {bounds && (
          <Rectangle
            bounds={bounds}
            pathOptions={{ color: "#dc2626", weight: 2, fillOpacity: 0.1, fillColor: "#fca5a5" }}
          />
        )}
      </MapContainer>
    </div>
  );
}

export default function WatershedMap({ layerKey, selectedId, onSelect, colorMode }) {
  const geoRef = useRef(null);
  const mapRef = useRef(null);
  const [mapReady, setMapReady] = useState(false);

  // Picker state for overlapping polygons
  const [picker, setPicker] = useState(null); // { latlng, items: [{id, name}] } | null
  const pickerRef = useRef(null);

  const { data: geojson, isLoading } = useQuery({
    queryKey: ["geojson", layerKey],
    queryFn: () => fetchLayerGeoJSON(layerKey),
    staleTime: 10 * 60 * 1000,
  });

  // CA outline (always)
  const { data: caOutline } = useQuery({
    queryKey: ["ca_outline"],
    queryFn: fetchCaOutline,
    staleTime: Infinity,
  });

  // HUC-10 underlay when training_watersheds is active
  const { data: huc10Geojson } = useQuery({
    queryKey: ["geojson", "huc10"],
    queryFn: () => fetchLayerGeoJSON("huc10"),
    staleTime: 10 * 60 * 1000,
    enabled: layerKey === "training_watersheds",
  });

  // Determine the id field for this layer
  const idField = useMemo(() => {
    if (!geojson?.features?.length) return null;
    const props = geojson.features[0].properties;
    for (const key of ["huc8", "huc10", "Pour Point ID"]) {
      if (key in props) return key;
    }
    return Object.keys(props)[0];
  }, [geojson]);

  // Build a lookup from id → feature for hit-testing
  const featureLookup = useMemo(() => {
    if (!geojson?.features || !idField) return new Map();
    const m = new Map();
    for (const f of geojson.features) {
      m.set(String(f.properties[idField]), f);
    }
    return m;
  }, [geojson, idField]);

  // Style function
  const styleFunc = useMemo(
    () => (feature) => {
      if (!idField) return defaultStyle(layerKey, feature, colorMode);
      const fid = String(feature.properties[idField]);
      return fid === String(selectedId)
        ? selectedStyle()
        : defaultStyle(layerKey, feature, colorMode);
    },
    [layerKey, selectedId, idField, colorMode],
  );

  // Find all features containing a latlng point
  const findFeaturesAt = useCallback(
    (latlng) => {
      if (!geojson?.features || !idField) return [];
      const pt = L.latLng(latlng);
      const hits = [];
      for (const f of geojson.features) {
        const layer = L.geoJSON(f);
        let inside = false;
        layer.eachLayer((l) => {
          if (l.getBounds && l.getBounds().contains(pt)) {
            if (l instanceof L.Polygon || l instanceof L.MultiPolygon) {
              inside = inside || pointInPolygonLayers(pt, l.getLatLngs());
            }
          }
        });
        if (inside) {
          const fid = String(f.properties[idField]);
          const name = f.properties.name || fid;
          hits.push({ id: fid, name });
        }
      }
      return hits;
    },
    [geojson, idField],
  );

  const onEachFeature = useMemo(
    () => (feature, layer) => {
      if (!idField) return;
      const fid = String(feature.properties[idField]);
      const name = feature.properties.name || fid;
      const props = feature.properties;

      layer.bindTooltip(name, { sticky: true, className: "watershed-tooltip" });
      layer.on("click", (e) => {
        L.DomEvent.stopPropagation(e);
        const hits = findFeaturesAt(e.latlng);
        if (hits.length <= 1) {
          setPicker(null);
          onSelect(layerKey, fid, name, props);
        } else {
          setPicker({ latlng: e.latlng, items: hits });
        }
      });
    },
    [layerKey, idField, onSelect, findFeaturesAt],
  );

  // Close picker when layer changes
  useEffect(() => setPicker(null), [layerKey]);

  // Manage the Leaflet popup for the picker
  useEffect(() => {
    if (!picker || !mapRef.current) {
      if (pickerRef.current) {
        pickerRef.current.remove();
        pickerRef.current = null;
      }
      return;
    }

    const container = document.createElement("div");
    container.className = "picker-popup";
    container.style.cssText = "max-height:200px;overflow-y:auto;min-width:140px;";

    for (const item of picker.items) {
      const btn = document.createElement("button");
      btn.textContent = item.name;
      btn.style.cssText =
        "display:block;width:100%;text-align:left;padding:4px 8px;border:none;background:none;cursor:pointer;font-size:12px;border-bottom:1px solid #e5e7eb;";
      btn.onmouseenter = () => (btn.style.background = "#eff6ff");
      btn.onmouseleave = () => (btn.style.background = "none");
      btn.onclick = () => {
        setPicker(null);
        const feat = geojson?.features?.find(
          (f) => String(f.properties[idField]) === item.id,
        );
        onSelect(layerKey, item.id, item.name, feat?.properties ?? {});
      };
      container.appendChild(btn);
    }

    const popup = L.popup({ closeButton: true, className: "picker-leaflet-popup" })
      .setLatLng(picker.latlng)
      .setContent(container)
      .openOn(mapRef.current);

    popup.on("remove", () => setPicker(null));
    pickerRef.current = popup;

    return () => {
      popup.remove();
      pickerRef.current = null;
    };
  }, [picker, layerKey, onSelect]);

  const geoKey = `${layerKey}-${selectedId || "none"}-${colorMode}`;

  const caStyle = { color: "#374151", weight: 2, opacity: 0.6, fillOpacity: 0 };
  const huc10Style = { color: "#9ca3af", weight: 0.7, opacity: 0.35, fillOpacity: 0, dashArray: "4 3" };

  return (
    <div className="h-full w-full relative">
      <MapContainer
        center={CA_CENTER}
        zoom={CA_ZOOM}
        className="h-full w-full"
        ref={(m) => {
          mapRef.current = m;
          if (m && !mapReady) setMapReady(true);
        }}
      >
        <TileLayer
          attribution='USGS TNM | &copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
          url="https://basemap.nationalmap.gov/arcgis/rest/services/USGSHydroCached/MapServer/tile/{z}/{y}/{x}"
          maxZoom={16}
        />
        <ResizeWatcher />
        {picker && <ClickAway onDismiss={() => setPicker(null)} />}

        {/* CA outline */}
        {caOutline && <GeoJSON key="ca-outline" data={caOutline} style={caStyle} interactive={false} />}

        {/* HUC-10 underlay */}
        {layerKey === "training_watersheds" && huc10Geojson && (
          <GeoJSON key="huc10-underlay" data={huc10Geojson} style={huc10Style} interactive={false} />
        )}

        {/* Active layer */}
        {geojson && (
          <GeoJSON
            key={geoKey}
            data={geojson}
            style={styleFunc}
            onEachFeature={onEachFeature}
            ref={geoRef}
          />
        )}
        {isLoading && (
          <div className="absolute top-3 right-3 z-[1000] bg-white/90 px-3 py-1 rounded shadow text-sm">
            Loading…
          </div>
        )}
      </MapContainer>

      {/* Legend — positioned outside MapContainer to avoid Leaflet z-index issues */}
      {layerKey === "training_watersheds" && (
        <div className="absolute bottom-4 right-4 z-[1000]">
          {(colorMode === "lstm_nse" || colorMode === "lstm_single_nse" || colorMode === "vic_nse") ? (
            <NSELegend colorMode={colorMode} />
          ) : colorMode === "tier" ? (
            <TierLegend />
          ) : null}
        </div>
      )}

      {/* Overview minimap */}
      <div className="absolute bottom-4 left-4 z-[1000]">
        {mapReady && <OverviewMinimap mainMap={mapRef.current} caOutline={caOutline} />}
      </div>
    </div>
  );
}

// --- Point-in-polygon helpers (works with Leaflet's nested LatLng arrays) ---

function pointInRing(pt, ring) {
  // ring is an array of L.LatLng
  let inside = false;
  const x = pt.lng, y = pt.lat;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i].lng, yi = ring[i].lat;
    const xj = ring[j].lng, yj = ring[j].lat;
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

function pointInPolygonLayers(pt, latlngs) {
  // latlngs can be: [[ring]] for Polygon or [[[ring]]] for MultiPolygon
  // Leaflet normalises differently — handle both
  if (!latlngs || !latlngs.length) return false;

  // Check if first element is a LatLng (simple ring)
  if (latlngs[0] instanceof L.LatLng || (latlngs[0] && typeof latlngs[0].lat === "number")) {
    return pointInRing(pt, latlngs);
  }

  // Array of rings (polygon with holes) or array of polygons (multi)
  for (const sub of latlngs) {
    if (pointInPolygonLayers(pt, sub)) return true;
  }
  return false;
}
