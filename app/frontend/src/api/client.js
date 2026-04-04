const BASE = "/api";

async function fetchJSON(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const fetchLayers = () => fetchJSON("/layers");
export const fetchLayerGeoJSON = (key) => fetchJSON(`/layers/${key}/geojson`);
export const fetchTimeseries = (layerKey, polygonId) =>
  fetchJSON(`/timeseries/${layerKey}/${polygonId}`);
export const fetchCaOutline = () => fetchJSON("/layers/ca_outline/geojson");
export const fetchStaticAttrs = (basinId) => fetchJSON(`/layers/static_attrs/${basinId}`);
