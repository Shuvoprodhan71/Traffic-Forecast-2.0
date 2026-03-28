/**
 * Unit tests for the Traffic Dashboard feature engineering and inference logic.
 * These tests verify the core mathematical operations used in the in-browser
 * prediction pipeline (scaler, RF tree traversal, feature construction).
 */
import { describe, expect, it } from "vitest";

// ── Scaler helpers (mirrors client/src/pages/Home.tsx) ──────────
interface ScalerParams {
  data_min: number[]; data_max: number[];
  scale: number[]; min: number[];
  feature_range: [number, number]; n_features: number;
}

function minMaxScale(value: number, idx: number, p: ScalerParams): number {
  return value * p.scale[idx] + p.min[idx];
}
function minMaxInverse(scaled: number, idx: number, p: ScalerParams): number {
  return (scaled - p.min[idx]) / p.scale[idx];
}

// ── RF helpers (mirrors client/src/pages/Home.tsx) ──────────────
interface RFTree {
  children_left: number[]; children_right: number[];
  threshold: number[]; feature: number[]; value: number[];
}

function predictTree(tree: RFTree, x: number[]): number {
  let node = 0;
  while (tree.children_left[node] !== -1) {
    node = x[tree.feature[node]] <= tree.threshold[node]
      ? tree.children_left[node] : tree.children_right[node];
  }
  return tree.value[node];
}

// ── Sample scaler (speed feature only, range [0,1]) ─────────────
const mockScaler: ScalerParams = {
  data_min:      [0],
  data_max:      [80],
  scale:         [0.0125],   // 1 / (80 - 0)
  min:           [0],
  feature_range: [0, 1],
  n_features:    1,
};

describe("MinMaxScaler", () => {
  it("scales 0 mph to 0.0", () => {
    expect(minMaxScale(0, 0, mockScaler)).toBeCloseTo(0.0, 5);
  });

  it("scales 80 mph to 1.0", () => {
    expect(minMaxScale(80, 0, mockScaler)).toBeCloseTo(1.0, 5);
  });

  it("scales 40 mph to 0.5", () => {
    expect(minMaxScale(40, 0, mockScaler)).toBeCloseTo(0.5, 5);
  });

  it("inverse-scales 0.5 back to 40 mph", () => {
    const scaled = minMaxScale(40, 0, mockScaler);
    expect(minMaxInverse(scaled, 0, mockScaler)).toBeCloseTo(40, 5);
  });

  it("round-trips arbitrary value through scale → inverse", () => {
    const original = 63.7;
    const scaled   = minMaxScale(original, 0, mockScaler);
    const restored = minMaxInverse(scaled, 0, mockScaler);
    expect(restored).toBeCloseTo(original, 5);
  });
});

describe("RF tree prediction", () => {
  // A minimal decision tree: root splits on feature[0] <= 0.5
  //   node 0: feature=0, threshold=0.5, left=1, right=2
  //   node 1: leaf, value=30  (low speed)
  //   node 2: leaf, value=65  (high speed)
  const mockTree: RFTree = {
    children_left:  [1, -1, -1],
    children_right: [2, -1, -1],
    threshold:      [0.5, -2, -2],
    feature:        [0, -2, -2],
    value:          [0, 30, 65],
  };

  it("routes to left leaf when feature <= threshold", () => {
    expect(predictTree(mockTree, [0.3])).toBe(30);
  });

  it("routes to right leaf when feature > threshold", () => {
    expect(predictTree(mockTree, [0.8])).toBe(65);
  });

  it("routes to left leaf at exact threshold boundary", () => {
    expect(predictTree(mockTree, [0.5])).toBe(30);
  });
});

describe("Cyclical time encoding", () => {
  it("hour_sin for hour=0 is 0", () => {
    expect(Math.sin(2 * Math.PI * 0 / 24)).toBeCloseTo(0, 5);
  });

  it("hour_cos for hour=0 is 1", () => {
    expect(Math.cos(2 * Math.PI * 0 / 24)).toBeCloseTo(1, 5);
  });

  it("hour=6 and hour=18 have opposite sin values", () => {
    const sin6  = Math.sin(2 * Math.PI * 6  / 24);
    const sin18 = Math.sin(2 * Math.PI * 18 / 24);
    expect(sin6).toBeCloseTo(-sin18, 5);
  });

  it("is_weekend is 1 for Sunday (dow=0)", () => {
    const dow = 0;
    const isWeekend = dow === 0 || dow === 6 ? 1 : 0;
    expect(isWeekend).toBe(1);
  });

  it("is_weekend is 0 for Monday (dow=1)", () => {
    const dow = 1;
    const isWeekend = dow === 0 || dow === 6 ? 1 : 0;
    expect(isWeekend).toBe(0);
  });
});

describe("CDN model URL constants", () => {
  const CDN_BASE = "https://d2xsxph8kpxj0f.cloudfront.net/310419663026719317/dtmbEsMR9U4rWX5HMG9VW9";
  const CDN_SCALER = `${CDN_BASE}/scaler_params_c39cbc52.json`;
  const CDN_RF     = `${CDN_BASE}/rf_model_80180a8e.json`;
  const CDN_LSTM   = `${CDN_BASE}/lstm_model_faca9604.json`;

  it("scaler CDN URL is a valid HTTPS CloudFront URL", () => {
    expect(CDN_SCALER).toMatch(/^https:\/\/.*cloudfront\.net\/.+\.json$/);
  });

  it("RF CDN URL is a valid HTTPS CloudFront URL", () => {
    expect(CDN_RF).toMatch(/^https:\/\/.*cloudfront\.net\/.+\.json$/);
  });

  it("LSTM CDN URL is a valid HTTPS CloudFront URL", () => {
    expect(CDN_LSTM).toMatch(/^https:\/\/.*cloudfront\.net\/.+\.json$/);
  });

  it("all three CDN URLs are distinct", () => {
    const urls = [CDN_SCALER, CDN_RF, CDN_LSTM];
    expect(new Set(urls).size).toBe(3);
  });
});

describe("Scaler JSON structure validation", () => {
  // Mirrors the actual scaler_params.json produced from scaler.pkl
  const realScalerParams: ScalerParams = {
    data_min:      [0.0, -1.0, -1.0, -0.9749279121818236, -0.9009688679024191, 0.0],
    data_max:      [70.0, 1.0, 1.0, 0.9749279121818236, 1.0, 1.0],
    scale:         [0.014285714285714285, 0.5, 0.5, 0.512858431636277, 0.5260475418008436, 1.0],
    min:           [0.0, 0.5, 0.5, 0.5, 0.47395245819915655, 0.0],
    feature_range: [0, 1],
    n_features:    6,
  };

  it("has exactly 6 features", () => {
    expect(realScalerParams.n_features).toBe(6);
  });

  it("feature_range is [0, 1]", () => {
    expect(realScalerParams.feature_range).toEqual([0, 1]);
  });

  it("all scale/min arrays have length 6", () => {
    expect(realScalerParams.scale.length).toBe(6);
    expect(realScalerParams.min.length).toBe(6);
    expect(realScalerParams.data_min.length).toBe(6);
    expect(realScalerParams.data_max.length).toBe(6);
  });

  it("speed feature (idx=0) max is 70 mph", () => {
    expect(realScalerParams.data_max[0]).toBe(70.0);
  });

  it("speed feature (idx=0) min is 0 mph", () => {
    expect(realScalerParams.data_min[0]).toBe(0.0);
  });

  it("scales 35 mph (midpoint) to ~0.5 using real scaler", () => {
    const scaled = minMaxScale(35, 0, realScalerParams);
    expect(scaled).toBeCloseTo(0.5, 2);
  });

  it("round-trips 60 mph through real scaler", () => {
    const scaled   = minMaxScale(60, 0, realScalerParams);
    const restored = minMaxInverse(scaled, 0, realScalerParams);
    expect(restored).toBeCloseTo(60, 4);
  });

  it("is_weekend feature (idx=5) has scale=1 and min=0 (binary passthrough)", () => {
    expect(realScalerParams.scale[5]).toBe(1.0);
    expect(realScalerParams.min[5]).toBe(0.0);
  });
});

// ── CSV parsing helpers (mirrors Batch section in Home.tsx) ─────
interface BatchRow {
  rowIndex: number;
  timestamp: string;
  speeds: number[];
  rfForecast: number | null;
  lstmForecast: number | null;
  error: string | null;
}

function isHeaderLabel(s: string): boolean {
  return /[a-zA-Z]/.test(s);
}
function looksLikeTimestamp(s: string): boolean {
  return /[:\-]/.test(s) && !isNaN(Date.parse(s));
}

function parseBatchCSV(text: string): BatchRow[] {
  const lines = text.trim().split(/\r?\n/);
  const rows: BatchRow[] = [];
  const firstCell = lines[0].split(",")[0].trim();
  const startLine = isHeaderLabel(firstCell) ? 1 : 0;
  for (let i = startLine; i < lines.length; i++) {
    const cells = lines[i].split(",").map((c) => c.trim());
    if (cells.length === 0 || (cells.length === 1 && !cells[0])) continue;
    let ts = new Date().toISOString().slice(0, 16);
    let speedCells: string[];
    if (cells.length >= 13 && looksLikeTimestamp(cells[0])) {
      ts = cells[0];
      speedCells = cells.slice(1, 13);
    } else {
      speedCells = cells.slice(0, 12);
    }
    const speeds = speedCells.map((s) => parseFloat(s));
    if (speeds.length !== 12 || speeds.some((s) => isNaN(s))) {
      rows.push({ rowIndex: i - startLine + 1, timestamp: ts, speeds: [], rfForecast: null, lstmForecast: null, error: `Row ${i - startLine + 1}: expected 12 numeric speed values` });
    } else {
      rows.push({ rowIndex: i - startLine + 1, timestamp: ts, speeds, rfForecast: null, lstmForecast: null, error: null });
    }
  }
  return rows;
}

describe("CSV batch parsing", () => {
  it("parses 12-column CSV without header", () => {
    const csv = "65.2,64.8,63.5,60.1,55.4,50.2,45.1,40.5,38.2,35.5,33.1,30.0";
    const rows = parseBatchCSV(csv);
    expect(rows).toHaveLength(1);
    expect(rows[0].error).toBeNull();
    expect(rows[0].speeds).toHaveLength(12);
    expect(rows[0].speeds[0]).toBeCloseTo(65.2, 1);
    expect(rows[0].speeds[11]).toBeCloseTo(30.0, 1);
  });

  it("parses CSV with timestamp as first column", () => {
    const csv = "2012-03-01 08:00,65.2,64.8,63.5,60.1,55.4,50.2,45.1,40.5,38.2,35.5,33.1,30.0";
    const rows = parseBatchCSV(csv);
    expect(rows).toHaveLength(1);
    expect(rows[0].error).toBeNull();
    expect(rows[0].timestamp).toBe("2012-03-01 08:00");
    expect(rows[0].speeds).toHaveLength(12);
  });

  it("skips header row when first cell is non-numeric", () => {
    const csv = "timestamp,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12\n2012-03-01 08:00,65.2,64.8,63.5,60.1,55.4,50.2,45.1,40.5,38.2,35.5,33.1,30.0";
    const rows = parseBatchCSV(csv);
    expect(rows).toHaveLength(1);
    expect(rows[0].error).toBeNull();
  });

  it("parses multiple rows correctly", () => {
    const csv = [
      "65.2,64.8,63.5,60.1,55.4,50.2,45.1,40.5,38.2,35.5,33.1,30.0",
      "62.1,60.5,58.3,55.0,52.4,49.8,47.2,44.6,42.0,39.4,36.8,34.2",
      "55.0,53.2,51.4,49.6,47.8,46.0,44.2,42.4,40.6,38.8,37.0,35.2",
    ].join("\n");
    const rows = parseBatchCSV(csv);
    expect(rows).toHaveLength(3);
    rows.forEach((r) => expect(r.error).toBeNull());
  });

  it("marks row as error when fewer than 12 speeds are provided", () => {
    const csv = "65.2,64.8,63.5,60.1,55.4,50.2";
    const rows = parseBatchCSV(csv);
    expect(rows).toHaveLength(1);
    expect(rows[0].error).not.toBeNull();
    expect(rows[0].speeds).toHaveLength(0);
  });

  it("marks row as error when a speed value is non-numeric", () => {
    const csv = "65.2,64.8,63.5,60.1,55.4,50.2,45.1,40.5,38.2,35.5,33.1,BAD";
    const rows = parseBatchCSV(csv);
    expect(rows).toHaveLength(1);
    expect(rows[0].error).not.toBeNull();
  });

  it("assigns sequential rowIndex values starting from 1", () => {
    const csv = [
      "65.2,64.8,63.5,60.1,55.4,50.2,45.1,40.5,38.2,35.5,33.1,30.0",
      "62.1,60.5,58.3,55.0,52.4,49.8,47.2,44.6,42.0,39.4,36.8,34.2",
    ].join("\n");
    const rows = parseBatchCSV(csv);
    expect(rows[0].rowIndex).toBe(1);
    expect(rows[1].rowIndex).toBe(2);
  });

  it("handles CRLF line endings correctly", () => {
    const csv = "65.2,64.8,63.5,60.1,55.4,50.2,45.1,40.5,38.2,35.5,33.1,30.0\r\n62.1,60.5,58.3,55.0,52.4,49.8,47.2,44.6,42.0,39.4,36.8,34.2";
    const rows = parseBatchCSV(csv);
    expect(rows).toHaveLength(2);
    rows.forEach((r) => expect(r.error).toBeNull());
  });

  it("ignores empty lines in the CSV", () => {
    const csv = "65.2,64.8,63.5,60.1,55.4,50.2,45.1,40.5,38.2,35.5,33.1,30.0\n\n62.1,60.5,58.3,55.0,52.4,49.8,47.2,44.6,42.0,39.4,36.8,34.2";
    const rows = parseBatchCSV(csv);
    expect(rows).toHaveLength(2);
  });
});
