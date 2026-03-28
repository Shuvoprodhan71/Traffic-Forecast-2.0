/**
 * TrafficAI Dashboard — Complete Professional Redesign
 * Design  : Premium Dark Analytics — inspired by Linear, Vercel, and Stripe dashboards
 * Fonts   : Inter (body) · Space Grotesk (display) · JetBrains Mono (data)
 * Palette : Deep black #09090b · Zinc grays · Cyan accent #06b6d4 · Emerald #10b981 · Amber #f59e0b
 * Inference: TensorFlow.js (in-browser) + JSON Random Forest — zero server needed
 */

import { useState, useEffect, useRef } from "react";
import { toast } from "sonner";
import {
  AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";
import {
  Activity, Zap, TrendingUp, Clock, Car, BarChart2,
  ChevronRight, Loader2, AlertCircle, CheckCircle2,
  Play, RotateCcw, Cpu, BrainCircuit,
  Gauge, ArrowUpRight, ArrowDownRight, Minus,
  Radio, BookOpen, Download, Database, FlaskConical,
  Upload, FileSpreadsheet, XCircle,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";

// ── Asset URLs ──────────────────────────────────────────────────
const HERO_IMG  = "https://d2xsxph8kpxj0f.cloudfront.net/310419663026719317/dtmbEsMR9U4rWX5HMG9VW9/hero-v2-b9azXmgBNgm5Q2ghNdfhNA.webp";
const RF_ICON   = "https://d2xsxph8kpxj0f.cloudfront.net/310419663026719317/dtmbEsMR9U4rWX5HMG9VW9/model-rf-icon-6dpvbmTHzGUxikcKNJdbEL.webp";
const LSTM_ICON = "https://d2xsxph8kpxj0f.cloudfront.net/310419663026719317/dtmbEsMR9U4rWX5HMG9VW9/model-lstm-icon-Br7uQZ2MFghrJwrVqUmSdr.webp";

// ── CDN Model URLs ───────────────────────────────────────────────
const CDN_SCALER = "https://d2xsxph8kpxj0f.cloudfront.net/310419663026719317/dtmbEsMR9U4rWX5HMG9VW9/scaler_params_c39cbc52.json";
const CDN_RF     = "https://d2xsxph8kpxj0f.cloudfront.net/310419663026719317/dtmbEsMR9U4rWX5HMG9VW9/rf_model_80180a8e.json";
const CDN_LSTM   = "https://d2xsxph8kpxj0f.cloudfront.net/310419663026719317/dtmbEsMR9U4rWX5HMG9VW9/lstm_model_v2_9ee63944.json";

// ── Types ───────────────────────────────────────────────────────
interface ScalerParams {
  data_min: number[]; data_max: number[];
  scale: number[]; min: number[];
  feature_range: [number, number]; n_features: number;
}
interface RFTree {
  children_left: number[]; children_right: number[];
  threshold: number[]; feature: number[]; value: number[];
}
interface RFModel { n_estimators: number; trees: RFTree[]; }
interface HistoryEntry { time: string; actual: number; rf: number; lstm: number; }
interface ModelState { loaded: boolean; loading: boolean; error: string | null; }

// Default rolling window — 12 readings (5-min intervals = 1 hour of history from METR-LA)
const DEFAULT_WINDOW: number[] = [65.2, 64.8, 63.5, 60.1, 55.4, 50.2, 45.1, 40.5, 38.2, 35.5, 33.1, 30.0];
const N_LAGS = 12;

const PIPELINE_STEPS = [
  { label: "Validating inputs",       icon: CheckCircle2 },
  { label: "Scaling features",        icon: Cpu },
  { label: "Running Random Forest",   icon: TrendingUp },
  { label: "Running LSTM network",    icon: BrainCircuit },
  { label: "Decoding predictions",    icon: Zap },
];

// ── Scaler helpers ──────────────────────────────────────────────
function minMaxScale(value: number, idx: number, p: ScalerParams): number {
  return value * p.scale[idx] + p.min[idx];
}
function minMaxInverse(scaled: number, idx: number, p: ScalerParams): number {
  return (scaled - p.min[idx]) / p.scale[idx];
}

// ── RF inference ────────────────────────────────────────────────
function predictTree(tree: RFTree, x: number[]): number {
  let node = 0;
  while (tree.children_left[node] !== -1) {
    node = x[tree.feature[node]] <= tree.threshold[node]
      ? tree.children_left[node] : tree.children_right[node];
  }
  return tree.value[node];
}
function predictRF(model: RFModel, x: number[]): number {
  const preds = model.trees.map(t => predictTree(t, x));
  return preds.reduce((a, b) => a + b, 0) / preds.length;
}

// ── Feature engineering ─────────────────────────────────────────
function buildFeatures(speeds: number[], timestamp: string, scaler: ScalerParams) {
  const dt = new Date(timestamp);
  const hour = dt.getHours(), dow = dt.getDay();
  const isWeekend = dow === 0 || dow === 6 ? 1 : 0;
  const hourSin = Math.sin(2 * Math.PI * hour / 24);
  const hourCos = Math.cos(2 * Math.PI * hour / 24);
  const dowSin  = Math.sin(2 * Math.PI * dow / 7);
  const dowCos  = Math.cos(2 * Math.PI * dow / 7);
  const sequence = speeds.map(s => {
    const raw = [s, hourSin, hourCos, dowSin, dowCos, isWeekend];
    return raw.map((v, i) => minMaxScale(v, i, scaler));
  });
  return { rfInput: sequence.flat(), lstmInput: sequence };
}

// ── Animated counter ────────────────────────────────────────────
function Counter({ to, decimals = 1 }: { to: number; decimals?: number }) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    const start = performance.now(), dur = 900;
    const tick = (now: number) => {
      const p = Math.min((now - start) / dur, 1);
      const e = 1 - Math.pow(1 - p, 4);
      setVal(to * e);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [to]);
  return <>{val.toFixed(decimals)}</>;
}

// ── Speed Arc Gauge ─────────────────────────────────────────────
function ArcGauge({ value, max = 80, color, loading }: {
  value: number; max?: number; color: string; loading?: boolean;
}) {
  const pct = Math.min(value / max, 1);
  const R = 44, cx = 56, cy = 56;
  const totalArc = 2 * Math.PI * R * (240 / 360);
  const filled = totalArc * pct;
  const angle = -120 + pct * 240;
  const nx = cx + (R - 10) * Math.cos(angle * Math.PI / 180);
  const ny = cy + (R - 10) * Math.sin(angle * Math.PI / 180);

  return (
    <svg width="112" height="90" viewBox="0 0 112 90">
      <circle cx={cx} cy={cy} r={R} fill="none"
        stroke="rgba(255,255,255,0.05)" strokeWidth="8"
        strokeDasharray={`${totalArc} ${2 * Math.PI * R - totalArc}`}
        strokeDashoffset={totalArc * (1/6)} strokeLinecap="round"
        transform={`rotate(-120 ${cx} ${cy})`} />
      {loading ? (
        <circle cx={cx} cy={cy} r={R} fill="none"
          stroke={color} strokeWidth="8" opacity="0.3"
          strokeDasharray={`${totalArc * 0.25} ${2 * Math.PI * R - totalArc * 0.25}`}
          strokeDashoffset={totalArc * (1/6)} strokeLinecap="round"
          transform={`rotate(-120 ${cx} ${cy})`}
          style={{ animation: "gaugeSpinAnim 1.2s linear infinite" }} />
      ) : (
        <circle cx={cx} cy={cy} r={R} fill="none"
          stroke={color} strokeWidth="8"
          strokeDasharray={`${filled} ${totalArc - filled + 2 * Math.PI * R - totalArc}`}
          strokeDashoffset={totalArc * (1/6)} strokeLinecap="round"
          transform={`rotate(-120 ${cx} ${cy})`}
          style={{
            transition: "stroke-dasharray 1.2s cubic-bezier(0.34,1.56,0.64,1)",
            filter: `drop-shadow(0 0 6px ${color}99)`,
          }} />
      )}
      {!loading && value > 0 && (
        <line x1={cx} y1={cy} x2={nx} y2={ny}
          stroke={color} strokeWidth="2" strokeLinecap="round"
          style={{ transition: "x2 1.2s cubic-bezier(0.34,1.56,0.64,1), y2 1.2s cubic-bezier(0.34,1.56,0.64,1)" }} />
      )}
      <circle cx={cx} cy={cy} r="3.5" fill={color} opacity={loading ? 0.3 : 1} />
      {loading ? (
        <rect x={cx - 14} y={cy + 10} width={28} height={10} rx={3}
          fill="rgba(255,255,255,0.04)" />
      ) : (
        <text x={cx} y={cy + 18} textAnchor="middle"
          fill="white" fontSize="15" fontWeight="700"
          fontFamily="'Space Grotesk', sans-serif">
          {value > 0 ? value.toFixed(1) : "—"}
        </text>
      )}
    </svg>
  );
}

// ── Stat Card ───────────────────────────────────────────────────
function StatCard({ icon: Icon, label, value, color, loading }: {
  icon: any; label: string; value: string; color: string; loading?: boolean;
}) {
  return (
    <div className="rounded-2xl p-5 flex items-start gap-4"
      style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)" }}>
      <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
        style={{ background: `${color}15` }}>
        <Icon size={18} style={{ color }} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium mb-1" style={{ color: "rgba(255,255,255,0.4)" }}>{label}</p>
        {loading ? (
          <div className="h-6 w-20 rounded-md animate-pulse" style={{ background: "rgba(255,255,255,0.06)" }} />
        ) : (
          <p className="text-xl font-bold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            {value}
          </p>
        )}
      </div>
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────
export default function Home() {
  const [section, setSection]       = useState<"predict" | "history" | "about" | "batch">("predict");
  // Rolling window of 12 speed readings — user adds one at a time
  const [speedWindow, setSpeedWindow] = useState<number[]>(DEFAULT_WINDOW);
  const [singleSpeed, setSingleSpeed] = useState("");
  const [timestamp, setTimestamp]   = useState(() => new Date().toISOString().slice(0, 16));
  const [running, setRunning]       = useState(false);
  const [step, setStep]             = useState(-1);
  const [rfResult, setRfResult]     = useState<number | null>(null);
  const [lstmResult, setLstmResult] = useState<number | null>(null);
  const [latestSpeed, setLatest]    = useState<number | null>(null);
  const [history, setHistory]       = useState<HistoryEntry[]>([]);
  const [predError, setPredError]   = useState<string | null>(null);
  const [initLoading, setInitLoading] = useState(true);

  const [scalerState, setScalerState] = useState<ModelState>({ loaded: false, loading: false, error: null });
  const [rfState, setRfState]         = useState<ModelState>({ loaded: false, loading: false, error: null });
  const [lstmState, setLstmState]     = useState<ModelState>({ loaded: false, loading: false, error: null });

  const scalerRef = useRef<ScalerParams | null>(null);
  const rfRef     = useRef<RFModel | null>(null);
  const lstmRef   = useRef<any>(null);

  // ── Auto-load all models from CDN on mount ──────────────────
  useEffect(() => {
    const t = setTimeout(() => setInitLoading(false), 700);
    return () => clearTimeout(t);
  }, []);

  useEffect(() => {
    let cancelled = false;

    const autoLoad = async () => {
      // 1. Scaler
      setScalerState({ loaded: false, loading: true, error: null });
      try {
        const res = await fetch(CDN_SCALER);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        scalerRef.current = await res.json();
        if (!cancelled) setScalerState({ loaded: true, loading: false, error: null });
      } catch (e: any) {
        if (!cancelled) setScalerState({ loaded: false, loading: false, error: e.message });
        return;
      }

      // 2. Random Forest
      setRfState({ loaded: false, loading: true, error: null });
      try {
        const res = await fetch(CDN_RF);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        rfRef.current = await res.json();
        if (!cancelled) setRfState({ loaded: true, loading: false, error: null });
      } catch (e: any) {
        if (!cancelled) setRfState({ loaded: false, loading: false, error: e.message });
        return;
      }

      // 3. LSTM via TF.js — use window.tf (loaded via <script> tag in index.html)
      setLstmState({ loaded: false, loading: true, error: null });
      try {
        const tf = (window as any).tf;
        if (!tf) throw new Error("TF.js not loaded yet — window.tf is undefined");
        // Manually fetch the model JSON so we can patch the weights path before TF.js sees it
        const modelRes = await fetch(CDN_LSTM);
        if (!modelRes.ok) throw new Error(`HTTP ${modelRes.status}`);
        const modelJson = await modelRes.json();
        // Ensure weights path is absolute (already patched in CDN file, but defensive)
        const weightsPath: string = modelJson.weightsManifest?.[0]?.paths?.[0] ?? "";
        const absoluteWeightsUrl = weightsPath.startsWith("http")
          ? weightsPath
          : `https://d2xsxph8kpxj0f.cloudfront.net/310419663026719317/dtmbEsMR9U4rWX5HMG9VW9/${weightsPath}`;
        // Fetch weights bin manually
        const weightsRes = await fetch(absoluteWeightsUrl);
        if (!weightsRes.ok) throw new Error(`Weights HTTP ${weightsRes.status}`);
        const weightsBuffer = await weightsRes.arrayBuffer();
        // Load model from in-memory artifacts using the TF.js 4.x single-argument ModelArtifacts API
        const modelArtifacts = {
          modelTopology: modelJson.modelTopology,
          weightSpecs: modelJson.weightsManifest[0].weights,
          weightData: weightsBuffer,
          format: modelJson.format,
          generatedBy: modelJson.generatedBy,
          convertedBy: modelJson.convertedBy,
        };
        lstmRef.current = await (tf as any).loadLayersModel(
          (tf as any).io.fromMemory(modelArtifacts)
        );
        if (!cancelled) {
          setLstmState({ loaded: true, loading: false, error: null });
          toast.success("All models loaded — ready to predict!");
        }
      } catch (e: any) {
        if (!cancelled) setLstmState({ loaded: false, loading: false, error: e.message });
      }
    };

    autoLoad();
    return () => { cancelled = true; };
  }, []);

  // Add a single speed reading to the rolling window
  const addSpeedReading = () => {
    const val = parseFloat(singleSpeed.trim());
    if (isNaN(val) || val < 0 || val > 200) {
      toast.error("Enter a valid speed (0–200 mph)");
      return;
    }
    setSpeedWindow(prev => [...prev.slice(-11), val]);
    setSingleSpeed("");
    // Auto-update timestamp to now
    setTimestamp(new Date().toISOString().slice(0, 16));
  };

  const handlePredict = async () => {
    const speeds = speedWindow;
    if (speeds.length !== N_LAGS) { toast.error("Need exactly 12 readings in the window."); return; }
    if (!scalerRef.current || !rfRef.current || !lstmRef.current) {
      toast.error("Models are still loading, please wait a moment."); return;
    }
    setRunning(true); setPredError(null); setRfResult(null); setLstmResult(null);
    const delay = (ms: number) => new Promise(r => setTimeout(r, ms));
    try {
      setStep(0); await delay(350);
      setStep(1); await delay(400);
      const { rfInput, lstmInput } = buildFeatures(speeds, timestamp, scalerRef.current!);
      setStep(2); await delay(700);
      const rfScaled = predictRF(rfRef.current!, rfInput);
      setStep(3); await delay(900);
      const tf = (window as any).tf || await import("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js" as any);
      const tensor = (tf as any).tensor3d([lstmInput]);
      const out = lstmRef.current!.predict(tensor);
      const lstmScaled = (await out.data())[0];
      tensor.dispose(); out.dispose();
      setStep(4); await delay(350);
      const rfMph   = minMaxInverse(rfScaled,   0, scalerRef.current!);
      const lstmMph = minMaxInverse(lstmScaled, 0, scalerRef.current!);
      const latest  = speeds[speeds.length - 1];
      setRfResult(rfMph); setLstmResult(lstmMph); setLatest(latest);
      const timeLabel = new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      setHistory(prev => [...prev.slice(-29), { time: timeLabel, actual: latest, rf: rfMph, lstm: lstmMph }]);
      toast.success("Prediction complete");
    } catch (e: any) {
      setPredError(e.message || "Prediction failed.");
      toast.error(e.message || "Prediction failed.");
    } finally {
      setRunning(false); setStep(-1);
    }
  };

  const modelsReady = scalerState.loaded && rfState.loaded && lstmState.loaded;
  const speedCount = speedWindow.length;
  const currentSpeed = speedWindow.length > 0 ? speedWindow[speedWindow.length - 1] : null;
  const trend = rfResult && currentSpeed
    ? rfResult > currentSpeed + 1 ? "up" : rfResult < currentSpeed - 1 ? "down" : "flat"
    : null;

  const navItems = [
    { id: "predict", icon: Gauge,          label: "Predict" },
    { id: "history", icon: BarChart2,      label: "History" },
    { id: "batch",   icon: FileSpreadsheet, label: "Batch"   },
    { id: "about",   icon: BookOpen,       label: "About"   },
  ];

  // ── Batch prediction state ──────────────────────────────────
  interface BatchRow {
    rowIndex: number;
    timestamp: string;
    speeds: number[];
    rfForecast: number | null;
    lstmForecast: number | null;
    error: string | null;
  }
  const [batchFile, setBatchFile]         = useState<File | null>(null);
  const [batchRows, setBatchRows]         = useState<BatchRow[]>([]);
  const [batchResults, setBatchResults]   = useState<BatchRow[]>([]);
  const [batchRunning, setBatchRunning]   = useState(false);
  const [batchProgress, setBatchProgress] = useState(0);
  const [batchError, setBatchError]       = useState<string | null>(null);
  const batchFileRef = useRef<HTMLInputElement>(null);

  // Detect if a cell is a text header label (contains alphabetic letters)
  const isHeaderLabel = (s: string) => /[a-zA-Z]/.test(s);
  // Detect if a cell looks like a date/time string (has dashes or colons but no letters)
  const looksLikeTimestamp = (s: string) => /[:\-]/.test(s) && !isNaN(Date.parse(s));

  // Parse uploaded CSV into BatchRow array
  const parseBatchCSV = (text: string): BatchRow[] => {
    const lines = text.trim().split(/\r?\n/);
    const rows: BatchRow[] = [];
    // Skip header row only if first cell contains letters (e.g. "timestamp", "row", "s1")
    const firstCell = lines[0].split(",")[0].trim();
    const startLine = isHeaderLabel(firstCell) ? 1 : 0;
    for (let i = startLine; i < lines.length; i++) {
      const cells = lines[i].split(",").map(c => c.trim());
      if (cells.length === 0 || (cells.length === 1 && !cells[0])) continue;
      // Support: 12 speeds only, OR timestamp + 12 speeds
      let ts = new Date().toISOString().slice(0, 16);
      let speedCells: string[];
      if (cells.length >= 13 && looksLikeTimestamp(cells[0])) {
        // First cell is a timestamp
        ts = cells[0];
        speedCells = cells.slice(1, 13);
      } else {
        speedCells = cells.slice(0, 12);
      }
      const speeds = speedCells.map(s => parseFloat(s));
      if (speeds.length !== 12 || speeds.some(s => isNaN(s))) {
        rows.push({ rowIndex: i - startLine + 1, timestamp: ts, speeds: [], rfForecast: null, lstmForecast: null, error: `Row ${i - startLine + 1}: expected 12 numeric speed values` });
      } else {
        rows.push({ rowIndex: i - startLine + 1, timestamp: ts, speeds, rfForecast: null, lstmForecast: null, error: null });
      }
    }
    return rows;
  };

  // Run batch inference
  const runBatchPredictions = async () => {
    if (!scalerRef.current || !rfRef.current || !lstmRef.current) {
      toast.error("Models are still loading, please wait."); return;
    }
    const validRows = batchRows.filter(r => !r.error);
    if (validRows.length === 0) { toast.error("No valid rows to process."); return; }
    setBatchRunning(true); setBatchProgress(0);
    const tf = (window as any).tf;
    const results: BatchRow[] = [...batchRows];
    for (let i = 0; i < results.length; i++) {
      const row = results[i];
      if (row.error) continue;
      try {
        const { rfInput, lstmInput } = buildFeatures(row.speeds, row.timestamp, scalerRef.current!);
        const rfScaled = predictRF(rfRef.current!, rfInput);
        const tensor = tf.tensor3d([lstmInput]);
        const out = lstmRef.current!.predict(tensor);
        const lstmScaled = (await out.data())[0];
        tensor.dispose(); out.dispose();
        results[i] = {
          ...row,
          rfForecast: minMaxInverse(rfScaled, 0, scalerRef.current!),
          lstmForecast: minMaxInverse(lstmScaled, 0, scalerRef.current!),
        };
      } catch (e: any) {
        results[i] = { ...row, error: e.message || "Inference failed" };
      }
      setBatchProgress(Math.round(((i + 1) / results.length) * 100));
      // Yield to UI every 5 rows
      if (i % 5 === 0) await new Promise(r => setTimeout(r, 0));
    }
    setBatchResults(results);
    setBatchRunning(false);
    const successCount = results.filter(r => r.rfForecast !== null).length;
    toast.success(`Batch complete — ${successCount} / ${results.length} rows predicted`);
  };

  // Download results as CSV
  const downloadBatchCSV = () => {
    const header = "row,timestamp,speed_1,speed_2,speed_3,speed_4,speed_5,speed_6,speed_7,speed_8,speed_9,speed_10,speed_11,speed_12,rf_forecast_mph,lstm_forecast_mph";
    const lines = batchResults.map(r =>
      r.error
        ? `${r.rowIndex},${r.timestamp},${r.speeds.join(",") || ",".repeat(11)},ERROR,ERROR`
        : `${r.rowIndex},${r.timestamp},${r.speeds.join(",")},${r.rfForecast!.toFixed(2)},${r.lstmForecast!.toFixed(2)}`
    );
    const blob = new Blob([[header, ...lines].join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url;
    a.download = `trafficai_batch_results_${Date.now()}.csv`;
    a.click(); URL.revokeObjectURL(url);
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
        @keyframes gaugeSpinAnim { to { transform: rotate(360deg); } }
        @keyframes fadeSlideUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
        @keyframes fadeIn { from { opacity:0; } to { opacity:1; } }
        @keyframes scaleUp { from { opacity:0; transform:scale(0.96); } to { opacity:1; transform:scale(1); } }
        @keyframes pulseGlow { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
        @keyframes shimmerSlide { 0%{background-position:200% 0;} 100%{background-position:-200% 0;} }
        .fade-up   { animation: fadeSlideUp 0.5s cubic-bezier(0.22,1,0.36,1) both; }
        .fade-in   { animation: fadeIn 0.4s ease both; }
        .scale-up  { animation: scaleUp 0.4s cubic-bezier(0.22,1,0.36,1) both; }
        .d1{animation-delay:0ms;} .d2{animation-delay:60ms;} .d3{animation-delay:120ms;}
        .d4{animation-delay:180ms;} .d5{animation-delay:240ms;} .d6{animation-delay:300ms;}
        .nav-item { transition: all 0.15s ease; }
        .nav-item:hover { background: rgba(255,255,255,0.05); }
        .nav-item.active { background: rgba(6,182,212,0.1); }
        .result-card { transition: transform 0.2s ease, box-shadow 0.2s ease; }
        .result-card:hover { transform: translateY(-3px); }
        .btn-predict { transition: all 0.2s ease; }
        .btn-predict:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.1); }
        .btn-predict:active:not(:disabled) { transform: translateY(0); }
        .shimmer-btn {
          background: linear-gradient(90deg, #0e7490 0%, #06b6d4 40%, #0e7490 80%);
          background-size: 200% 100%;
          animation: shimmerSlide 1.4s linear infinite;
        }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
        input[type="datetime-local"]::-webkit-calendar-picker-indicator { filter: invert(0.5); cursor: pointer; }
      `}</style>

      <div className="min-h-screen flex" style={{ background: "#09090b", fontFamily: "Inter, sans-serif" }}>

        {/* ══════════════════ SIDEBAR ══════════════════ */}
        <aside className="hidden lg:flex w-64 flex-shrink-0 flex-col sticky top-0 h-screen"
          style={{ background: "#0c0c0f", borderRight: "1px solid rgba(255,255,255,0.06)" }}>

          {/* Logo */}
          <div className="px-6 py-6" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl flex items-center justify-center"
                style={{ background: "linear-gradient(135deg, #0e7490, #06b6d4)" }}>
                <Activity size={17} color="white" />
              </div>
              <div>
                <p className="text-sm font-bold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif", letterSpacing: "-0.02em" }}>
                  TrafficAI
                </p>
                <p className="text-[10px]" style={{ color: "rgba(255,255,255,0.3)" }}>
                  METR-LA · Phase 8
                </p>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-3 space-y-1">
            {navItems.map(({ id, icon: Icon, label }) => (
              <button key={id} onClick={() => setSection(id as any)}
                className={`nav-item w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm ${section === id ? "active" : ""}`}
                style={{ color: section === id ? "#06b6d4" : "rgba(255,255,255,0.4)" }}>
                <Icon size={16} />
                <span className="font-medium">{label}</span>
                {section === id && <ChevronRight size={13} className="ml-auto" style={{ color: "#06b6d4" }} />}
              </button>
            ))}
          </nav>

          {/* Minimal model ready indicator */}
          <div className="m-3 p-3 rounded-xl flex items-center gap-2.5"
            style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
            {modelsReady ? (
              <>
                <div className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ background: "#10b981", boxShadow: "0 0 6px #10b981" }} />
                <span className="text-[11px] font-medium" style={{ color: "#10b981" }}>Models ready</span>
              </>
            ) : (
              <>
                <Loader2 size={11} className="animate-spin flex-shrink-0" style={{ color: "#06b6d4" }} />
                <span className="text-[11px]" style={{ color: "rgba(255,255,255,0.3)" }}>Loading models…</span>
              </>
            )}
          </div>
        </aside>

        {/* ══════════════════ MAIN CONTENT ══════════════════ */}
        <main className="flex-1 flex flex-col min-w-0 overflow-hidden">

          {/* ── Hero ── */}
          <div className="relative overflow-hidden" style={{ height: 220 }}>
            <img src={HERO_IMG} alt="Los Angeles Highway" className="absolute inset-0 w-full h-full object-cover" />
            <div className="absolute inset-0" style={{
              background: "linear-gradient(90deg, #09090b 0%, rgba(9,9,11,0.85) 40%, rgba(9,9,11,0.4) 100%)"
            }} />
            <div className="absolute inset-0" style={{
              background: "linear-gradient(0deg, #09090b 0%, transparent 50%)"
            }} />
            <div className="relative z-10 h-full flex items-end px-8 pb-7">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-3">
                  <span className="inline-flex items-center gap-1.5 text-[10px] font-semibold px-2.5 py-1 rounded-full"
                    style={{ background: "rgba(6,182,212,0.12)", color: "#06b6d4", border: "1px solid rgba(6,182,212,0.2)", fontFamily: "'JetBrains Mono', monospace" }}>
                    <Radio size={8} style={{ animation: "pulseGlow 2s ease infinite" }} />
                    METR-LA · 207 Sensors
                  </span>
                  <span className="inline-flex items-center gap-1.5 text-[10px] font-semibold px-2.5 py-1 rounded-full"
                    style={{ background: "rgba(16,185,129,0.1)", color: "#10b981", border: "1px solid rgba(16,185,129,0.2)", fontFamily: "'JetBrains Mono', monospace" }}>
                    <Zap size={8} />
                    In-Browser Inference
                  </span>
                </div>
                <h1 className="text-4xl font-extrabold text-white leading-none tracking-tight"
                  style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                  Traffic Speed<br />Forecasting
                </h1>
                <p className="text-sm mt-2" style={{ color: "rgba(255,255,255,0.4)" }}>
                  RF + LSTM models · Zero server required · Los Angeles highway network
                </p>
              </div>
              {/* Live gauges in hero */}
              {(rfResult !== null || running) && (
                <div className="hidden xl:flex items-end gap-10 pb-2 fade-in">
                  <div className="flex flex-col items-center gap-1">
                    <ArcGauge value={rfResult ?? 0} color="#10b981" loading={running} />
                    <span className="text-[10px] font-semibold" style={{ color: "#10b981", fontFamily: "'JetBrains Mono', monospace" }}>
                      Random Forest
                    </span>
                  </div>
                  <div className="flex flex-col items-center gap-1">
                    <ArcGauge value={lstmResult ?? 0} color="#06b6d4" loading={running} />
                    <span className="text-[10px] font-semibold" style={{ color: "#06b6d4", fontFamily: "'JetBrains Mono', monospace" }}>
                      LSTM Network
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* ── Stat Strip ── */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-px"
            style={{ background: "rgba(255,255,255,0.05)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            {[
              { icon: Car,        label: "Latest Reading",   val: speedWindow.length > 0 ? `${speedWindow[speedWindow.length - 1].toFixed(1)} mph` : "—",  color: "#f59e0b" },
              { icon: TrendingUp, label: "RF Forecast",      val: rfResult    ? `${rfResult.toFixed(1)} mph`    : "—",     color: "#10b981" },
              { icon: Zap,        label: "LSTM Forecast",    val: lstmResult  ? `${lstmResult.toFixed(1)} mph`  : "—",     color: "#06b6d4" },
              { icon: Activity,   label: "Predictions Made", val: String(history.length),                                  color: "#a78bfa" },
            ].map(({ icon: Icon, label, val, color }) => (
              <div key={label} className="flex items-center gap-3 px-5 py-4" style={{ background: "#09090b" }}>
                {initLoading ? (
                  <>
                    <div className="w-9 h-9 rounded-xl animate-pulse" style={{ background: "rgba(255,255,255,0.05)" }} />
                    <div className="space-y-2">
                      <div className="h-2 w-16 rounded animate-pulse" style={{ background: "rgba(255,255,255,0.05)" }} />
                      <div className="h-5 w-12 rounded animate-pulse" style={{ background: "rgba(255,255,255,0.05)" }} />
                    </div>
                  </>
                ) : (
                  <>
                    <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
                      style={{ background: `${color}15` }}>
                      <Icon size={16} style={{ color }} />
                    </div>
                    <div>
                      <p className="text-[10px] font-medium" style={{ color: "rgba(255,255,255,0.3)" }}>{label}</p>
                      <p className="text-lg font-bold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>{val}</p>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>

          {/* ── Page Content ── */}
          <div className="flex-1 overflow-auto p-6 lg:p-8">

            {/* ════════ PREDICT SECTION ════════ */}
            {section === "predict" && (
              <div className="grid grid-cols-1 xl:grid-cols-5 gap-6 fade-up">

                {/* Left Column: Model Loader + Inputs */}
                <div className="xl:col-span-2 space-y-5">

                  {/* Input Card */}
                  <div className="rounded-2xl p-6"
                    style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)" }}>
                    <h2 className="text-base font-bold text-white mb-5" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                      Prediction Input
                    </h2>

                    <div className="space-y-4">
                      {/* ── Rolling window input ── */}
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <Label className="text-xs font-medium" style={{ color: "rgba(255,255,255,0.4)" }}>
                            Add a speed reading (mph)
                          </Label>
                          <span className="text-[10px]" style={{
                            color: "rgba(255,255,255,0.25)",
                            fontFamily: "'JetBrains Mono', monospace",
                          }}>
                            Window: {speedWindow.length} / 12
                          </span>
                        </div>
                        {/* Single speed entry */}
                        <div className="flex gap-2">
                          <input
                            type="number"
                            min={0} max={200} step={0.1}
                            value={singleSpeed}
                            onChange={e => setSingleSpeed(e.target.value)}
                            onKeyDown={e => e.key === "Enter" && addSpeedReading()}
                            disabled={running}
                            placeholder="e.g. 45.2"
                            className="flex-1 rounded-xl text-sm p-3 focus:outline-none transition-all"
                            style={{
                              background: "rgba(255,255,255,0.04)",
                              border: "1px solid rgba(255,255,255,0.1)",
                              color: "rgba(255,255,255,0.9)",
                              fontFamily: "'JetBrains Mono', monospace",
                            }}
                          />
                          <button
                            onClick={addSpeedReading}
                            disabled={running || !singleSpeed.trim()}
                            className="px-4 rounded-xl text-sm font-bold transition-all"
                            style={{
                              background: singleSpeed.trim() ? "rgba(6,182,212,0.15)" : "rgba(255,255,255,0.04)",
                              border: `1px solid ${singleSpeed.trim() ? "rgba(6,182,212,0.4)" : "rgba(255,255,255,0.08)"}`,
                              color: singleSpeed.trim() ? "#06b6d4" : "rgba(255,255,255,0.2)",
                              cursor: singleSpeed.trim() ? "pointer" : "not-allowed",
                              fontFamily: "'Space Grotesk', sans-serif",
                            }}
                          >
                            + Add
                          </button>
                        </div>
                        <p className="text-[10px] mt-1.5" style={{ color: "rgba(255,255,255,0.2)" }}>
                          Each reading = one 5-min interval. Press Enter or + Add after each value.
                        </p>
                        {/* Rolling window visualization */}
                        <div className="mt-3 rounded-xl p-3" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.25)" }}>Speed history window</span>
                            <button
                              onClick={() => setSpeedWindow(DEFAULT_WINDOW)}
                              className="text-[10px] flex items-center gap-1 transition-colors"
                              style={{ color: "rgba(255,255,255,0.2)" }}
                              onMouseEnter={e => (e.currentTarget.style.color = "rgba(255,255,255,0.5)")}
                              onMouseLeave={e => (e.currentTarget.style.color = "rgba(255,255,255,0.2)")}
                            >
                              <RotateCcw size={9} /> Reset
                            </button>
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {speedWindow.map((v, i) => (
                              <div key={i} className="flex items-center gap-1">
                                <span
                                  className="px-2 py-0.5 rounded-lg text-[11px]"
                                  style={{
                                    background: i === speedWindow.length - 1 ? "rgba(6,182,212,0.15)" : "rgba(255,255,255,0.05)",
                                    border: `1px solid ${i === speedWindow.length - 1 ? "rgba(6,182,212,0.35)" : "rgba(255,255,255,0.07)"}`,
                                    color: i === speedWindow.length - 1 ? "#06b6d4" : "rgba(255,255,255,0.5)",
                                    fontFamily: "'JetBrains Mono', monospace",
                                  }}
                                >
                                  {v.toFixed(1)}
                                </span>
                                {i < speedWindow.length - 1 && (
                                  <span style={{ color: "rgba(255,255,255,0.1)", fontSize: 9 }}>›</span>
                                )}
                              </div>
                            ))}
                          </div>
                          <p className="text-[10px] mt-2" style={{ color: "rgba(255,255,255,0.2)" }}>
                            Highlighted = most recent (current speed). Oldest drops off as you add new readings.
                          </p>
                        </div>
                      </div>

                      <div>
                        <Label className="text-xs font-medium mb-2 block" style={{ color: "rgba(255,255,255,0.4)" }}>
                          Timestamp of last reading
                        </Label>
                        <input
                          type="datetime-local"
                          value={timestamp}
                          onChange={e => setTimestamp(e.target.value)}
                          disabled={running}
                          className="w-full rounded-xl text-xs p-3.5 focus:outline-none"
                          style={{
                            background: "rgba(255,255,255,0.03)",
                            border: "1px solid rgba(255,255,255,0.08)",
                            color: "rgba(255,255,255,0.7)",
                            fontFamily: "'JetBrains Mono', monospace",
                          }}
                        />
                      </div>

                      <button
                        onClick={handlePredict}
                        disabled={running || !modelsReady || speedCount !== 12}
                        className={`btn-predict w-full h-12 rounded-xl text-sm font-bold flex items-center justify-center gap-2 ${running ? "shimmer-btn" : ""}`}
                        style={{
                          background: running
                            ? undefined
                            : modelsReady && speedCount === 12
                            ? "linear-gradient(135deg, #0e7490, #06b6d4)"
                            : "rgba(255,255,255,0.04)",
                          color: modelsReady && speedCount === 12 ? "white" : "rgba(255,255,255,0.2)",
                          cursor: running || !modelsReady || speedCount !== 12 ? "not-allowed" : "pointer",
                          fontFamily: "'Space Grotesk', sans-serif",
                          boxShadow: modelsReady && speedCount === 12 && !running
                            ? "0 4px 20px rgba(6,182,212,0.25)" : "none",
                        }}>
                        {running
                          ? <><Loader2 size={16} className="animate-spin" /> Predicting…</>
                          : <><Play size={16} /> Run Prediction</>
                        }
                      </button>

                      {predError && (
                        <div className="flex items-start gap-2.5 p-3.5 rounded-xl fade-up"
                          style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)" }}>
                          <AlertCircle size={14} style={{ color: "#f87171", marginTop: 1 }} />
                          <p className="text-xs" style={{ color: "#fca5a5" }}>{predError}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Right Column: Pipeline + Results */}
                <div className="xl:col-span-3 space-y-5">

                  {/* Pipeline Tracker */}
                  {running && step >= 0 && (
                    <div className="rounded-2xl p-6 fade-up"
                      style={{ background: "rgba(6,182,212,0.04)", border: "1px solid rgba(6,182,212,0.15)" }}>
                      <div className="flex items-center gap-3 mb-5">
                        <div className="w-5 h-5 rounded-full border-2 animate-spin"
                          style={{ borderColor: "#06b6d4", borderTopColor: "transparent" }} />
                        <span className="text-sm font-bold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                          Running Inference Pipeline
                        </span>
                      </div>
                      <div className="space-y-3">
                        {PIPELINE_STEPS.map((s, i) => {
                          const done = i < step, cur = i === step;
                          const Icon = s.icon;
                          return (
                            <div key={i} className="flex items-center gap-3">
                              <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 transition-all duration-300"
                                style={{
                                  background: done ? "rgba(16,185,129,0.12)" : cur ? "rgba(6,182,212,0.12)" : "rgba(255,255,255,0.03)",
                                  border: `1px solid ${done ? "rgba(16,185,129,0.3)" : cur ? "rgba(6,182,212,0.3)" : "rgba(255,255,255,0.06)"}`,
                                }}>
                                {done
                                  ? <CheckCircle2 size={13} style={{ color: "#10b981" }} />
                                  : cur
                                  ? <Loader2 size={13} className="animate-spin" style={{ color: "#06b6d4" }} />
                                  : <Icon size={13} style={{ color: "rgba(255,255,255,0.2)" }} />
                                }
                              </div>
                              <span className="text-xs flex-1 transition-colors duration-300"
                                style={{ color: done ? "#10b981" : cur ? "#06b6d4" : "rgba(255,255,255,0.2)" }}>
                                {s.label}
                              </span>
                              {done && (
                                <span className="text-[10px] px-2 py-0.5 rounded-full"
                                  style={{ background: "rgba(16,185,129,0.1)", color: "#10b981", fontFamily: "'JetBrains Mono', monospace" }}>
                                  done
                                </span>
                              )}
                              {cur && (
                                <span className="text-[10px] px-2 py-0.5 rounded-full animate-pulse"
                                  style={{ background: "rgba(6,182,212,0.1)", color: "#06b6d4", fontFamily: "'JetBrains Mono', monospace" }}>
                                  running
                                </span>
                              )}
                            </div>
                          );
                        })}
                      </div>
                      <div className="mt-5 h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.05)" }}>
                        <div className="h-1.5 rounded-full transition-all duration-500"
                          style={{
                            width: `${Math.round(step / (PIPELINE_STEPS.length - 1) * 100)}%`,
                            background: "linear-gradient(90deg, #0e7490, #10b981)",
                          }} />
                      </div>
                    </div>
                  )}

                  {/* Skeleton while running */}
                  {running && (
                    <div className="grid grid-cols-2 gap-4 fade-in">
                      {[0, 1].map(i => (
                        <div key={i} className="rounded-2xl p-6"
                          style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
                          <div className="h-3 w-24 rounded-full animate-pulse mb-4" style={{ background: "rgba(255,255,255,0.06)" }} />
                          <div className="h-12 w-28 rounded-xl animate-pulse mb-3" style={{ background: "rgba(255,255,255,0.06)" }} />
                          <div className="h-2 w-full rounded-full animate-pulse" style={{ background: "rgba(255,255,255,0.04)" }} />
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Results */}
                  {rfResult !== null && lstmResult !== null && !running && (
                    <>
                      <div className="grid grid-cols-2 gap-4">
                        {/* RF Result */}
                        <div className="result-card rounded-2xl p-6 scale-up d1"
                          style={{ background: "rgba(16,185,129,0.05)", border: "1px solid rgba(16,185,129,0.2)" }}>
                          <div className="flex items-center gap-2 mb-4">
                            <div className="w-2 h-2 rounded-full" style={{ background: "#10b981", boxShadow: "0 0 8px #10b981" }} />
                            <span className="text-xs font-semibold" style={{ color: "#10b981" }}>Random Forest</span>
                          </div>
                          <p className="text-5xl font-extrabold text-white mb-1" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                            <Counter to={rfResult} />
                          </p>
                          <p className="text-xs mb-4" style={{ color: "rgba(16,185,129,0.5)" }}>mph in +5 min</p>
                          <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(16,185,129,0.1)" }}>
                            <div className="h-1.5 rounded-full"
                              style={{
                                width: `${Math.min(rfResult / 80 * 100, 100)}%`,
                                background: "linear-gradient(90deg, #059669, #10b981)",
                                transition: "width 1.2s cubic-bezier(0.34,1.56,0.64,1)",
                              }} />
                          </div>
                        </div>

                        {/* LSTM Result */}
                        <div className="result-card rounded-2xl p-6 scale-up d2"
                          style={{ background: "rgba(6,182,212,0.05)", border: "1px solid rgba(6,182,212,0.2)" }}>
                          <div className="flex items-center gap-2 mb-4">
                            <div className="w-2 h-2 rounded-full" style={{ background: "#06b6d4", boxShadow: "0 0 8px #06b6d4" }} />
                            <span className="text-xs font-semibold" style={{ color: "#06b6d4" }}>LSTM Network</span>
                          </div>
                          <p className="text-5xl font-extrabold text-white mb-1" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                            <Counter to={lstmResult} />
                          </p>
                          <p className="text-xs mb-4" style={{ color: "rgba(6,182,212,0.5)" }}>mph in +5 min</p>
                          <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(6,182,212,0.1)" }}>
                            <div className="h-1.5 rounded-full"
                              style={{
                                width: `${Math.min(lstmResult / 80 * 100, 100)}%`,
                                background: "linear-gradient(90deg, #0e7490, #06b6d4)",
                                transition: "width 1.2s cubic-bezier(0.34,1.56,0.64,1) 0.1s",
                              }} />
                          </div>
                        </div>
                      </div>

                      {/* Comparison Panel */}
                      <div className="rounded-2xl p-6 scale-up d3"
                        style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)" }}>
                        <div className="flex items-center justify-between mb-5">
                          <h3 className="text-sm font-bold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                            Model Comparison
                          </h3>
                          {trend && (
                            <div className="flex items-center gap-1.5 text-xs"
                              style={{ color: trend === "up" ? "#10b981" : trend === "down" ? "#f87171" : "rgba(255,255,255,0.4)" }}>
                              {trend === "up" ? <ArrowUpRight size={14} /> : trend === "down" ? <ArrowDownRight size={14} /> : <Minus size={14} />}
                              <span>{trend === "up" ? "Speed increasing" : trend === "down" ? "Speed decreasing" : "Speed stable"}</span>
                            </div>
                          )}
                        </div>
                        <div className="space-y-4">
                          {[
                            { label: "Random Forest", value: rfResult,     color: "#10b981" },
                            { label: "LSTM Network",  value: lstmResult,   color: "#06b6d4" },
                            { label: "Current Speed", value: currentSpeed!, color: "#f59e0b" },
                          ].map(({ label, value, color }, i) => (
                            <div key={label} className="flex items-center gap-4">
                              <span className="text-xs w-28 flex-shrink-0" style={{ color: "rgba(255,255,255,0.4)" }}>{label}</span>
                              <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.05)" }}>
                                <div className="h-2 rounded-full"
                                  style={{
                                    width: `${Math.min(value / 80 * 100, 100)}%`,
                                    background: color,
                                    transition: `width 1.2s cubic-bezier(0.34,1.56,0.64,1) ${i * 80}ms`,
                                    boxShadow: `0 0 8px ${color}60`,
                                  }} />
                              </div>
                              <span className="text-xs font-bold w-16 text-right"
                                style={{ color, fontFamily: "'JetBrains Mono', monospace" }}>
                                {value.toFixed(1)} mph
                              </span>
                            </div>
                          ))}
                        </div>
                        <div className="mt-5 pt-4 flex items-center justify-between"
                          style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                          <span className="text-xs" style={{ color: "rgba(255,255,255,0.3)" }}>RF vs LSTM delta</span>
                          <span className="text-sm font-bold" style={{ color: "rgba(255,255,255,0.7)", fontFamily: "'JetBrains Mono', monospace" }}>
                            ±{Math.abs(rfResult - lstmResult).toFixed(2)} mph
                          </span>
                        </div>
                      </div>
                    </>
                  )}

                  {/* Empty State */}
                  {rfResult === null && !running && (
                    <div className="rounded-2xl p-16 flex flex-col items-center justify-center text-center fade-in"
                      style={{ background: "rgba(255,255,255,0.015)", border: "1px solid rgba(255,255,255,0.06)", minHeight: 300 }}>
                      <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5"
                        style={{ background: "rgba(6,182,212,0.08)", border: "1px solid rgba(6,182,212,0.12)" }}>
                        <Gauge size={26} style={{ color: "#06b6d4", opacity: 0.6 }} />
                      </div>
                      <p className="text-base font-bold text-white mb-2" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                        Ready to Predict
                      </p>
                      <p className="text-sm max-w-xs" style={{ color: "rgba(255,255,255,0.3)" }}>
                        {modelsReady
                          ? "Enter 12 speed readings and click Run Prediction."
                          : "Initialising models… ready in a moment."}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* ════════ HISTORY SECTION ════════ */}
            {section === "history" && (
              <div className="space-y-6 fade-up">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-extrabold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                      Prediction History
                    </h2>
                    <p className="text-sm mt-1" style={{ color: "rgba(255,255,255,0.3)" }}>
                      {history.length} predictions this session
                    </p>
                  </div>
                  {history.length > 0 && (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => {
                          const header = "Time,Actual (mph),RF (mph),LSTM (mph),Delta RF vs LSTM";
                          const rows = [...history].reverse().map(h =>
                            `${h.time},${h.actual.toFixed(2)},${h.rf.toFixed(2)},${h.lstm.toFixed(2)},${Math.abs(h.rf - h.lstm).toFixed(2)}`
                          );
                          const csv = [header, ...rows].join("\n");
                          const blob = new Blob([csv], { type: "text/csv" });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement("a");
                          a.href = url;
                          a.download = `traffic_predictions_${new Date().toISOString().slice(0,10)}.csv`;
                          a.click();
                          URL.revokeObjectURL(url);
                          toast.success("CSV downloaded!");
                        }}
                        className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium transition-all hover:bg-white/5"
                        style={{ color: "#06b6d4", border: "1px solid rgba(6,182,212,0.2)" }}>
                        <Download size={12} /> Download CSV
                      </button>
                      <button onClick={() => setHistory([])}
                        className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium transition-all hover:bg-white/5"
                        style={{ color: "rgba(255,255,255,0.4)", border: "1px solid rgba(255,255,255,0.08)" }}>
                        <RotateCcw size={12} /> Clear
                      </button>
                    </div>
                  )}
                </div>

                {history.length > 0 ? (
                  <>
                    {/* Area Chart */}
                    <div className="rounded-2xl p-6"
                      style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)" }}>
                      <h3 className="text-sm font-bold text-white mb-5" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                        Speed Over Time
                      </h3>
                      <ResponsiveContainer width="100%" height={260}>
                        <AreaChart data={history} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                          <defs>
                            <linearGradient id="gA" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%"  stopColor="#f59e0b" stopOpacity={0.2} />
                              <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="gR" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%"  stopColor="#10b981" stopOpacity={0.15} />
                              <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="gL" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%"  stopColor="#06b6d4" stopOpacity={0.15} />
                              <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                          <XAxis dataKey="time" tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 10 }} axisLine={false} tickLine={false} />
                          <YAxis tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 10 }} axisLine={false} tickLine={false} />
                          <Tooltip
                            contentStyle={{ background: "#111113", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, fontSize: 11 }}
                            labelStyle={{ color: "rgba(255,255,255,0.5)" }} />
                          <Legend wrapperStyle={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }} />
                          <Area type="monotone" dataKey="actual" stroke="#f59e0b" strokeWidth={2} fill="url(#gA)" name="Actual" dot={false} isAnimationActive animationDuration={800} />
                          <Area type="monotone" dataKey="rf"     stroke="#10b981" strokeWidth={2} fill="url(#gR)" name="RF"     dot={false} strokeDasharray="5 3" isAnimationActive animationDuration={900} />
                          <Area type="monotone" dataKey="lstm"   stroke="#06b6d4" strokeWidth={2} fill="url(#gL)" name="LSTM"   dot={false} strokeDasharray="5 3" isAnimationActive animationDuration={1000} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Bar Chart */}
                    <div className="rounded-2xl p-6"
                      style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)" }}>
                      <h3 className="text-sm font-bold text-white mb-5" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                        RF vs LSTM — Last 10 Predictions
                      </h3>
                      <ResponsiveContainer width="100%" height={220}>
                        <BarChart data={history.slice(-10)} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                          <XAxis dataKey="time" tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 10 }} axisLine={false} tickLine={false} />
                          <YAxis tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 10 }} axisLine={false} tickLine={false} />
                          <Tooltip
                            contentStyle={{ background: "#111113", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, fontSize: 11 }}
                            labelStyle={{ color: "rgba(255,255,255,0.5)" }} />
                          <Legend wrapperStyle={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }} />
                          <Bar dataKey="rf"   fill="#10b981" name="RF"   radius={[4, 4, 0, 0]} opacity={0.85} isAnimationActive animationDuration={700} />
                          <Bar dataKey="lstm" fill="#06b6d4" name="LSTM" radius={[4, 4, 0, 0]} opacity={0.85} isAnimationActive animationDuration={800} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Table */}
                    <div className="rounded-2xl overflow-hidden"
                      style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)" }}>
                      <div className="px-6 py-4" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                        <h3 className="text-sm font-bold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                          Prediction Log
                        </h3>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                              {["Time", "Actual (mph)", "RF (mph)", "LSTM (mph)", "Δ RF vs LSTM"].map(h => (
                                <th key={h} className="text-left px-6 py-3 text-[10px] font-semibold uppercase tracking-wider"
                                  style={{ color: "rgba(255,255,255,0.25)", fontFamily: "'JetBrains Mono', monospace" }}>
                                  {h}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {[...history].reverse().map((h, i) => (
                              <tr key={i} className="transition-colors hover:bg-white/[0.02]"
                                style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                                <td className="px-6 py-3.5 text-xs" style={{ color: "rgba(255,255,255,0.4)", fontFamily: "'JetBrains Mono', monospace" }}>{h.time}</td>
                                <td className="px-6 py-3.5 text-xs font-bold" style={{ color: "#f59e0b", fontFamily: "'JetBrains Mono', monospace" }}>{h.actual.toFixed(1)}</td>
                                <td className="px-6 py-3.5 text-xs font-bold" style={{ color: "#10b981", fontFamily: "'JetBrains Mono', monospace" }}>{h.rf.toFixed(1)}</td>
                                <td className="px-6 py-3.5 text-xs font-bold" style={{ color: "#06b6d4", fontFamily: "'JetBrains Mono', monospace" }}>{h.lstm.toFixed(1)}</td>
                                <td className="px-6 py-3.5 text-xs" style={{ color: "rgba(255,255,255,0.35)", fontFamily: "'JetBrains Mono', monospace" }}>{Math.abs(h.rf - h.lstm).toFixed(2)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="rounded-2xl p-20 flex flex-col items-center text-center"
                    style={{ background: "rgba(255,255,255,0.015)", border: "1px solid rgba(255,255,255,0.06)" }}>
                    <Clock size={32} style={{ color: "rgba(255,255,255,0.1)", marginBottom: 16 }} />
                    <p className="text-base font-bold text-white mb-2" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                      No History Yet
                    </p>
                    <p className="text-sm" style={{ color: "rgba(255,255,255,0.3)" }}>
                      Run predictions to build your history charts.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* ════════ BATCH SECTION ════════ */}
            {section === "batch" && (
              <div className="space-y-5 fade-up">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-extrabold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>Batch Predictions</h2>
                    <p className="text-sm mt-0.5" style={{ color: "rgba(255,255,255,0.3)" }}>Upload a CSV to run RF + LSTM on multiple sequences at once</p>
                  </div>
                  {batchResults.length > 0 && (
                    <button onClick={downloadBatchCSV}
                      className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all"
                      style={{ background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.3)", color: "#10b981" }}>
                      <Download size={14} /> Download Results CSV
                    </button>
                  )}
                </div>

                {/* CSV Format Guide + Sample Download */}
                <div className="rounded-xl p-4" style={{ background: "rgba(6,182,212,0.05)", border: "1px solid rgba(6,182,212,0.15)" }}>
                  <div className="flex items-start gap-3">
                    <FileSpreadsheet size={16} style={{ color: "#06b6d4", marginTop: 2, flexShrink: 0 }} />
                    <div className="flex-1">
                      <p className="text-sm font-semibold" style={{ color: "#06b6d4" }}>Expected CSV format</p>
                      <p className="text-xs mt-1" style={{ color: "rgba(255,255,255,0.4)" }}>
                        Each row = one prediction request. Provide <strong style={{ color: "rgba(255,255,255,0.7)" }}>12 consecutive speed readings</strong> (5-min intervals, 1 hour of history).
                        Optionally include a timestamp as the first column.
                      </p>
                      <div className="mt-2 rounded-lg px-3 py-2" style={{ background: "rgba(0,0,0,0.3)", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "rgba(255,255,255,0.5)" }}>
                        <div style={{ color: "rgba(255,255,255,0.25)" }}># Option A — 12 speeds only:</div>
                        <div>65.2,64.8,63.5,60.1,55.4,50.2,45.1,40.5,38.2,35.5,33.1,30.0</div>
                        <div className="mt-1" style={{ color: "rgba(255,255,255,0.25)" }}># Option B — timestamp + 12 speeds:</div>
                        <div>2012-03-01 08:00,65.2,64.8,63.5,60.1,55.4,50.2,45.1,40.5,38.2,35.5,33.1,30.0</div>
                      </div>
                      <button
                        onClick={() => {
                          const sample = "timestamp,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12\n2012-03-01 08:00,65.2,64.8,63.5,60.1,55.4,50.2,45.1,40.5,38.2,35.5,33.1,30.0\n2012-03-01 09:00,62.1,60.5,58.3,55.0,52.4,49.8,47.2,44.6,42.0,39.4,36.8,34.2\n2012-03-01 10:00,55.0,53.2,51.4,49.6,47.8,46.0,44.2,42.4,40.6,38.8,37.0,35.2\n2012-03-01 11:00,48.3,47.1,45.9,44.7,43.5,42.3,41.1,39.9,38.7,37.5,36.3,35.1\n2012-03-01 12:00,40.0,39.2,38.4,37.6,36.8,36.0,35.2,34.4,33.6,32.8,32.0,31.2";
                          const blob = new Blob([sample], { type: "text/csv" });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement("a"); a.href = url;
                          a.download = "sample_batch.csv"; a.click(); URL.revokeObjectURL(url);
                        }}
                        className="mt-2 text-xs flex items-center gap-1.5 transition-colors"
                        style={{ color: "rgba(6,182,212,0.6)" }}
                        onMouseEnter={e => (e.currentTarget.style.color = "#06b6d4")}
                        onMouseLeave={e => (e.currentTarget.style.color = "rgba(6,182,212,0.6)")}>
                        <Download size={11} /> Download sample CSV
                      </button>
                    </div>
                  </div>
                </div>

                {/* Upload drop zone */}
                <div
                  className="rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all"
                  style={{
                    background: batchFile ? "rgba(16,185,129,0.05)" : "rgba(255,255,255,0.02)",
                    border: `2px dashed ${batchFile ? "rgba(16,185,129,0.4)" : "rgba(255,255,255,0.1)"}`,
                  }}
                  onClick={() => batchFileRef.current?.click()}
                  onDragOver={e => { e.preventDefault(); }}
                  onDrop={e => {
                    e.preventDefault();
                    const file = e.dataTransfer.files[0];
                    if (!file) return;
                    if (!file.name.endsWith(".csv")) { toast.error("Please upload a .csv file"); return; }
                    setBatchFile(file); setBatchResults([]); setBatchError(null);
                    file.text().then(text => {
                      const rows = parseBatchCSV(text);
                      if (rows.length === 0) { setBatchError("CSV appears empty or unreadable."); return; }
                      setBatchRows(rows);
                      toast.success(`Loaded ${rows.length} rows from ${file.name}`);
                    });
                  }}
                >
                  <input ref={batchFileRef} type="file" accept=".csv" className="hidden"
                    onChange={e => {
                      const file = e.target.files?.[0];
                      if (!file) return;
                      setBatchFile(file); setBatchResults([]); setBatchError(null);
                      file.text().then(text => {
                        const rows = parseBatchCSV(text);
                        if (rows.length === 0) { setBatchError("CSV appears empty or unreadable."); return; }
                        setBatchRows(rows);
                        toast.success(`Loaded ${rows.length} rows from ${file.name}`);
                      });
                    }}
                  />
                  {batchFile ? (
                    <>
                      <div className="w-12 h-12 rounded-2xl flex items-center justify-center mb-3" style={{ background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.25)" }}>
                        <FileSpreadsheet size={22} style={{ color: "#10b981" }} />
                      </div>
                      <p className="text-sm font-semibold text-white">{batchFile.name}</p>
                      <p className="text-xs mt-1" style={{ color: "rgba(255,255,255,0.3)" }}>
                        {batchRows.length} rows parsed · {batchRows.filter(r => !r.error).length} valid
                        {batchRows.filter(r => r.error).length > 0 && (
                          <span style={{ color: "#f87171" }}> · {batchRows.filter(r => r.error).length} errors</span>
                        )}
                      </p>
                      <button onClick={e => { e.stopPropagation(); setBatchFile(null); setBatchRows([]); setBatchResults([]); setBatchError(null); }}
                        className="mt-3 flex items-center gap-1.5 text-xs transition-colors"
                        style={{ color: "rgba(255,255,255,0.3)" }}
                        onMouseEnter={e => (e.currentTarget.style.color = "#f87171")}
                        onMouseLeave={e => (e.currentTarget.style.color = "rgba(255,255,255,0.3)")}>
                        <XCircle size={12} /> Remove file
                      </button>
                    </>
                  ) : (
                    <>
                      <div className="w-12 h-12 rounded-2xl flex items-center justify-center mb-3" style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)" }}>
                        <Upload size={22} style={{ color: "rgba(255,255,255,0.3)" }} />
                      </div>
                      <p className="text-sm font-semibold text-white">Drop your CSV here</p>
                      <p className="text-xs mt-1" style={{ color: "rgba(255,255,255,0.3)" }}>or click to browse · .csv files only</p>
                    </>
                  )}
                </div>

                {batchError && (
                  <div className="flex items-center gap-2 p-3 rounded-xl" style={{ background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.2)" }}>
                    <AlertCircle size={14} style={{ color: "#f87171" }} />
                    <p className="text-sm" style={{ color: "#f87171" }}>{batchError}</p>
                  </div>
                )}

                {/* Preview table */}
                {batchRows.length > 0 && batchResults.length === 0 && (
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <p className="text-sm font-semibold text-white">Preview <span style={{ color: "rgba(255,255,255,0.3)" }}>— first 5 rows</span></p>
                      <button
                        onClick={runBatchPredictions}
                        disabled={batchRunning || !modelsReady}
                        className="flex items-center gap-2 px-5 py-2 rounded-xl text-sm font-bold transition-all"
                        style={{
                          background: modelsReady ? "linear-gradient(135deg, #0e7490, #06b6d4)" : "rgba(255,255,255,0.05)",
                          color: modelsReady ? "white" : "rgba(255,255,255,0.2)",
                          cursor: modelsReady ? "pointer" : "not-allowed",
                          boxShadow: modelsReady ? "0 4px 20px rgba(6,182,212,0.25)" : "none",
                          fontFamily: "'Space Grotesk', sans-serif",
                        }}>
                        {batchRunning ? <><Loader2 size={14} className="animate-spin" /> Running…</> : <><Play size={14} /> Run {batchRows.filter(r => !r.error).length} Predictions</>}
                      </button>
                    </div>
                    {batchRunning && (
                      <div className="mb-4">
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="text-xs" style={{ color: "rgba(255,255,255,0.4)" }}>Processing rows…</span>
                          <span className="text-xs" style={{ color: "#06b6d4", fontFamily: "'JetBrains Mono', monospace" }}>{batchProgress}%</span>
                        </div>
                        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.06)" }}>
                          <div className="h-1.5 rounded-full transition-all duration-300"
                            style={{ width: `${batchProgress}%`, background: "linear-gradient(90deg, #0e7490, #06b6d4)" }} />
                        </div>
                      </div>
                    )}
                    <div className="rounded-xl overflow-hidden" style={{ border: "1px solid rgba(255,255,255,0.07)" }}>
                      <table className="w-full text-xs">
                        <thead>
                          <tr style={{ background: "rgba(255,255,255,0.03)", borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
                            {["#", "Timestamp", "Speeds (12 readings)", "Status"].map(h => (
                              <th key={h} className="text-left px-4 py-3 font-semibold uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.3)", fontSize: 10 }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {batchRows.slice(0, 5).map(row => (
                            <tr key={row.rowIndex} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                              <td className="px-4 py-3" style={{ color: "rgba(255,255,255,0.3)", fontFamily: "'JetBrains Mono', monospace" }}>{row.rowIndex}</td>
                              <td className="px-4 py-3" style={{ color: "rgba(255,255,255,0.5)", fontFamily: "'JetBrains Mono', monospace" }}>{row.timestamp}</td>
                              <td className="px-4 py-3" style={{ color: "rgba(255,255,255,0.5)", fontFamily: "'JetBrains Mono', monospace" }}>
                                {row.error ? <span style={{ color: "#f87171" }}>—</span> : `${row.speeds[0].toFixed(1)} … ${row.speeds[11].toFixed(1)}`}
                              </td>
                              <td className="px-4 py-3">
                                {row.error
                                  ? <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ background: "rgba(248,113,113,0.1)", color: "#f87171" }}>Error</span>
                                  : <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ background: "rgba(16,185,129,0.1)", color: "#10b981" }}>Ready</span>
                                }
                              </td>
                            </tr>
                          ))}
                          {batchRows.length > 5 && (
                            <tr><td colSpan={4} className="px-4 py-2 text-center" style={{ color: "rgba(255,255,255,0.2)", fontSize: 11 }}>…and {batchRows.length - 5} more rows</td></tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Results table */}
                {batchResults.length > 0 && (
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <p className="text-sm font-semibold text-white">
                        Results <span style={{ color: "rgba(255,255,255,0.3)" }}>— {batchResults.filter(r => r.rfForecast !== null).length} / {batchResults.length} predicted</span>
                      </p>
                      <div className="flex gap-2">
                        <button onClick={() => { setBatchResults([]); }}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-all"
                          style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", color: "rgba(255,255,255,0.4)" }}>
                          <RotateCcw size={11} /> Re-run
                        </button>
                        <button onClick={downloadBatchCSV}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all"
                          style={{ background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.3)", color: "#10b981" }}>
                          <Download size={11} /> Download CSV
                        </button>
                      </div>
                    </div>
                    <div className="rounded-xl overflow-hidden" style={{ border: "1px solid rgba(255,255,255,0.07)" }}>
                      <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr style={{ background: "rgba(255,255,255,0.03)", borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
                              {["#", "Timestamp", "Current Speed", "RF Forecast", "LSTM Forecast", "Δ RF vs LSTM"].map(h => (
                                <th key={h} className="text-left px-4 py-3 font-semibold uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.3)", fontSize: 10 }}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {batchResults.map(row => (
                              <tr key={row.rowIndex} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}
                                onMouseEnter={e => (e.currentTarget.style.background = "rgba(255,255,255,0.02)")}
                                onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                                <td className="px-4 py-3" style={{ color: "rgba(255,255,255,0.3)", fontFamily: "'JetBrains Mono', monospace" }}>{row.rowIndex}</td>
                                <td className="px-4 py-3" style={{ color: "rgba(255,255,255,0.5)", fontFamily: "'JetBrains Mono', monospace" }}>{row.timestamp}</td>
                                <td className="px-4 py-3 font-bold" style={{ color: "#f59e0b", fontFamily: "'JetBrains Mono', monospace" }}>
                                  {row.speeds.length > 0 ? `${row.speeds[11].toFixed(1)} mph` : "—"}
                                </td>
                                {row.error ? (
                                  <td colSpan={3} className="px-4 py-3" style={{ color: "#f87171" }}>{row.error}</td>
                                ) : (
                                  <>
                                    <td className="px-4 py-3 font-bold" style={{ color: "#10b981", fontFamily: "'JetBrains Mono', monospace" }}>{row.rfForecast!.toFixed(2)} mph</td>
                                    <td className="px-4 py-3 font-bold" style={{ color: "#06b6d4", fontFamily: "'JetBrains Mono', monospace" }}>{row.lstmForecast!.toFixed(2)} mph</td>
                                    <td className="px-4 py-3" style={{ color: "rgba(255,255,255,0.4)", fontFamily: "'JetBrains Mono', monospace" }}>±{Math.abs(row.rfForecast! - row.lstmForecast!).toFixed(2)}</td>
                                  </>
                                )}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ════════ ABOUT SECTION ════════ */}
            {section === "about" && (
              <div className="space-y-6 max-w-4xl fade-up">
                <div>
                  <h2 className="text-xl font-extrabold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                    About the Models
                  </h2>
                  <p className="text-sm mt-1" style={{ color: "rgba(255,255,255,0.3)" }}>
                    Thesis project — Traffic Speed Forecasting · METR-LA Dataset
                  </p>
                </div>

                {/* Model Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  {[
                    {
                      title: "Random Forest", img: RF_ICON, color: "#10b981",
                      badge: "Baseline Model",
                      desc: "An ensemble of 100–200 decision trees trained on 72 flattened features (12 timesteps × 6 features). Serves as the interpretable baseline.",
                      details: [
                        ["Input",        "72 features (12×6 flattened)"],
                        ["Architecture", "200 trees, max depth 20"],
                        ["Split",        "70% chronological"],
                        ["Inference",    "In-browser via JSON trees"],
                      ],
                    },
                    {
                      title: "LSTM Network", img: LSTM_ICON, color: "#06b6d4",
                      badge: "Deep Learning",
                      desc: "A 2-layer stacked LSTM trained on sequential windows of shape (12, 6). Captures temporal dependencies across 60-minute windows.",
                      details: [
                        ["Input shape",  "(12, 6) sequences"],
                        ["Architecture", "LSTM(64) → LSTM(32) → Dense(1)"],
                        ["Training",     "EarlyStopping · T4 GPU · Colab Pro"],
                        ["Inference",    "TensorFlow.js in-browser"],
                      ],
                    },
                  ].map(({ title, img, color, badge, desc, details }) => (
                    <div key={title} className="rounded-2xl overflow-hidden"
                      style={{ background: "rgba(255,255,255,0.02)", border: `1px solid ${color}25` }}>
                      <div className="relative h-36 overflow-hidden">
                        <img src={img} alt={title} className="w-full h-full object-cover" />
                        <div className="absolute inset-0" style={{ background: `linear-gradient(0deg, rgba(9,9,11,0.95) 0%, rgba(9,9,11,0.3) 100%)` }} />
                        <div className="absolute bottom-3 left-4">
                          <span className="text-[10px] font-semibold px-2.5 py-1 rounded-full"
                            style={{ background: `${color}15`, color, border: `1px solid ${color}30`, fontFamily: "'JetBrains Mono', monospace" }}>
                            {badge}
                          </span>
                        </div>
                      </div>
                      <div className="p-5">
                        <h3 className="text-base font-bold text-white mb-2" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                          {title}
                        </h3>
                        <p className="text-xs leading-relaxed mb-4" style={{ color: "rgba(255,255,255,0.4)" }}>{desc}</p>
                        <div className="space-y-2.5 pt-4" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                          {details.map(([k, v]) => (
                            <div key={k} className="flex justify-between items-center">
                              <span className="text-[11px]" style={{ color: "rgba(255,255,255,0.3)" }}>{k}</span>
                              <span className="text-[11px] font-semibold" style={{ color: "rgba(255,255,255,0.7)", fontFamily: "'JetBrains Mono', monospace" }}>{v}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Feature Engineering */}
                <div className="rounded-2xl p-6"
                  style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)" }}>
                  <div className="flex items-center gap-2 mb-5">
                    <FlaskConical size={16} style={{ color: "#a78bfa" }} />
                    <h3 className="text-sm font-bold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                      Feature Engineering — 6 Features per Timestep
                    </h3>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {[
                      { name: "speed",      desc: "Raw traffic speed (mph)",    color: "#f59e0b" },
                      { name: "hour_sin",   desc: "Cyclical hour — sine",       color: "#06b6d4" },
                      { name: "hour_cos",   desc: "Cyclical hour — cosine",     color: "#06b6d4" },
                      { name: "dow_sin",    desc: "Day-of-week — sine",         color: "#a78bfa" },
                      { name: "dow_cos",    desc: "Day-of-week — cosine",       color: "#a78bfa" },
                      { name: "is_weekend", desc: "Binary weekend flag",        color: "#10b981" },
                    ].map(({ name, desc, color }) => (
                      <div key={name} className="p-3.5 rounded-xl"
                        style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
                        <code className="text-xs font-bold block mb-1" style={{ color, fontFamily: "'JetBrains Mono', monospace" }}>
                          {name}
                        </code>
                        <p className="text-[10px]" style={{ color: "rgba(255,255,255,0.3)" }}>{desc}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* ── Model Performance Metrics ── */}
                <div className="rounded-2xl overflow-hidden"
                  style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)" }}>
                  {/* Header */}
                  <div className="px-6 py-5" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <BarChart2 size={16} style={{ color: "#a78bfa" }} />
                        <h3 className="text-sm font-bold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                          Model Performance Metrics
                        </h3>
                      </div>
                      <span className="text-[10px] font-semibold px-2.5 py-1 rounded-full"
                        style={{ background: "rgba(167,139,250,0.1)", color: "#a78bfa", border: "1px solid rgba(167,139,250,0.2)", fontFamily: "'JetBrains Mono', monospace" }}>
                        Test Set · METR-LA
                      </span>
                    </div>
                    <p className="text-xs mt-1" style={{ color: "rgba(255,255,255,0.3)" }}>
                      Evaluated on the held-out 30% chronological test split (sensor-averaged)
                    </p>
                  </div>

                  {/* Table */}
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                          {[
                            { label: "Model",   tip: null },
                            { label: "MAE",     tip: "Mean Absolute Error (mph) — lower is better" },
                            { label: "RMSE",    tip: "Root Mean Squared Error (mph) — lower is better" },
                            { label: "R²",      tip: "Coefficient of determination — higher is better" },
                            { label: "Winner",  tip: null },
                          ].map(({ label, tip }) => (
                            <th key={label} className="text-left px-6 py-3.5 text-[10px] font-semibold uppercase tracking-wider"
                              style={{ color: "rgba(255,255,255,0.25)", fontFamily: "'JetBrains Mono', monospace" }}
                              title={tip ?? undefined}>
                              {label}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {/* Random Forest row */}
                        <tr className="transition-colors hover:bg-white/[0.02]"
                          style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-2.5">
                              <div className="w-2 h-2 rounded-full flex-shrink-0"
                                style={{ background: "#10b981", boxShadow: "0 0 6px #10b981" }} />
                              <span className="text-sm font-semibold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                                Random Forest
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-bold" style={{ color: "#10b981", fontFamily: "'JetBrains Mono', monospace" }}>3.24</span>
                              <span className="text-[10px]" style={{ color: "rgba(255,255,255,0.25)" }}>mph</span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-bold" style={{ color: "#10b981", fontFamily: "'JetBrains Mono', monospace" }}>5.18</span>
                              <span className="text-[10px]" style={{ color: "rgba(255,255,255,0.25)" }}>mph</span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <span className="text-sm font-bold" style={{ color: "#10b981", fontFamily: "'JetBrains Mono', monospace" }}>0.872</span>
                          </td>
                          <td className="px-6 py-4">
                            <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full"
                              style={{ background: "rgba(255,255,255,0.05)", color: "rgba(255,255,255,0.3)" }}>
                              MAE / RMSE
                            </span>
                          </td>
                        </tr>

                        {/* LSTM row */}
                        <tr className="transition-colors hover:bg-white/[0.02]">
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-2.5">
                              <div className="w-2 h-2 rounded-full flex-shrink-0"
                                style={{ background: "#06b6d4", boxShadow: "0 0 6px #06b6d4" }} />
                              <span className="text-sm font-semibold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                                LSTM Network
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-bold" style={{ color: "#06b6d4", fontFamily: "'JetBrains Mono', monospace" }}>2.91</span>
                              <span className="text-[10px]" style={{ color: "rgba(255,255,255,0.25)" }}>mph</span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-bold" style={{ color: "#06b6d4", fontFamily: "'JetBrains Mono', monospace" }}>4.73</span>
                              <span className="text-[10px]" style={{ color: "rgba(255,255,255,0.25)" }}>mph</span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <span className="text-sm font-bold" style={{ color: "#06b6d4", fontFamily: "'JetBrains Mono', monospace" }}>0.901</span>
                          </td>
                          <td className="px-6 py-4">
                            <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full"
                              style={{ background: "rgba(6,182,212,0.1)", color: "#06b6d4", border: "1px solid rgba(6,182,212,0.2)" }}>
                              R²
                            </span>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  {/* Delta summary bar */}
                  <div className="px-6 py-4" style={{ borderTop: "1px solid rgba(255,255,255,0.06)", background: "rgba(255,255,255,0.01)" }}>
                    <div className="grid grid-cols-3 gap-6">
                      {[
                        { metric: "MAE improvement",  delta: "−0.33 mph",  pct: "−10.2%",  color: "#06b6d4" },
                        { metric: "RMSE improvement", delta: "−0.45 mph",  pct: "−8.7%",   color: "#06b6d4" },
                        { metric: "R² improvement",   delta: "+0.029",     pct: "+3.3%",   color: "#06b6d4" },
                      ].map(({ metric, delta, pct, color }) => (
                        <div key={metric}>
                          <p className="text-[10px] mb-1" style={{ color: "rgba(255,255,255,0.3)" }}>{metric}</p>
                          <div className="flex items-baseline gap-1.5">
                            <span className="text-sm font-bold" style={{ color, fontFamily: "'JetBrains Mono', monospace" }}>{delta}</span>
                            <span className="text-[10px] font-semibold" style={{ color: "rgba(6,182,212,0.6)" }}>{pct}</span>
                          </div>
                          <p className="text-[10px] mt-0.5" style={{ color: "rgba(255,255,255,0.2)" }}>LSTM vs RF</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Dataset Info */}
                <div className="rounded-2xl p-6"
                  style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)" }}>
                  <div className="flex items-center gap-2 mb-4">
                    <Database size={16} style={{ color: "#06b6d4" }} />
                    <h3 className="text-sm font-bold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                      Dataset — METR-LA
                    </h3>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    {[
                      { label: "Sensors",    val: "207" },
                      { label: "Timesteps",  val: "~34,000" },
                      { label: "Interval",   val: "5 min" },
                      { label: "Duration",   val: "4 months" },
                    ].map(({ label, val }) => (
                      <div key={label} className="p-3 rounded-xl text-center"
                        style={{ background: "rgba(6,182,212,0.05)", border: "1px solid rgba(6,182,212,0.1)" }}>
                        <p className="text-lg font-bold text-white" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>{val}</p>
                        <p className="text-[10px] mt-0.5" style={{ color: "rgba(255,255,255,0.3)" }}>{label}</p>
                      </div>
                    ))}
                  </div>
                  <p className="text-xs leading-relaxed" style={{ color: "rgba(255,255,255,0.4)" }}>
                    Traffic speed readings from 207 loop detectors on Los Angeles County highways, recorded
                    March–June 2012. Source:{" "}
                    <a href="https://zenodo.org/records/5724362" target="_blank" rel="noreferrer"
                      className="underline" style={{ color: "#06b6d4" }}>
                      zenodo.org/records/5724362
                    </a>
                  </p>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </>
  );
}
