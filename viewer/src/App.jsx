import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fromBandText } from './io/viewermodel.js';
import BandStructurePlot from './components/BandStructurePlot.jsx';
import CrystalViewer from './components/CrystalViewer.jsx';
import ModeInspector from './components/ModeInspector.jsx';
import InsPanel from './components/InsPanel.jsx';

/**
 * Mode viewer for the mlip-quantum-thermal pipeline.
 *
 * Input is any phonopy-standard band yaml this repo emits — band.yaml
 * (harmonic), band_T.yaml (quantum-sampled effective FCs), band_rmc.yaml
 * (RMC-snapshot effective FCs), or modes_irrep.yaml (the published distortion
 * patterns as Bloch eigenvectors). Frequencies are converted THz -> meV on
 * load (io/viewermodel.js); eigenvectors keep phonopy's mass-weighted gauge.
 */

const TABS = [
  { id: 'mode', label: 'Mode details' },
  { id: 'ins', label: 'Simulated INS' },
];

function Field({ label, children }) {
  return (
    <label className="flex items-center gap-2 text-[12px]" style={{ color: 'var(--dim)' }}>
      <span className="whitespace-nowrap">{label}</span>
      {children}
    </label>
  );
}

export default function App() {
  const [model, setModel] = useState(null);
  const [name, setName] = useState('');
  const [error, setError] = useState(null);
  const [sel, setSel] = useState({ k: 0, m: 0 });
  const [tab, setTab] = useState('mode');

  // viewer controls
  const [playing, setPlaying] = useState(true);
  const [amplitude, setAmplitude] = useState(2.0);
  const [speed, setSpeed] = useState(0.08);
  const [nx, setNx] = useState(2);
  const [ny, setNy] = useState(2);
  const [nz, setNz] = useState(1);
  const [showVectors, setShowVectors] = useState(true);
  const [showBonds, setShowBonds] = useState(true);
  const [showCell, setShowCell] = useState(true);
  const [cameraAxis, setCameraAxis] = useState(null);
  const [temperature, setTemperature] = useState(5);

  const fileRef = useRef(null);

  const load = useCallback(async (file) => {
    setError(null);
    try {
      const text = await file.text();
      const m = fromBandText(text);
      setModel(m);
      setName(file.name);
      setSel({ k: 0, m: 0 });
      if (!m.hasEig) {
        setError(`${file.name} carries no eigenvectors — bands will plot, but the ` +
          '3D animation needs them. Re-run without --no-eigenvectors.');
      }
    } catch (e) {
      setModel(null);
      setError(String(e.message || e));
    }
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0];
    if (f) load(f);
  }, [load]);

  // Deep link: ?load=<url> fetches a band yaml on startup, so a run directory
  // served over http (or the bundled example) opens without the file picker.
  useEffect(() => {
    const url = new URLSearchParams(window.location.search).get('load');
    if (!url) return;
    (async () => {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const text = await res.text();
        const m = fromBandText(text);
        setModel(m);
        setName(url.split('/').pop());
        setSel({ k: 0, m: 0 });
      } catch (e) {
        setError(`could not load ${url}: ${e.message || e}`);
      }
    })();
  }, []);

  const eigenvector = model?.eigvecs?.[sel.k]?.[sel.m] ?? null;
  const qPoint = model?.qPoints?.[sel.k] ?? null;
  const freq = model?.bands?.[sel.k]?.[sel.m];

  const qLabel = useMemo(() => {
    if (!qPoint) return '';
    return `(${qPoint.map((v) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(3)).join(', ')})`;
  }, [qPoint]);

  const hs = model?.kpathMeta?.hsymIndex?.[sel.k];

  return (
    <div
      className="min-h-screen"
      style={{ background: 'var(--bg)', color: 'var(--ink)' }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={onDrop}
    >
      {/* ── header ─────────────────────────────────────────────────────── */}
      <header
        className="flex flex-wrap items-center gap-3 px-5 py-3 border-b"
        style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
      >
        <div className="flex flex-col">
          <span className="text-[15px] font-semibold tracking-tight">Mode Viewer</span>
          <span className="text-[11px]" style={{ color: 'var(--faint)' }}>
            mlip-quantum-thermal
          </span>
        </div>
        <div className="flex-1" />
        {name && (
          <span
            className="text-[12px] px-2 py-1 rounded-md font-mono"
            style={{ background: 'var(--soft)', color: 'var(--accentInk)' }}
          >
            {name}
          </span>
        )}
        <input
          ref={fileRef}
          type="file"
          accept=".yaml,.yml,.json"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && load(e.target.files[0])}
        />
        <button
          onClick={() => fileRef.current?.click()}
          className="text-[13px] px-3 py-1.5 rounded-md font-medium text-white"
          style={{ background: 'var(--accent)' }}
        >
          Load band yaml
        </button>
      </header>

      {error && (
        <div
          className="px-5 py-2 text-[12px]"
          style={{ background: '#fff4f0', color: 'var(--warnInk)' }}
        >
          {error}
        </div>
      )}

      {/* ── empty state ────────────────────────────────────────────────── */}
      {!model && (
        <div className="max-w-2xl mx-auto mt-24 px-6 text-center">
          <div
            className="rounded-xl border-2 border-dashed p-12"
            style={{ borderColor: 'var(--border)', background: 'var(--card)' }}
          >
            <p className="text-[15px] font-medium mb-2">Drop a band yaml here</p>
            <p className="text-[13px] leading-relaxed" style={{ color: 'var(--dim)' }}>
              Any phonopy-standard band file the pipeline emits —{' '}
              <code className="font-mono">band.yaml</code>,{' '}
              <code className="font-mono">band_T.yaml</code>,{' '}
              <code className="font-mono">band_rmc.yaml</code>, or{' '}
              <code className="font-mono">modes_irrep.yaml</code>. Everything runs
              locally in the browser; nothing is uploaded.
            </p>
          </div>
        </div>
      )}

      {/* ── main ───────────────────────────────────────────────────────── */}
      {model && (
        <main className="p-4 grid gap-4 grid-cols-1 xl:grid-cols-2">
          {/* bands */}
          <section
            className="rounded-xl border p-3"
            style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
          >
            <h2 className="text-[13px] font-semibold mb-2">Band structure</h2>
            <BandStructurePlot
              bands={model.bands}
              qPoints={model.qPoints}
              baseStructure={model.baseStructure}
              kpathMeta={model.kpathMeta}
              selected={sel}
              onPick={(k, m) => setSel({ k, m })}
              unit="meV"
            />
            <p className="text-[11px] mt-1" style={{ color: 'var(--faint)' }}>
              Click a point to select a mode; drag to zoom.
            </p>
          </section>

          {/* 3D */}
          <section
            className="rounded-xl border p-3 flex flex-col"
            style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
          >
            <div className="flex items-baseline gap-3 mb-2">
              <h2 className="text-[13px] font-semibold">3D mode</h2>
              <span className="text-[12px] font-mono" style={{ color: 'var(--accentInk)' }}>
                {hs ? `${hs} ` : ''}q = {qLabel}
                {Number.isFinite(freq) ? ` · ${freq.toFixed(3)} meV · branch ${sel.m + 1}` : ''}
              </span>
            </div>

            <div
              className="rounded-xl flex-1"
              style={{ background: 'var(--background)', minHeight: 380 }}
            >
              <CrystalViewer
                baseStructure={model.baseStructure}
                eigenvector={eigenvector}
                qPoint={qPoint}
                isPlaying={playing}
                amplitude={amplitude}
                speed={speed}
                supercell={[nx, ny, nz]}
                showVectors={showVectors}
                showBonds={showBonds}
                showCell={showCell}
                cameraAxis={cameraAxis}
              />
            </div>

            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mt-3">
              <button
                onClick={() => setPlaying((p) => !p)}
                className="text-[12px] px-3 py-1 rounded-md font-medium"
                style={{ background: 'var(--inset2)', color: 'var(--ink)' }}
              >
                {playing ? 'Pause' : 'Play'}
              </button>
              <Field label="Amplitude">
                <input type="range" min="0.2" max="8" step="0.1" value={amplitude}
                  onChange={(e) => setAmplitude(+e.target.value)} />
              </Field>
              <Field label="Speed">
                <input type="range" min="0.01" max="0.3" step="0.01" value={speed}
                  onChange={(e) => setSpeed(+e.target.value)} />
              </Field>
              <Field label="Supercell">
                {[[nx, setNx], [ny, setNy], [nz, setNz]].map(([v, set], i) => (
                  <input
                    key={i} type="number" min="1" max="6" value={v}
                    onChange={(e) => set(Math.max(1, Math.min(6, +e.target.value || 1)))}
                    className="w-11 px-1 py-0.5 rounded border text-center"
                    style={{ borderColor: 'var(--border)', background: 'var(--inset)' }}
                  />
                ))}
              </Field>
              <Field label="Vectors">
                <input type="checkbox" checked={showVectors}
                  onChange={(e) => setShowVectors(e.target.checked)} />
              </Field>
              <Field label="Bonds">
                <input type="checkbox" checked={showBonds}
                  onChange={(e) => setShowBonds(e.target.checked)} />
              </Field>
              <Field label="Cell">
                <input type="checkbox" checked={showCell}
                  onChange={(e) => setShowCell(e.target.checked)} />
              </Field>
              <div className="flex gap-1">
                {['x', 'y', 'z'].map((ax) => (
                  <button
                    key={ax}
                    onClick={() => setCameraAxis(`${ax}:${Date.now()}`)}
                    className="text-[11px] w-6 h-6 rounded font-mono"
                    style={{ background: 'var(--inset2)' }}
                  >
                    {ax}
                  </button>
                ))}
                <button
                  onClick={() => setCameraAxis(`reset:${Date.now()}`)}
                  className="text-[11px] px-2 h-6 rounded"
                  style={{ background: 'var(--inset2)' }}
                >
                  reset
                </button>
              </div>
            </div>
          </section>

          {/* tabs */}
          <section
            className="rounded-xl border p-3 xl:col-span-2"
            style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
          >
            <div className="flex items-center gap-1 mb-3">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className="text-[12px] px-3 py-1.5 rounded-md font-medium"
                  style={
                    tab === t.id
                      ? { background: 'var(--soft)', color: 'var(--accentInk)' }
                      : { color: 'var(--dim)' }
                  }
                >
                  {t.label}
                </button>
              ))}
              <div className="flex-1" />
              {tab === 'ins' && (
                <Field label="T (K)">
                  <input
                    type="number" min="0" max="1200" step="5" value={temperature}
                    onChange={(e) => setTemperature(Math.max(0, +e.target.value || 0))}
                    className="w-16 px-1 py-0.5 rounded border text-center"
                    style={{ borderColor: 'var(--border)', background: 'var(--inset)' }}
                  />
                </Field>
              )}
            </div>

            {tab === 'mode' && (
              <ModeInspector
                results={model}
                selectedK={sel.k}
                selectedMode={sel.m}
                unit="meV"
              />
            )}
            {tab === 'ins' && <InsPanel results={model} temperature={temperature} />}
          </section>
        </main>
      )}
    </div>
  );
}
