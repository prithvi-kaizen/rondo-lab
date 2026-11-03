import { useEffect, useRef, useState } from "react";
import { Pitch, atTime, defaultLayers, layerNames, type Layers } from "./Pitch";
import DecisionInspector from "./DecisionInspector";
import type { Replay } from "./types";
import "./style.css";
import "./demo.css";

type Recording = {
  id: string;
  label: string;
  seed: number;
  file: string;
  checkpoint: string | null;
  sha256: string | null;
  steps: number;
  generation: number | null;
  duration: number;
  passes: number;
  reason: string;
};
const base = import.meta.env.BASE_URL + "demo/";
const repository = "https://github.com/prithvi-kaizen/rondo-lab";
const cache = new Map<string, Replay>();
async function recording(file: string) {
  if (cache.has(file)) return cache.get(file)!;
  const response = await fetch(base + file);
  if (!response.ok)
    throw new Error(`Could not load recording (${response.status}).`);
  const replay = (await response.json()) as Replay;
  cache.set(file, replay);
  return replay;
}
export default function Demo() {
  const [entries, setEntries] = useState<Recording[]>([]);
  const [id, setId] = useState("scripted-42"),
    [compareId, setCompareId] = useState("");
  const [replay, setReplay] = useState<Replay | null>(null),
    [comparison, setComparison] = useState<Replay | null>(null);
  const [time, setTime] = useState(0),
    [playing, setPlaying] = useState(true),
    [speed, setSpeed] = useState(1),
    [loop, setLoop] = useState(true);
  const [layers, setLayers] = useState<Layers>(defaultLayers),
    [selected, setSelected] = useState(0);
  const [error, setError] = useState(""),
    [loading, setLoading] = useState(true),
    [retry, setRetry] = useState(0);
  const clock = useRef(0),
    currentTime = useRef(0);
  const entry = entries.find((e) => e.id === id),
    compareEntry = entries.find((e) => e.id === compareId);
  useEffect(() => {
    let active = true;
    fetch(base + "manifest.json")
      .then(async (r) => {
        if (!r.ok) throw new Error("Could not load simulation catalog.");
        const data = await r.json();
        if (active) setEntries(data.recordings);
      })
      .catch((e) => {
        if (active) {
          setError(String(e));
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, [retry]);
  useEffect(() => {
    if (!entry) return;
    let active = true;
    setLoading(true);
    setError("");
    Promise.all([
      recording(entry.file),
      compareEntry ? recording(compareEntry.file) : Promise.resolve(null),
    ])
      .then(([a, b]) => {
        if (active) {
          setReplay(a);
          setComparison(b);
          currentTime.current = 0;
          setTime(0);
          clock.current = 0;
          setPlaying(true);
        }
      })
      .catch((e) => {
        if (active) setError(String(e));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [entry, compareEntry, retry]);
  const end = Math.max(
    replay?.frames.at(-1)?.time || 0,
    comparison?.frames.at(-1)?.time || 0,
  );
  useEffect(() => {
    if (!playing || loading || !replay) return;
    let raf = 0;
    clock.current = 0;
    const tick = (now: number) => {
      if (clock.current && !document.hidden) {
        currentTime.current +=
          Math.min((now - clock.current) / 1000, 0.1) * speed;
        if (currentTime.current >= end) {
          currentTime.current = loop ? 0 : end;
          if (!loop) setPlaying(false);
        }
        setTime(currentTime.current);
      }
      clock.current = now;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, loading, replay, speed, end, loop]);
  const seek = (value: number) => {
    currentTime.current = Math.min(end, Math.max(0, value));
    setTime(currentTime.current);
    clock.current = 0;
  };
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, select, button, a")) return;
      if (e.code === "Space") {
        e.preventDefault();
        setPlaying((p) => !p);
      }
      if (e.code === "ArrowRight") {
        e.preventDefault();
        setPlaying(false);
        seek(currentTime.current + 0.05);
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [end]);
  const frame = replay ? atTime(replay, time) : null;
  const choose = (value: string) => {
    setId(value);
    const next = entries.find((e) => e.id === value);
    if (next?.seed !== compareEntry?.seed || value === compareId)
      setCompareId("");
  };
  return (
    <div className="public-demo">
      <a className="skip-link" href="#demo-pitch">
        Skip to simulator
      </a>
      <header className="demo-header">
        <a href={repository} className="demo-brand">
          r. <strong>rondolab</strong>
        </a>
        <span>Simulation gallery</span>
        <a href={repository}>Code & local training</a>
      </header>
      <main className="demo-main">
        <div className="demo-intro">
          <div>
            <p>Six keep the ball. Two try to win it.</p>
            <h1>Read the rondo.</h1>
          </div>
          <p>
            Watch real simulator recordings. Pause a pass, inspect a player, or
            compare saved policies from the same opening scenario.
          </p>
        </div>
        <div className="demo-selectors">
          <label>
            Simulation
            <select value={id} onChange={(e) => choose(e.target.value)}>
              {entries.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.label} · seed {e.seed}
                </option>
              ))}
            </select>
          </label>
          <label>
            Compare same scenario
            <select
              value={compareId}
              onChange={(e) => setCompareId(e.target.value)}
            >
              <option value="">Single pitch</option>
              {entries
                .filter((e) => e.seed === entry?.seed && e.id !== id)
                .map((e) => (
                  <option key={e.id} value={e.id}>
                    {e.label}
                  </option>
                ))}
            </select>
          </label>
        </div>
        {loading && <p role="status">Loading recorded simulation…</p>}
        {error && (
          <div role="alert" className="error">
            {error}{" "}
            <button onClick={() => setRetry((r) => r + 1)}>
              Retry loading
            </button>
          </div>
        )}
        {replay && frame && (
          <>
            <section
              id="demo-pitch"
              aria-label="Simulation playback"
              className={`demo-pitches ${comparison ? "paired" : ""}`}
            >
              <div>
                <div className="demo-pitch-title">
                  <strong>{entry?.label}</strong>
                  <span>
                    Seed {replay.seed} · {frame.passes} passes
                  </span>
                </div>
                <Pitch
                  replay={replay}
                  time={time}
                  layers={layers}
                  selected={selected}
                  onSelect={setSelected}
                />
              </div>
              {comparison && (
                <div>
                  <div className="demo-pitch-title">
                    <strong>{compareEntry?.label}</strong>
                    <span>
                      Seed {comparison.seed} · {atTime(comparison, time).passes}{" "}
                      passes
                    </span>
                  </div>
                  <Pitch
                    replay={comparison}
                    time={time}
                    layers={layers}
                    selected={selected}
                    onSelect={setSelected}
                  />
                </div>
              )}
            </section>
            <section className="demo-controls" aria-label="Timeline controls">
              <div className="demo-time">
                <strong>{time.toFixed(2)}s</strong>
                <span>
                  {time >= replay.frames.at(-1)!.time
                    ? String(replay.metrics.reason).replaceAll("_", " ")
                    : "In play"}
                </span>
                <span>{end.toFixed(2)}s</span>
              </div>
              <input
                aria-label="Episode timeline"
                type="range"
                min={0}
                max={end}
                step={0.01}
                value={time}
                disabled={loading}
                onChange={(e) => {
                  setPlaying(false);
                  seek(Number(e.target.value));
                }}
              />
              <div className="demo-buttons">
                <button onClick={() => seek(0)} disabled={loading}>
                  Restart
                </button>
                <button
                  className="primary"
                  onClick={() => {
                    if (time >= end) seek(0);
                    setPlaying((p) => !p);
                  }}
                  disabled={loading}
                >
                  {playing ? "Pause" : "Play"}
                </button>
                <button
                  onClick={() => {
                    setPlaying(false);
                    seek(time + 0.05);
                  }}
                  disabled={loading}
                >
                  Step 0.05s
                </button>
                <label>
                  Speed
                  <select
                    value={speed}
                    onChange={(e) => setSpeed(Number(e.target.value))}
                  >
                    {[0.25, 0.5, 1, 2, 4].map((s) => (
                      <option value={s} key={s}>
                        {s}×
                      </option>
                    ))}
                  </select>
                </label>
                <label className="demo-check">
                  <input
                    type="checkbox"
                    checked={loop}
                    onChange={(e) => setLoop(e.target.checked)}
                  />
                  Loop
                </label>
              </div>
              <div
                className="demo-events"
                aria-label="Recorded football events"
              >
                {replay.events
                  .filter((e) =>
                    [
                      "pass",
                      "interception",
                      "tackle",
                      "failed_reception",
                    ].includes(e.type),
                  )
                  .map((e, i) => (
                    <button
                      key={i}
                      onClick={() => {
                        setPlaying(false);
                        seek(e.time);
                      }}
                    >
                      {e.type.replaceAll("_", " ")} · {e.time.toFixed(1)}s
                    </button>
                  ))}
              </div>
            </section>
            <div className="demo-detail">
              <section>
                <h2>Tactical layers</h2>
                <div className="demo-layers">
                  {Object.entries(layerNames).map(([k, label]) => (
                    <label key={k}>
                      <input
                        type="checkbox"
                        checked={layers[k as keyof Layers]}
                        onChange={(e) =>
                          setLayers((l) => ({ ...l, [k]: e.target.checked }))
                        }
                      />
                      {label}
                    </label>
                  ))}
                </div>
                <button onClick={() => setLayers(defaultLayers)}>
                  Reset overlays
                </button>
              </section>
              <section>
                <h2>
                  Player {selected + 1} · {frame.players[selected].control}
                </h2>
                <p>
                  Click a shirt on the first pitch to inspect its recorded
                  decision.
                </p>
                <DecisionInspector
                  replay={replay}
                  frame={frame}
                  player={frame.players[selected]}
                />
              </section>
            </div>
            <section className="demo-provenance">
              <h2>What you’re watching</h2>
              <p>
                This public demo plays stored 20 Hz simulator states with smooth
                interpolation. Training and new simulations run in the
                downloadable local application. No Python server, account or
                model download is needed to view these recordings.
              </p>
              <p>
                Generation 0 is untrained. Generations 5 and 10 are actual
                checkpoints from a short, 5,120-decision alternating PPO run.
                Later possession policies often stall at the holding limit.
                These recordings demonstrate the training pipeline; they do not
                establish useful improvement by both teams.
              </p>
              <p>
                {entry?.checkpoint
                  ? `Source: ${entry.checkpoint} · ${entry.steps.toLocaleString()} decision steps · scripted cover/press opponent.`
                  : "Source: scripted football agents, environment v2, 12 × 12m grid."}{" "}
                {comparison &&
                  "Both pitches share seed and physical configuration. A finished episode holds its final state while the other continues."}
              </p>
              {entry?.sha256 && (
                <details>
                  <summary>Checkpoint fingerprint</summary>
                  <code>{entry.sha256}</code>
                </details>
              )}
              <a href={base + (entry?.file || "")} download>
                Download this recorded replay
              </a>
            </section>
          </>
        )}
        <footer className="demo-footer">
          <span>Rondo Lab · recorded football, inspectable decisions.</span>
          <a href={repository}>Run the full lab locally</a>
        </footer>
      </main>
    </div>
  );
}
