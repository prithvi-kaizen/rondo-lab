import React, { useCallback, useEffect, useRef, useState } from "react";
import { Pitch, atTime, defaultLayers, layerNames, type Layers } from "./Pitch";
import {
  api,
  labels,
  defenderLabels,
  type Config,
  type Policy,
  type Replay,
  type Experiment,
  type TrainingStatus,
} from "./types";
import "./style.css";

type Page =
  | "Playground"
  | "Watch AI Learn"
  | "Training"
  | "Experiments"
  | "RL Explorer"
  | "Settings";
const pages: Page[] = [
  "Playground",
  "Watch AI Learn",
  "Training",
  "Experiments",
  "RL Explorer",
  "Settings",
];
const fmt = (t: number) =>
  `${Math.floor(t / 60)
    .toString()
    .padStart(2, "0")}:${(t % 60).toFixed(1).padStart(4, "0")}`;
const pretty = (s: string) => s.replaceAll("_", " ");
export default function App() {
  const [pitchFocus, setPitchFocus] = useState(false);
  useEffect(() => {
    const exit = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPitchFocus(false);
    };
    window.addEventListener("keydown", exit);
    return () => window.removeEventListener("keydown", exit);
  }, []);
  const [page, setPage] = useState<Page>("Playground"),
    [replay, setReplay] = useState<Replay | null>(null),
    [comparison, setComparison] = useState<Replay | null>(null);
  const [policies, setPolicies] = useState<Policy[]>([]),
    [experiments, setExperiments] = useState<Experiment[]>([]),
    [config, setConfig] = useState<Config>({}),
    [presets, setPresets] = useState<Record<string, Config>>({}),
    [defaults, setDefaults] = useState<Config>({});
  const [checkpoint, setCheckpoint] = useState(""),
    [compareId, setCompareId] = useState(""),
    [seed, setSeed] = useState(42),
    [episode, setEpisode] = useState(0),
    [preset, setPreset] = useState("Classic 6v2");
  const [time, setTime] = useState(0),
    [playing, setPlaying] = useState(true),
    [speed, setSpeed] = useState(1),
    [loop, setLoop] = useState(false),
    [selected, setSelected] = useState<number | null>(null),
    [layers, setLayers] = useState<Layers>(defaultLayers);
  const [error, setError] = useState(""),
    [loading, setLoading] = useState(true),
    [sideTab, setSideTab] = useState<"Tactics" | "Player" | "Session">(
      "Tactics",
    ),
    [training, setTraining] = useState<TrainingStatus>({ status: "idle" });
  const [trainMode, setTrainMode] = useState("possession"),
    [scenario, setScenario] = useState("basic"),
    [steps, setSteps] = useState(4096),
    [trainSeed, setTrainSeed] = useState(42),
    [interval, setIntervalSteps] = useState(2048),
    [busy, setBusy] = useState(false),
    [reward, setReward] = useState("moderate");
  const [experimentA, setExperimentA] = useState(""),
    [experimentB, setExperimentB] = useState(""),
    [chartMetric, setChartMetric] = useState("mean_passes");
  const input = useRef<HTMLInputElement>(null),
    clock = useRef({ last: 0 }),
    timeRef = useRef(0);
  const loadSequence = useRef(0);
  const refresh = useCallback(async () => {
    const [p, e] = await Promise.all([
      api<Policy[]>("/policies"),
      api<Experiment[]>("/experiments"),
    ]);
    setPolicies(p);
    setExperiments(e);
  }, []);
  const load = useCallback(
    async (c: Config, s: number, id = "", compare = "") => {
      const request = ++loadSequence.current;
      setLoading(true);
      setError("");
      setPlaying(false);
      try {
        const [r, b] = await Promise.all([
          api<Replay>("/episodes", {
            config: c,
            seed: s,
            checkpoint: id || null,
          }),
          compare
            ? api<Replay>("/episodes", {
                config: c,
                seed: s,
                checkpoint: compare,
              })
            : Promise.resolve(null),
        ]);
        if (request !== loadSequence.current) return;
        setReplay(r);
        setEpisode(s >= 42 && s <= 46 ? s - 42 : -1);
        setComparison(b);
        setTime(0);
        timeRef.current = 0;
        setPlaying(true);
        clock.current.last = 0;
      } catch (e) {
        if (request === loadSequence.current) setError(String(e));
      } finally {
        if (request === loadSequence.current) setLoading(false);
      }
    },
    [],
  );
  useEffect(() => {
    let active = true;
    api<{ defaults: Config; presets: Record<string, Config> }>("/config")
      .then(async (data) => {
        if (!active) return;
        setConfig(data.defaults);
        setDefaults(data.defaults);
        setPresets(data.presets);
        await load(data.defaults, 42);
        await refresh();
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [load, refresh]);
  useEffect(() => {
    if (!replay || !playing || loading) return;
    let raf = 0;
    clock.current.last = 0;
    const tick = (now: number) => {
      if (clock.current.last) {
        const delta = Math.min((now - clock.current.last) / 1000, 0.1) * speed;
        const end = Math.max(
          replay.frames.at(-1)!.time,
          comparison?.frames.at(-1)!.time || 0,
        );
        timeRef.current = Math.min(end, timeRef.current + delta);
        if (timeRef.current >= end) {
          if (loop) timeRef.current = 0;
          else setPlaying(false);
        }
        setTime(timeRef.current);
      }
      clock.current.last = now;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, replay, comparison, loop, loading]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLElement &&
        ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(e.target.tagName)
      )
        return;
      if (e.code === "Space") {
        e.preventDefault();
        setPlaying((p) => !p);
      }
      if (e.code === "ArrowRight") {
        e.preventDefault();
        scrub(timeRef.current + 0.05);
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [replay]);
  useEffect(() => {
    if (page !== "Training") return;
    let mounted = true;
    const poll = () =>
      api<TrainingStatus>("/training")
        .then((s) => {
          if (mounted) {
            setTraining(s);
            if (s.status === "completed") void refresh();
          }
        })
        .catch((e) => {
          if (mounted) setError(String(e));
        });
    void poll();
    const timer = window.setInterval(poll, 2000);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [page, refresh]);
  const end = Math.max(
    replay?.frames.at(-1)!.time || 0,
    comparison?.frames.at(-1)!.time || 0,
  );
  const scrub = (t: number) => {
    setPlaying(false);
    timeRef.current = Math.max(0, Math.min(t, end));
    setTime(timeRef.current);
  };
  const restart = () => {
    timeRef.current = 0;
    setTime(0);
    setPlaying(true);
    clock.current.last = 0;
  };
  const changePolicy = (id: string) => {
    setCheckpoint(id);
    void load(config, seed, id, compareId);
  };
  const nav = (p: Page) => {
    setPage(p);
    setError("");
    if (p === "Watch AI Learn") {
      setComparison(null);
      setCompareId("");
    } else if (p === "Playground" && checkpoint) {
      setCheckpoint("");
      setCompareId("");
      void load(config, seed);
    }
  };
  const updateConfig = (key: string, value: number | string) =>
    setConfig((c) => ({ ...c, [key]: value }));
  const frame = replay ? atTime(replay, time) : null,
    player = frame && selected !== null ? frame.players[selected] : null;
  const isPolicyReplay = Boolean(
    replay?.policy.team && replay.policy.team !== "none",
  );
  const visibleEvents = replay?.events.filter((e) => e.time <= time) || [];
  const latest = visibleEvents.at(-1);
  const exportReplay = () => {
    if (!replay) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(replay)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `rondo-${replay.seed}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  const importReplay = async (file?: File) => {
    if (!file) return;
    setLoading(true);
    try {
      const r = await api<Replay>(
        "/replays/validate",
        JSON.parse(await file.text()),
      );
      setReplay(r);
      setComparison(null);
      setCheckpoint("");
      setTime(0);
      timeRef.current = 0;
      setPlaying(false);
      setSeed(r.seed);
      setEpisode(r.seed >= 42 && r.seed <= 46 ? r.seed - 42 : -1);
      setConfig(r.config);
      setError("");
    } catch (e) {
      setError(`Replay import failed: ${e}`);
    } finally {
      setLoading(false);
      if (input.current) input.current.value = "";
    }
  };
  const startTraining = async () => {
    setBusy(true);
    setError("");
    try {
      await api("/training", {
        mode: trainMode,
        scenario,
        timesteps: steps,
        seed: trainSeed,
        checkpoint_interval: interval,
        config: { ...config, reward },
      });
      setTraining(await api("/training"));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  const selectPlayer = (id: number) => {
    setSelected(id);
    setSideTab("Player");
  };
  const policyLabel = (p: Policy) =>
    `${p.stage === "untrained" ? "Untrained" : p.stage === "intermediate" ? "Intermediate" : "Trained checkpoint"} · ${p.steps.toLocaleString()} steps · ${p.team}`;
  const renderStage = () => (
    <>
      <div className="session-bar">
        <div>
          <span className="live-mark" />
          <strong>
            {isPolicyReplay ? "Saved policy replay" : "Scripted 6v2"}
          </strong>
          <span className="muted">
            {isPolicyReplay
              ? `${Number(replay?.policy.steps || 0).toLocaleString()} environment steps`
              : "Cover & press baseline"}
          </span>
        </div>
        <div className="session-meta">
          <span>Seed {replay?.seed || seed}</span>
          <span>
            {Math.round(1 / Number(replay?.config.dt || config.dt || 0.05))} Hz
            physics
          </span>
          <span>6 vs 2</span>
          <button
            className="focus-toggle"
            aria-pressed={pitchFocus}
            onClick={() => setPitchFocus((p) => !p)}
          >
            {pitchFocus ? "Exit pitch focus" : "Focus pitch"}
          </button>
        </div>
      </div>
      {page === "Watch AI Learn" && (
        <div className="learn-picker">
          <div className="learn-intro">
            <h2>Same setup. Different decisions.</h2>
            <p>
              Replay recorded policy weights from one initial scenario. A
              training stage describes its provenance, not its quality.
            </p>
          </div>
          <label>
            Policy checkpoint
            <select
              value={checkpoint}
              onChange={(e) => changePolicy(e.target.value)}
            >
              <option value="">Scripted baseline</option>
              {policies.map((p) => (
                <option key={p.id} value={p.id}>
                  {policyLabel(p)} · {p.id.split("/")[0]}
                </option>
              ))}
            </select>
          </label>
          <label>
            Compare alongside
            <select
              value={compareId}
              onChange={(e) => {
                setCompareId(e.target.value);
                void load(config, seed, checkpoint, e.target.value);
              }}
            >
              <option value="">Single replay</option>
              {policies.map((p) => (
                <option key={p.id} value={p.id}>
                  {policyLabel(p)} · {p.id.split("/")[0]}
                </option>
              ))}
            </select>
          </label>
          <p className="provenance">
            {policies.length
              ? `${policies.length} actual checkpoint files available. ${policies.some((p) => p.stage === "intermediate") ? "Intermediate checkpoints recorded." : "No intermediate checkpoints have been recorded."} Short runs verify training; they do not establish strong football.`
              : "No checkpoints recorded yet. Run PPO in Training to create real learning stages."}
          </p>
        </div>
      )}
      <div className={"workbench " + (comparison ? "compare-mode" : "")}>
        <section className="pitch-column" aria-label="Match replay">
          <div className={"pitch-stage " + (comparison ? "dual" : "")}>
            {loading && (
              <div className="loading" role="status">
                <span className="spinner" />
                Loading episode…
              </div>
            )}
            {replay ? (
              <>
                <div className="pitch-wrap">
                  {comparison && (
                    <div className="pitch-caption">
                      A · {checkpoint || "Scripted baseline"}
                    </div>
                  )}
                  <Pitch
                    replay={replay}
                    time={time}
                    layers={layers}
                    selected={selected}
                    onSelect={selectPlayer}
                  />
                  <div className="pitch-hud">
                    <span>
                      <i className="white-dot" />
                      Possession
                    </span>
                    <span>
                      <i className="red-dot" />
                      Pressing
                    </span>
                    <span className="ball-status">
                      {frame?.ball.in_flight
                        ? "Ball in transit"
                        : frame?.ball.carrier !== null
                          ? `Ball with ${frame!.ball.carrier! < 6 ? "P" + (frame!.ball.carrier! + 1) : "D" + (frame!.ball.carrier! - 5)}`
                          : "Loose ball"}
                    </span>
                  </div>
                </div>
                {comparison && (
                  <div className="pitch-wrap">
                    <div className="pitch-caption">B · {compareId}</div>
                    <Pitch
                      replay={comparison}
                      time={time}
                      layers={layers}
                      selected={selected}
                      onSelect={selectPlayer}
                      compact
                    />
                  </div>
                )}
              </>
            ) : (
              !loading && (
                <div className="empty">
                  No replay loaded. Generate an episode or import a replay.
                </div>
              )
            )}
          </div>
          <div className="playback">
            <div className="timeline-top">
              <span>{fmt(time)}</span>
              <span className="event-caption">
                {latest
                  ? `${pretty(latest.type)}${latest.player !== undefined ? ` · P${latest.player + 1}` : ""}`
                  : "Initial formation"}
              </span>
              <span>{fmt(end)}</span>
            </div>
            <div className="timeline">
              <input
                aria-label="Episode timeline"
                type="range"
                min="0"
                max={end || 1}
                step=".01"
                value={time}
                onChange={(e) => scrub(Number(e.target.value))}
                style={
                  {
                    "--progress": `${(time / (end || 1)) * 100}%`,
                  } as React.CSSProperties
                }
              />
              <div className="event-marks">
                {replay?.events.map((e, i) => (
                  <button
                    key={i}
                    className={
                      ["interception", "tackle", "out_of_bounds"].includes(
                        e.type,
                      )
                        ? "turnover-mark"
                        : ""
                    }
                    style={{ left: `${(e.time / (end || 1)) * 100}%` }}
                    title={`${pretty(e.type)} at ${fmt(e.time)}`}
                    aria-label={`Jump to ${pretty(e.type)} at ${fmt(e.time)}`}
                    onClick={() => scrub(e.time)}
                  />
                ))}
              </div>
            </div>
            <div className="transport">
              <div className="transport-main">
                <button
                  aria-label="Restart episode"
                  title="Restart"
                  onClick={restart}
                >
                  <Icon name="restart" />
                </button>
                <button
                  className="play"
                  aria-label={playing ? "Pause playback" : "Play playback"}
                  onClick={() => {
                    if (timeRef.current >= end) restart();
                    else setPlaying((p) => !p);
                  }}
                  disabled={!replay || loading}
                >
                  <Icon name={playing ? "pause" : "play"} />
                </button>
                <button
                  aria-label="Step one frame"
                  title="Step 0.05 seconds"
                  onClick={() => scrub(time + 0.05)}
                >
                  <Icon name="step" />
                </button>
                <select
                  aria-label="Playback speed"
                  value={speed}
                  onChange={(e) => setSpeed(Number(e.target.value))}
                >
                  {[0.25, 0.5, 1, 1.5, 2, 4].map((s) => (
                    <option key={s} value={s}>
                      {s}×
                    </option>
                  ))}
                </select>
              </div>
              <label className="check compact-check">
                <input
                  type="checkbox"
                  checked={loop}
                  onChange={(e) => setLoop(e.target.checked)}
                />
                Loop episode
              </label>
              <label className="episode-select">
                Episode
                <select
                  aria-label="Episode selection"
                  value={episode}
                  onChange={(e) => {
                    const n = Number(e.target.value);
                    setEpisode(n);
                    setSeed(42 + n);
                    void load(config, 42 + n, checkpoint, compareId);
                  }}
                >
                  {episode === -1 && (
                    <option value={-1}>
                      Custom · seed {replay?.seed ?? seed}
                    </option>
                  )}
                  {[0, 1, 2, 3, 4].map((n) => (
                    <option value={n} key={n}>
                      {String(n + 1).padStart(2, "0")} · seed {42 + n}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>
          <div className="match-stats">
            <div>
              <span>Completed passes</span>
              <strong>
                {frame?.passes || 0}
                <small> / {frame?.attempts || 0}</small>
              </strong>
            </div>
            <div>
              <span>Pass completion</span>
              <strong>
                {frame?.attempts
                  ? Math.round((frame.passes / frame.attempts) * 100)
                  : 0}
                <small>%</small>
              </strong>
            </div>
            <div>
              <span>Possession time</span>
              <strong>
                {Math.min(time, replay?.frames.at(-1)!.time || 0).toFixed(1)}
                <small>s</small>
              </strong>
            </div>
            <div>
              <span>Sequence status</span>
              <strong className="status-text">
                {replay && time >= replay.frames.at(-1)!.time
                  ? pretty(String(replay.metrics.reason))
                  : "In play"}
              </strong>
            </div>
          </div>
          <div className="event-strip">
            <span className="section-label">Sequence</span>
            {visibleEvents
              .filter((e) => e.type === "reception")
              .slice(-8)
              .map((e, i) => (
                <button key={i} onClick={() => scrub(e.time)}>
                  {e.origin !== undefined ? `P${e.origin + 1}` : "?"}{" "}
                  <span>→</span> P{e.player! + 1}
                  <small>{fmt(e.time)}</small>
                </button>
              ))}
            {!visibleEvents.some((e) => e.type === "reception") && (
              <span className="muted">Completed passes will appear here.</span>
            )}
          </div>
        </section>
        {!comparison && (
          <aside className="inspector">
            <div className="tabs" role="tablist" aria-label="Inspection panels">
              {(["Tactics", "Player", "Session"] as const).map((t) => (
                <button
                  key={t}
                  role="tab"
                  aria-selected={sideTab === t}
                  onClick={() => setSideTab(t)}
                >
                  {t}
                </button>
              ))}
            </div>
            {sideTab === "Tactics" && (
              <>
                <div className="panel-heading">
                  <h2>Read the game</h2>
                  <p>Layer tactical information onto the pitch.</p>
                </div>
                <div className="layer-list">
                  {Object.entries(layerNames).map(([key, label]) => (
                    <label key={key} className="layer">
                      <span>{label}</span>
                      <input
                        type="checkbox"
                        role="switch"
                        checked={layers[key as keyof Layers]}
                        onChange={(e) =>
                          setLayers((l) => ({ ...l, [key]: e.target.checked }))
                        }
                      />
                    </label>
                  ))}
                </div>
                <div className="panel-note">
                  Passing lanes use the current ball position and defender
                  reach. An open lane is a geometric observation, not a
                  guarantee.
                </div>
                <button
                  className="text-button"
                  onClick={() => setLayers(defaultLayers)}
                >
                  Reset overlays
                </button>
              </>
            )}
            {sideTab === "Player" && (
              <div className="panel-content">
                <h2>
                  {player
                    ? `${player.id < 6 ? "Possession player" : "Defender"} ${player.id + 1}`
                    : "Inspect a player"}
                </h2>
                {player ? (
                  <>
                    <p>
                      {player.control} policy · {player.team}
                    </p>
                    <dl>
                      <dt>Position</dt>
                      <dd>
                        {player.position.map((n) => n.toFixed(2)).join(", ")} m
                      </dd>
                      <dt>Speed</dt>
                      <dd>{Math.hypot(...player.velocity).toFixed(2)} m/s</dd>
                      <dt>Facing</dt>
                      <dd>
                        {(
                          ((((player.facing * 180) / Math.PI) % 360) + 360) %
                          360
                        ).toFixed(0)}
                        °
                      </dd>
                      <dt>Intention</dt>
                      <dd>{player.intention}</dd>
                      <dt>Selected action</dt>
                      <dd>
                        {
                          (player.id < 6 ? labels : defenderLabels)[
                            player.action
                          ]
                        }
                      </dd>
                      <dt>Value estimate</dt>
                      <dd>
                        {player.value?.toFixed(3) ?? "Scripted: unavailable"}
                      </dd>
                    </dl>
                    {player.probabilities ? (
                      <>
                        <h3>Action probabilities</h3>
                        <p>
                          Policy outputs at the last decision. These are not an
                          explanation of internal reasoning.
                        </p>
                        {player.probabilities.map(
                          (prob, i) =>
                            prob > 0 && (
                              <div className="probability" key={i}>
                                <span>
                                  {(player.id < 6 ? labels : defenderLabels)[i]}
                                </span>
                                <b>{(prob * 100).toFixed(1)}%</b>
                                <div style={{ width: `${prob * 100}%` }} />
                              </div>
                            ),
                        )}
                      </>
                    ) : (
                      <p className="panel-note">
                        Scripted agents use football heuristics. They do not
                        expose neural action probabilities.
                      </p>
                    )}
                  </>
                ) : (
                  <p>
                    Select a shirt on the pitch with a click or keyboard focus.
                  </p>
                )}
                <h3>Reward components</h3>
                {Object.entries(frame?.reward_components || {}).map(
                  ([key, value]) => (
                    <div className="data-row" key={key}>
                      <span>{pretty(key)}</span>
                      <b>{value.toFixed(3)}</b>
                    </div>
                  ),
                )}
                <p className="muted">
                  Possession reward for the current decision interval; defending
                  uses its negative.
                </p>
              </div>
            )}
            {sideTab === "Session" && (
              <div className="panel-content">
                <h2>Session setup</h2>
                <label>
                  Drill preset
                  <select
                    value={preset}
                    onChange={(e) => {
                      setPreset(e.target.value);
                      setConfig({ ...defaults, ...presets[e.target.value] });
                    }}
                  >
                    {Object.keys(presets).map((p) => (
                      <option key={p}>{p}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Random seed
                  <input
                    type="number"
                    min="0"
                    max="2147483647"
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                  />
                </label>
                <NumberControl
                  label="Defender speed"
                  unit="m/s"
                  name="defender_speed"
                  config={config}
                  min={0}
                  max={7}
                  step={0.1}
                  onChange={updateConfig}
                />
                <NumberControl
                  label="Ball speed"
                  unit="m/s"
                  name="ball_speed"
                  config={config}
                  min={4}
                  max={18}
                  step={0.5}
                  onChange={updateConfig}
                />
                <label>
                  Defending strategy
                  <select
                    value={config.defender_strategy || "cover_press"}
                    onChange={(e) =>
                      updateConfig("defender_strategy", e.target.value)
                    }
                  >
                    {["chase", "nearest", "lanes", "cover_press"].map((s) => (
                      <option key={s} value={s}>
                        {pretty(s)}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  className="primary"
                  onClick={() => void load(config, seed, checkpoint, compareId)}
                >
                  Generate episode
                </button>
                <p>Settings apply when you generate a new episode.</p>
                <div className="file-controls">
                  <button onClick={exportReplay}>Export replay</button>
                  <button onClick={() => input.current?.click()}>
                    Import replay
                  </button>
                </div>
              </div>
            )}
          </aside>
        )}
      </div>
    </>
  );
  return (
    <>
      <a href="#main" className="skip-link">
        Skip to simulator
      </a>
      <div className="app-shell">
        <nav className="sidebar" aria-label="Main navigation">
          <a className="brand" href="#main" onClick={() => nav("Playground")}>
            <span className="brand-mark">
              r<span>.</span>
            </span>
            <span>
              rondo<span className="brand-light">lab</span>
            </span>
          </a>
          <div className="nav-label">Workspace</div>
          <div className="nav-items">
            {pages.map((p, i) => (
              <button
                key={p}
                className={page === p ? "active" : ""}
                onClick={() => nav(p)}
              >
                <Icon
                  name={
                    [
                      "pitch",
                      "learn",
                      "training",
                      "experiments",
                      "book",
                      "settings",
                    ][i]
                  }
                />
                <span>{p}</span>
                {p === "Watch AI Learn" && (
                  <span className="nav-counter">
                    {policies.filter((p) => p.steps > 0).length}
                  </span>
                )}
              </button>
            ))}
          </div>
          <div className="sidebar-bottom">
            <div className="tiny-pitch">
              <span />
              <i />
              <b />
            </div>
            <p>
              Football intelligence,
              <br />
              one decision at a time.
            </p>
            <span className="local-status">
              <i />
              Local workspace
            </span>
            <small>Rondo Lab / v0.1</small>
          </div>
        </nav>
        <main id="main">
          <header className="topbar">
            <div>
              <span className="breadcrumb">
                Workspace <span>/</span> {page}
              </span>
            </div>
            <div className="topbar-right">
              <span className="mode-label">6v2 football laboratory</span>
              <button
                onClick={() => {
                  setPage("RL Explorer");
                }}
              >
                How it works <Icon name="book" />
              </button>
            </div>
          </header>
          <div className="page-content">
            <div className="page-heading">
              <div>
                <div className="eyebrow">
                  {page === "Playground"
                    ? "The training ground"
                    : page === "Watch AI Learn"
                      ? "Policy replay"
                      : page === "Training"
                        ? "Learning by playing"
                        : page === "Experiments"
                          ? "Evidence from the pitch"
                          : page === "RL Explorer"
                            ? "Inside the learning loop"
                            : "Your football environment"}
                </div>
                <h1>{page}</h1>
                <p>
                  {page === "Playground"
                    ? "Keep the ball. Find the angle. Read the press."
                    : page === "Watch AI Learn"
                      ? "Watch how decisions change across genuine training checkpoints."
                      : page === "Training"
                        ? "Train shared policies and evaluate what actually changes."
                        : page === "Experiments"
                          ? "Compare saved runs on fixed evaluation scenarios."
                          : page === "RL Explorer"
                            ? "Connect reinforcement learning concepts to the match in front of you."
                            : "Configure the physical drill. Regenerate an episode to apply your changes."}
                </p>
              </div>
              <div className="heading-actions">
                {["Playground", "Watch AI Learn"].includes(page) && (
                  <>
                    <button onClick={exportReplay} disabled={!replay}>
                      <Icon name="download" />
                      Export replay
                    </button>
                    <button
                      className="primary"
                      onClick={() => {
                        setSeed((s) => s + 1);
                        void load(config, seed + 1, checkpoint, compareId);
                      }}
                      disabled={loading}
                    >
                      <Icon name="restart" />
                      New episode
                    </button>
                  </>
                )}
              </div>
            </div>
            {error && (
              <div className="error" role="alert">
                {error}
                <button
                  onClick={() => {
                    setError("");
                    if (!replay && Object.keys(config).length)
                      void load(config, seed);
                    else if (!Object.keys(config).length)
                      window.location.reload();
                  }}
                >
                  Retry
                </button>
              </div>
            )}
            {["Playground", "Watch AI Learn"].includes(page) && renderStage()}
            {page === "Training" && (
              <div className="training-layout">
                <section className="form-panel">
                  <h2>Set up a training run</h2>
                  <p>PyTorch PPO · separate local process · CPU</p>
                  <label>
                    Learning mode
                    <select
                      value={trainMode}
                      onChange={(e) => setTrainMode(e.target.value)}
                    >
                      <option value="possession">
                        A · Possession learning
                      </option>
                      <option value="pressing">B · Pressing learning</option>
                      <option value="both">
                        C · Alternating competitive learning
                      </option>
                    </select>
                  </label>
                  <label>
                    Possession actions
                    <select
                      value={scenario}
                      onChange={(e) => setScenario(e.target.value)}
                    >
                      <option value="basic">
                        Basic passing · carrier decisions
                      </option>
                      <option value="adaptive">
                        Adaptive possession · off-ball movement
                      </option>
                    </select>
                  </label>
                  <label>
                    Reward design
                    <select
                      value={reward}
                      onChange={(e) => setReward(e.target.value)}
                    >
                      <option value="sparse">
                        Sparse · passes and turnovers
                      </option>
                      <option value="moderate">
                        Moderate · add possession time
                      </option>
                      <option value="heavy">
                        Heavy · add pressure shaping
                      </option>
                    </select>
                  </label>
                  <label>
                    Environment decision steps
                    <input
                      type="number"
                      min="32"
                      max="10000000"
                      value={steps}
                      onChange={(e) => setSteps(Number(e.target.value))}
                    />
                  </label>
                  <label>
                    Checkpoint interval
                    <input
                      type="number"
                      min="32"
                      value={interval}
                      onChange={(e) => setIntervalSteps(Number(e.target.value))}
                    />
                  </label>
                  <label>
                    Training seed
                    <input
                      type="number"
                      min="0"
                      value={trainSeed}
                      onChange={(e) => setTrainSeed(Number(e.target.value))}
                    />
                  </label>
                  <p>
                    Evaluation uses separate seeds starting at 10001. Current
                    pitch physics and rules apply to this run.
                  </p>
                  <div className="file-controls">
                    <button
                      className="primary"
                      disabled={busy || training.status === "running"}
                      onClick={() => void startTraining()}
                    >
                      Start PPO training
                    </button>
                    <button
                      disabled={training.status !== "running"}
                      onClick={async () => {
                        try {
                          await api("/training", undefined, "DELETE");
                          setTraining(await api("/training"));
                          void refresh();
                        } catch (e) {
                          setError(String(e));
                        }
                      }}
                    >
                      Stop training
                    </button>
                  </div>
                </section>
                <section className="training-results">
                  <div className="section-head">
                    <h2>Training monitor</h2>
                    <span className="status-badge">{training.status}</span>
                  </div>
                  {training.progress ? (
                    <>
                      <progress
                        max={training.progress.target}
                        value={training.progress.steps}
                      />
                      <p>
                        {training.progress.steps.toLocaleString()} /{" "}
                        {training.progress.target.toLocaleString()} steps ·{" "}
                        {training.progress.updates} updates ·{" "}
                        {training.progress.episodes} episodes
                      </p>
                      <div className="metric-list">
                        {Object.entries(training.progress.latest)
                          .filter(([k]) =>
                            [
                              "policy_loss",
                              "value_loss",
                              "entropy",
                              "approx_kl",
                              "explained_variance",
                            ].includes(k),
                          )
                          .map(([k, v]) => (
                            <div className="data-row" key={k}>
                              <span>{pretty(k)}</span>
                              <b>
                                {typeof v === "number"
                                  ? v.toFixed(4)
                                  : "Unavailable"}
                              </b>
                            </div>
                          ))}
                      </div>
                      <h3>Latest fixed-seed evaluation</h3>
                      <p>
                        {training.progress.latest.evaluation.mean_passes.toFixed(
                          1,
                        )}{" "}
                        mean passes ·{" "}
                        {(
                          training.progress.latest.evaluation.pass_completion *
                          100
                        ).toFixed(0)}
                        % completion
                      </p>
                    </>
                  ) : (
                    <div className="empty">
                      <Icon name="training" />
                      <h3>
                        {training.status === "running"
                          ? "Collecting the first rollout"
                          : "Ready for a real experiment"}
                      </h3>
                      <p>
                        {training.status === "running"
                          ? "Metrics will appear after the first PPO update."
                          : "Start a short run to verify the pipeline, then train longer to investigate football behaviour."}
                      </p>
                    </div>
                  )}
                  {training.error && (
                    <pre className="error">{training.error}</pre>
                  )}
                  <h3>Recorded checkpoints</h3>
                  {policies.length ? (
                    <div className="checkpoint-list">
                      {policies.slice(-12).map((p) => (
                        <button
                          key={p.id}
                          onClick={() => {
                            setPage("Watch AI Learn");
                            setCheckpoint(p.id);
                            void load(config, seed, p.id);
                          }}
                        >
                          <span>
                            {p.id.split("/")[0]}
                            <small>{policyLabel(p)}</small>
                          </span>
                          <Icon name="play" />
                        </button>
                      ))}
                    </div>
                  ) : (
                    <p>No checkpoints yet.</p>
                  )}
                  <p className="panel-note">
                    A short run proves the optimizer and replay pipeline work.
                    Its final checkpoint may play worse than a scripted or
                    random policy.
                  </p>
                </section>
              </div>
            )}
            {page === "Experiments" && (
              <section className="experiment-page">
                <div className="comparison-selectors">
                  <label>
                    Run A
                    <select
                      value={experimentA || experiments[0]?.id || ""}
                      onChange={(e) => setExperimentA(e.target.value)}
                    >
                      {experiments.map((e) => (
                        <option key={e.id}>{e.id}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Run B
                    <select
                      value={experimentB}
                      onChange={(e) => setExperimentB(e.target.value)}
                    >
                      <option value="">Select another run</option>
                      {experiments.map((e) => (
                        <option key={e.id}>{e.id}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Evaluation metric
                    <select
                      value={chartMetric}
                      onChange={(e) => setChartMetric(e.target.value)}
                    >
                      <option value="mean_passes">Mean passing streak</option>
                      <option value="pass_completion">Pass completion</option>
                      <option value="mean_duration">Possession duration</option>
                      <option value="interceptions">
                        Defensive interceptions
                      </option>
                    </select>
                  </label>
                  <button
                    onClick={() =>
                      void refresh().catch((e) => setError(String(e)))
                    }
                  >
                    Refresh saved runs
                  </button>
                </div>
                {experiments.length ? (
                  <>
                    <LearningChart
                      experiments={experiments.filter(
                        (e) =>
                          e.id === (experimentA || experiments[0]?.id) ||
                          e.id === experimentB,
                      )}
                      metric={chartMetric}
                    />
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Experiment</th>
                            <th>Steps</th>
                            <th>Mean passes</th>
                            <th>95% CI ±</th>
                            <th>Completion</th>
                            <th>Duration</th>
                            <th>Interceptions</th>
                          </tr>
                        </thead>
                        <tbody>
                          {experiments.map((e) => {
                            const row = e.history.at(-1);
                            return (
                              row && (
                                <tr key={e.id}>
                                  <td>{e.id}</td>
                                  <td>{row.steps}</td>
                                  <td>
                                    {row.evaluation.mean_passes.toFixed(2)}
                                  </td>
                                  <td>
                                    {row.evaluation.passes_ci95_normal.toFixed(
                                      2,
                                    )}
                                  </td>
                                  <td>
                                    {(
                                      row.evaluation.pass_completion * 100
                                    ).toFixed(1)}
                                    %
                                  </td>
                                  <td>
                                    {row.evaluation.mean_duration.toFixed(1)}s
                                  </td>
                                  <td>{row.evaluation.interceptions}</td>
                                </tr>
                              )
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                    <p>
                      Curves contain actual logged PPO updates. Intervals use a
                      normal approximation across scenario seeds; they do not
                      measure variation across independently trained runs. Small
                      evaluation batches are noisy.
                    </p>
                  </>
                ) : (
                  <div className="empty">
                    No saved experiments. Run PPO training to record an
                    experiment.
                  </div>
                )}
              </section>
            )}
            {page === "RL Explorer" && (
              <div className="explorer">
                <section className="concept-list">
                  <h2>The decision loop</h2>
                  {[
                    [
                      "State & observation",
                      "The world stores all eight players, ball motion and rules. Each actor receives 90 normalized, player-relative features: positions, velocities, body orientation, possession and boundaries. No future action is included.",
                    ],
                    [
                      "Action & policy",
                      "The actor turns the current observation into a categorical distribution over legal actions. Impossible passes are masked before sampling and before the PPO loss. A shared team policy makes separate decisions for each player.",
                    ],
                    [
                      "Reward & shaping",
                      "Sparse: +1 per completed pass, loss penalty on a turnover. Moderate adds +0.02 per second of possession. Heavy uses +1.5 per pass and a small pressure penalty. Defenders receive the opposite team reward. Invalid actions cost their actor 0.05.",
                    ],
                    [
                      "Value function & discount",
                      "The value head predicts discounted future team reward from the actor’s observation. Gamma 0.99 weights future rewards. GAE combines reward prediction errors over the rollout, bootstrapping time limits and stopping at resets.",
                    ],
                    [
                      "PPO & exploration",
                      "PPO clips changes in action probability ratios to 0.8–1.2. An entropy bonus encourages exploration. The actor and local critic share a 96 × 96 network; gradients are clipped to 0.5. Evaluation uses deterministic highest-probability actions.",
                    ],
                    [
                      "Cooperation & self-play",
                      "Six possession players share weights, as do two defenders. Competitive mode alternates team updates against frozen opponents from a pool of up to five checkpoints. Its decentralized critic uses no extra global information.",
                    ],
                    [
                      "Curriculum",
                      "Difficulty can be selected manually in Settings. The Python curriculum module recommends promotion only after at least ten evaluation episodes exceed a passing threshold. Promotion is not triggered by elapsed training steps.",
                    ],
                  ].map(([title, text], i) => (
                    <details key={title} open={i === 0}>
                      <summary>
                        <span>{String(i + 1).padStart(2, "0")}</span>
                        {title}
                      </summary>
                      <p>{text}</p>
                    </details>
                  ))}
                </section>
                <aside className="guided">
                  <h2>Try an experiment</h2>
                  <p>These are questions to test, not promised outcomes.</p>
                  {[
                    [
                      "Faster defenders",
                      "defender_speed",
                      3.8,
                      "Does faster pressure reduce passing streaks?",
                    ],
                    [
                      "Costly turnovers",
                      "loss_penalty",
                      5,
                      "Does a larger loss penalty change receiver selection?",
                    ],
                    [
                      "Coordinated pressing",
                      "defender_strategy",
                      "cover_press",
                      "How does cover and press compare with chasing?",
                    ],
                    [
                      "Harder opponents",
                      "defender_speed",
                      3.4,
                      "Does a policy trained on easy opponents generalize?",
                    ],
                  ].map(([title, key, value, desc]) => (
                    <div className="guided-item" key={String(title)}>
                      <h3>{title}</h3>
                      <p>{desc}</p>
                      <button
                        onClick={() => {
                          updateConfig(String(key), value);
                          setSideTab("Session");
                          setPage("Playground");
                          void load(
                            { ...config, [String(key)]: value },
                            seed,
                            checkpoint,
                          );
                        }}
                      >
                        Run this setup
                      </button>
                    </div>
                  ))}
                  <h3>Learned vs random passing</h3>
                  <p>
                    Use the CLI evaluation command with a saved checkpoint and
                    compare the same evaluation seeds against the random and
                    heuristic baselines. See README for commands and metric
                    definitions.
                  </p>
                </aside>
              </div>
            )}
            {page === "Settings" && (
              <div className="settings-layout">
                <section className="form-panel">
                  <h2>Drill & physics</h2>
                  <p>
                    The fully supported configuration is six possession players
                    and two defenders.
                  </p>
                  <label>
                    Preset
                    <select
                      value={preset}
                      onChange={(e) => {
                        setPreset(e.target.value);
                        setConfig({ ...defaults, ...presets[e.target.value] });
                      }}
                    >
                      {Object.keys(presets).map((p) => (
                        <option key={p}>{p}</option>
                      ))}
                    </select>
                  </label>
                  {[
                    ["Rondo width", "width", 10, 25, 0.5, "m"],
                    ["Rondo height", "height", 10, 25, 0.5, "m"],
                    ["Episode duration", "duration", 5, 180, 5, "s"],
                    ["Possession speed", "player_speed", 0.1, 7, 0.1, "m/s"],
                    ["Defender speed", "defender_speed", 0, 7, 0.1, "m/s"],
                    ["Acceleration", "acceleration", 1, 15, 0.5, "m/s²"],
                    ["Turning rate", "turning_rate", 0.5, 12, 0.5, "rad/s"],
                    ["Ball speed", "ball_speed", 4, 18, 0.5, "m/s"],
                    ["Rolling resistance", "resistance", 0, 2, 0.05, "m/s²"],
                    ["Pass error", "pass_error", 0, 0.3, 0.005, "rad"],
                    ["Control radius", "control_radius", 0.3, 1, 0.05, "m"],
                    [
                      "Interception radius",
                      "interception_radius",
                      0.2,
                      1,
                      0.05,
                      "m",
                    ],
                    ["Reaction delay", "reaction_time", 0, 0.6, 0.05, "s"],
                    ["Support freedom", "support_radius", 0, 3, 0.1, "m"],
                    ["Maximum hold", "max_hold", 0.4, 10, 0.1, "s"],
                    ["Turnover penalty", "loss_penalty", 0, 10, 0.5, "reward"],
                  ].map(([label, name, min, max, step, unit]) => (
                    <NumberControl
                      key={name}
                      label={String(label)}
                      name={String(name)}
                      min={Number(min)}
                      max={Number(max)}
                      step={Number(step)}
                      unit={String(unit)}
                      config={config}
                      onChange={updateConfig}
                    />
                  ))}
                </section>
                <section className="form-panel">
                  <h2>Football rules</h2>
                  <label>
                    Touch constraint
                    <select
                      value={config.max_touches ?? 0}
                      onChange={(e) =>
                        updateConfig("max_touches", Number(e.target.value))
                      }
                    >
                      <option value="0">Unlimited</option>
                      <option value="1">One touch · quick release</option>
                      <option value="2">
                        Two touch · control then release
                      </option>
                    </select>
                  </label>
                  <p>
                    Touches are approximated by a release window, not individual
                    foot contacts. One touch shortens scripted hold time. The
                    maximum holding limit always applies to learned policies.
                  </p>
                  <label>
                    Repeated-pair limit
                    <input
                      type="number"
                      min="0"
                      max="8"
                      value={config.pair_limit ?? 0}
                      onChange={(e) =>
                        updateConfig("pair_limit", Number(e.target.value))
                      }
                    />
                  </label>
                  <p>
                    0 disables the restriction. Otherwise repeated passes within
                    the same pair are masked after the limit.
                  </p>
                  <label>
                    Difficulty level
                    <select
                      value={config.curriculum || 4}
                      onChange={(e) => {
                        const level = Number(e.target.value),
                          speeds = [0, 0, 1, 1.8, 2.3, 2.8, 3.2],
                          strategies = [
                            "",
                            "chase",
                            "chase",
                            "nearest",
                            "lanes",
                            "cover_press",
                            "cover_press",
                          ];
                        setConfig((c) => ({
                          ...c,
                          curriculum: level,
                          defender_speed: speeds[level],
                          defender_strategy: strategies[level],
                        }));
                      }}
                    >
                      {[
                        "Stationary",
                        "Slow chase",
                        "Simple press",
                        "Passing lanes",
                        "Cover & press",
                        "Competitive opponents",
                      ].map((s, i) => (
                        <option key={s} value={i + 1}>
                          {i + 1} · {s}
                        </option>
                      ))}
                    </select>
                  </label>
                  <p>
                    Level 6 changes physical difficulty here. Select alternating
                    competitive mode in Training to use frozen learned
                    opponents.
                  </p>
                  <button
                    className="primary"
                    onClick={() => {
                      setPage("Playground");
                      void load(config, seed, checkpoint);
                    }}
                  >
                    Apply & watch episode
                  </button>
                  <button
                    onClick={() => {
                      setConfig(defaults);
                      setPreset("Classic 6v2");
                    }}
                  >
                    Reset to defaults
                  </button>
                  <h3>Replay files</h3>
                  <p>
                    Exported replays contain all states and policy outputs.
                    Watching an imported replay needs no model weights.
                  </p>
                  <div className="file-controls">
                    <button onClick={exportReplay}>
                      Export current replay
                    </button>
                    <button onClick={() => input.current?.click()}>
                      Import replay
                    </button>
                  </div>
                </section>
              </div>
            )}
            <footer>
              <span>Rondo Lab</span>
              <span>
                Scripted intelligence ≠ learned policy. Every replay comes from
                the simulator.
              </span>
              <span className="keyboard-hint">Space to pause · → to step</span>
            </footer>
          </div>
        </main>
      </div>
      <input
        ref={input}
        className="visually-hidden"
        type="file"
        accept=".json,application/json"
        aria-label="Import replay file"
        onChange={(e) => void importReplay(e.target.files?.[0])}
      />
    </>
  );
}
function NumberControl({
  label,
  name,
  config,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string;
  name: string;
  config: Config;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (key: string, value: number) => void;
}) {
  return (
    <label className="number-control">
      <span>
        {label}
        <b>
          {Number(config[name] ?? min).toFixed(
            step < 0.01 ? 3 : step < 1 ? 2 : 0,
          )}{" "}
          <small>{unit}</small>
        </b>
      </span>
      <input
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={config[name] ?? min}
        onChange={(e) => onChange(name, Number(e.target.value))}
      />
    </label>
  );
}
function LearningChart({
  experiments,
  metric,
}: {
  experiments: Experiment[];
  metric: string;
}) {
  const values = experiments.flatMap((e) =>
      e.history.map((r) =>
        Number(r.evaluation[metric as keyof typeof r.evaluation]),
      ),
    ),
    max = Math.max(1, ...values) * 1.1,
    maxSteps = Math.max(
      1,
      ...experiments.flatMap((e) => e.history.map((r) => r.steps)),
    );
  return (
    <div className="learning-chart">
      <div className="section-head">
        <h2>{pretty(metric)} over training</h2>
        <div>
          {experiments.map((e, i) => (
            <span
              key={e.id}
              className="chart-legend"
              style={{ color: i ? "#e9a08c" : "#d1e5a6" }}
            >
              {e.id}
            </span>
          ))}
        </div>
      </div>
      <svg
        viewBox="0 0 900 300"
        role="img"
        aria-label="Actual evaluation metrics by environment steps"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line
              x1="60"
              y1={250 - t * 210}
              x2="870"
              y2={250 - t * 210}
              stroke="#3b443f"
            />
            <text x="45" y={255 - t * 210} textAnchor="end">
              {(t * max).toFixed(1)}
            </text>
          </g>
        ))}
        {experiments.map((e, i) => (
          <g key={e.id}>
            <polyline
              fill="none"
              stroke={i ? "#e9a08c" : "#d1e5a6"}
              strokeWidth="2"
              points={e.history
                .map(
                  (r) =>
                    `${60 + (r.steps / maxSteps) * 810},${250 - (Number(r.evaluation[metric as keyof typeof r.evaluation]) / max) * 210}`,
                )
                .join(" ")}
            />
            {e.history.map((r) => (
              <circle
                key={r.steps}
                cx={60 + (r.steps / maxSteps) * 810}
                cy={
                  250 -
                  (Number(r.evaluation[metric as keyof typeof r.evaluation]) /
                    max) *
                    210
                }
                r="3"
                fill={i ? "#e9a08c" : "#d1e5a6"}
              >
                <title>
                  {r.steps} steps:{" "}
                  {Number(
                    r.evaluation[metric as keyof typeof r.evaluation],
                  ).toFixed(3)}
                </title>
              </circle>
            ))}
          </g>
        ))}
        <text x="60" y="285">
          0
        </text>
        <text x="870" y="285" textAnchor="end">
          {maxSteps.toLocaleString()} environment steps
        </text>
      </svg>
    </div>
  );
}
function Icon({ name }: { name: string }) {
  const paths: Record<string, React.ReactNode> = {
    play: <path d="m9 5 11 7-11 7Z" fill="currentColor" stroke="none" />,
    pause: (
      <>
        <path
          d="M8 5h3v14H8zM15 5h3v14h-3z"
          fill="currentColor"
          stroke="none"
        />
      </>
    ),
    restart: (
      <>
        <path d="M4 10a8 8 0 1 1 1 8M4 4v6h6" />
      </>
    ),
    step: (
      <>
        <path d="m5 5 10 7-10 7ZM19 5v14" />
      </>
    ),
    pitch: (
      <>
        <rect x="3" y="5" width="18" height="14" rx="1" />
        <path d="M12 5v14M3 9h4v6H3m18-6h-4v6h4" />
        <circle cx="12" cy="12" r="3" />
      </>
    ),
    learn: (
      <>
        <path d="m3 9 9-5 9 5-9 5ZM6 11v6l6 3 6-3v-6M21 9v8" />
      </>
    ),
    training: (
      <>
        <path d="M4 19V9m8 10V4m8 15v-7M2 20h20" />
      </>
    ),
    experiments: (
      <>
        <path d="M8 3h8m-6 0v7l-6 9a1 1 0 0 0 1 2h14a1 1 0 0 0 1-2l-6-9V3M8 14h8" />
      </>
    ),
    book: (
      <>
        <path d="M12 5c-3-2-7-2-9-1v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-3-1-6-1-9 1Zm0 0v15" />
      </>
    ),
    settings: (
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="m10 3-1 3-3 1-3 3 2 2-1 4 3 1 2 4 3-2 4 1 1-3 4-2-2-3 1-4-3-1-2-3Z" />
      </>
    ),
    download: (
      <>
        <path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5" />
      </>
    ),
  };
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] || paths.pitch}
    </svg>
  );
}
