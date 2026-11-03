import { useEffect, useMemo, useState } from "react";
import { api, type Config, type Experiment, type Policy } from "./types";

type Opponent = {
  id: string;
  team: string;
  kind: string;
  generation: number;
  strategy?: string;
  steps?: number;
};
type Evaluation = {
  mean_passes: number;
  mean_duration: number;
  pass_completion: number;
  success_rate: number;
  interceptions: number;
  interception_frequency?: number;
  forced_turnover_rate: number;
  average_pressure: number;
  adjacent_pass_fraction: number;
  coordination_fraction: number;
  mean_time_to_regain: number | null;
  pass_direction_bins: number[];
  note?: string;
};
type LabMetric = {
  steps: number;
  generation: number;
  team: string;
  mean_step_reward: number;
  entropy: number;
  approx_kl: number;
  value_loss: number;
  reward_components: Record<string, number>;
  evaluation: Evaluation;
  historical_evaluation: Evaluation;
  opponent: Opponent;
  team_generation: Record<string, number>;
};
type Detail = {
  steps: number;
  target: number;
  generation: number;
  learning_team: string;
  frozen_team: string;
  sampled_opponent: Opponent;
  opponent_pool: Opponent[];
  team_generations: Record<string, number>;
  curriculum_stage: number;
  latest?: LabMetric;
};
type Job = {
  id?: string;
  status: string;
  detail?: Detail;
  progress?: Detail;
  resume?: string;
  error?: string;
};
type Session = {
  id: string;
  steps: number;
  generation: number;
  mode: string;
  scenario: string;
  seed: number;
  config: Config;
};
type ResultRow = Partial<Evaluation> & {
  flags?: string[];
  label?: string;
  possession?: string;
  defending?: string;
  passes?: number;
  duration?: number;
  average_nearest_defender_distance?: number;
};
type Results = {
  matchups?: ResultRow[];
  results?: ResultRow[];
  seeds: number[];
  note?: string;
};
const stageNames = [
  "Basic passing",
  "Under pressure",
  "Adaptive possession",
  "Learn defending",
  "Adversarial populations",
  "Generalization",
];
const short = (id: string) =>
  id.startsWith("scripted") || id === "random-possession"
    ? id
    : id.split("/").slice(-2).join("/");
const policyName = (p: Policy) =>
  `${p.team === "possession" ? "P" : "D"}${p.team_generation ?? p.generation ?? 0} · G${p.generation ?? 0} · ${p.steps.toLocaleString()} steps · ${p.id.split("/")[0]}`;
export default function AdversarialLab({
  policies,
  experiments,
  config,
  onChanged,
  onWatch,
}: {
  policies: Policy[];
  experiments: Experiment[];
  config: Config;
  onChanged: () => Promise<void>;
  onWatch: (possession: string, defending: string, seed: number) => void;
}) {
  const compatible = policies.filter((p) => p.environment_version === 2);
  const possession = compatible.filter((p) => p.team === "possession"),
    defending = compatible.filter((p) => p.team === "pressing");
  const [tab, setTab] = useState<"Training" | "Matchups" | "Experiments">(
      "Training",
    ),
    [job, setJob] = useState<Job>({ status: "idle" }),
    [sessions, setSessions] = useState<Session[]>([]),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const [mode, setMode] = useState("both"),
    [budget, setBudget] = useState("smoke"),
    [steps, setSteps] = useState(2048),
    [block, setBlock] = useState(256),
    [envs, setEnvs] = useState(2),
    [stage, setStage] = useState(5),
    [auto, setAuto] = useState(false),
    [evalEpisodes, setEvalEpisodes] = useState(2),
    [seed, setSeed] = useState(42),
    [device, setDevice] = useState("cpu"),
    [reward, setReward] = useState("moderate"),
    [resume, setResume] = useState("");
  const [picksP, setPicksP] = useState<string[]>(["scripted"]),
    [picksD, setPicksD] = useState<string[]>(["scripted"]),
    [run, setRun] = useState(""),
    [replaySeed, setReplaySeed] = useState(42),
    [generation, setGeneration] = useState("all"),
    [results, setResults] = useState<Results | null>(null),
    [experiment, setExperiment] = useState("geometry");
  const runs = experiments.filter(
    (e) =>
      e.metadata.environment_version === 2 &&
      e.metadata.engine_revision === "2.0",
  );
  const selectedRun =
    runs.find((e) => e.id === run) ||
    runs.find((e) => e.id === job.id) ||
    [...runs].sort(
      (a, b) =>
        Number(b.metadata.created_at || 0) -
          Number(a.metadata.created_at || 0) ||
        (b.history.at(-1)?.steps || 0) - (a.history.at(-1)?.steps || 0),
    )[0];
  const lineage = String(
    selectedRun?.metadata.lineage || selectedRun?.id || "",
  );
  const seriesPolicies = compatible.filter(
    (p) => (p.lineage || p.id.split("/")[0]) === lineage,
  );
  const history = (selectedRun?.history || []) as unknown as LabMetric[];
  const detail = job.detail || job.progress;
  const running = ["running", "pausing", "stopping"].includes(job.status);
  const generations = useMemo(
    () =>
      [...new Set(seriesPolicies.map((p) => p.generation ?? 0))].sort(
        (a, b) => a - b,
      ),
    [policies, lineage],
  );
  const refresh = async () => {
    await onChanged();
    setSessions(await api<Session[]>("/sessions"));
  };
  useEffect(() => {
    let mounted = true;
    let previous = "";
    const poll = async () => {
      try {
        const s = await api<Job>("/training");
        if (!mounted) return;
        setJob(s);
        if (
          s.status !== previous &&
          ["completed", "paused", "cancelled"].includes(s.status)
        ) {
          await onChanged();
          const list = await api<Session[]>("/sessions");
          if (mounted) setSessions(list);
        }
        previous = s.status;
      } catch (e) {
        if (mounted) setError(String(e));
      }
    };
    void poll();
    api<Session[]>("/sessions")
      .then((s) => {
        if (mounted) setSessions(s);
      })
      .catch((e) => setError(String(e)));
    const timer = window.setInterval(poll, 1500);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [onChanged]);
  const begin = async (resuming = false) => {
    setBusy(true);
    setError("");
    try {
      await api("/training", {
        mode,
        scenario: mode === "possession" && stage <= 2 ? "basic" : "adaptive",
        timesteps: steps,
        seed,
        checkpoint_interval: block,
        block_steps: block,
        num_envs: envs,
        config: { ...config, environment_version: 2, reward },
        resume: resuming ? resume || job.resume : null,
        device,
        curriculum_stage: stage,
        auto_curriculum: auto,
        evaluation_episodes: evalEpisodes,
      });
      setJob(await api<Job>("/training"));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  const control = async (action: string) => {
    setBusy(true);
    try {
      await api("/training/control", { action });
      setJob(await api<Job>("/training"));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  const evaluate = async (kind?: string) => {
    setBusy(true);
    setError("");
    setResults(null);
    try {
      setResults(
        await api<Results>(kind ? "/football-experiments" : "/matchups", {
          possession: picksP,
          defending: picksD,
          config: { ...config, environment_version: 2 },
          episodes: evalEpisodes,
          ...(kind ? { experiment: kind } : {}),
        }),
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  const toggle = (id: string, team: "P" | "D") => {
    const set = team === "P" ? setPicksP : setPicksD;
    set((old) =>
      old.includes(id)
        ? old.length > 1
          ? old.filter((v) => v !== id)
          : old
        : old.length < 4
          ? [...old, id]
          : old,
    );
  };
  const chooseGeneration = (value: string) => {
    setGeneration(value);
    if (value === "all") return;
    const gen = Number(value);
    for (const [list, set] of [
      [possession, setPicksP],
      [defending, setPicksD],
    ] as const) {
      const candidates = list
        .filter((p) => (p.lineage || p.id.split("/")[0]) === lineage)
        .filter((p) => (p.generation ?? 0) <= gen)
        .sort((a, b) => a.steps - b.steps);
      set(candidates.length ? [candidates.at(-1)!.id] : ["scripted"]);
    }
  };
  const current = detail || null;
  const poolEntries =
    current?.opponent_pool || selectedRun?.population?.entries || [];
  const last = history.at(-1);
  const teamGenerations = current?.team_generations ||
    last?.team_generation || { possession: 0, pressing: 0 };
  const rows = results?.matchups || results?.results || [];
  return (
    <div className="adversarial-lab">
      <div
        className="lab-tabs"
        role="tablist"
        aria-label="Adversarial lab views"
      >
        {(["Training", "Matchups", "Experiments"] as const).map((t) => (
          <button
            key={t}
            role="tab"
            aria-selected={tab === t}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      <section
        className="population-board"
        aria-label="Competing policy populations"
      >
        <div>
          <span className="section-label">Possession population</span>
          <strong>
            P{running && !current ? "?" : teamGenerations.possession || 0}
          </strong>
          <span>Six shared-policy actors</span>
          <b>
            {running && !current
              ? "Initializing worker"
              : running && current?.learning_team === "possession"
                ? "Learning · PPO update"
                : running
                  ? "Frozen opponent"
                  : "Recorded policy"}
          </b>
        </div>
        <div className="population-connection">
          <span>
            Generation{" "}
            {current?.generation ??
              (running ? "loading" : (last?.generation ?? 0))}
          </span>
          <svg
            viewBox="0 0 240 60"
            role="img"
            aria-label="Alternating policy updates against frozen opponents"
          >
            <path
              d="M5 18H230m-10-7 10 7-10 7M235 42H10m10-7-10 7 10 7"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            />
          </svg>
          <small>
            {running && !current
              ? "Worker is initializing policies and optimizers"
              : running
                ? `${current?.learning_team || "Worker starting"} learns against ${current?.sampled_opponent ? short(current.sampled_opponent.id) : "a frozen opponent"}`
                : "Alternating updates · immutable opponent snapshots"}
          </small>
        </div>
        <div>
          <span className="section-label">Defending population</span>
          <strong>
            D{running && !current ? "?" : teamGenerations.pressing || 0}
          </strong>
          <span>Two shared-policy actors</span>
          <b>
            {running && !current
              ? "Initializing worker"
              : running && current?.learning_team === "pressing"
                ? "Learning · PPO update"
                : running
                  ? "Frozen opponent"
                  : "Recorded policy"}
          </b>
        </div>
      </section>
      {tab === "Training" && (
        <div className="lab-training">
          <section className="form-panel">
            <h2>Train the competition</h2>
            <label>
              Training budget
              <select
                value={budget}
                onChange={(e) => {
                  const value = e.target.value;
                  setBudget(value);
                  const presets: Record<string, number[]> = {
                    smoke: [2048, 256, 2],
                    standard: [100000, 1024, 4],
                    extended: [1000000, 2048, 4],
                  };
                  const [s, b, n] = presets[value];
                  setSteps(s);
                  setBlock(b);
                  setEnvs(n);
                }}
              >
                <option value="smoke">Smoke · pipeline verification</option>
                <option value="standard">Standard · personal experiment</option>
                <option value="extended">Extended · longer populations</option>
              </select>
            </label>
            <label>
              Learning arrangement
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="both">Alternating adversarial teams</option>
                <option value="possession">
                  Possession against fixed defenders
                </option>
                <option value="pressing">
                  Defenders against fixed possession
                </option>
              </select>
            </label>
            <div className="lab-form-grid">
              <label>
                Total decision steps
                <input
                  type="number"
                  min="32"
                  max="10000000"
                  value={steps}
                  onChange={(e) => setSteps(Number(e.target.value))}
                />
              </label>
              <label>
                Steps per training block
                <input
                  type="number"
                  min="16"
                  max="8192"
                  value={block}
                  onChange={(e) => setBlock(Number(e.target.value))}
                />
              </label>
              <label>
                Batched environments
                <input
                  type="number"
                  min="1"
                  max="16"
                  value={envs}
                  onChange={(e) => setEnvs(Number(e.target.value))}
                />
              </label>
              <label>
                Training seed
                <input
                  type="number"
                  min="0"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                />
              </label>
            </div>
            <label>
              Football curriculum
              <select
                value={stage}
                onChange={(e) => setStage(Number(e.target.value))}
              >
                {stageNames.map((s, i) => (
                  <option key={s} value={i + 1}>
                    {i + 1} · {s}
                  </option>
                ))}
              </select>
            </label>
            <label className="check lab-inline">
              <input
                type="checkbox"
                checked={auto}
                onChange={(e) => {
                  setAuto(e.target.checked);
                  if (e.target.checked) setEvalEpisodes((n) => Math.max(10, n));
                }}
              />
              Promote using held-out football performance
            </label>
            <label>
              Evaluation scenarios per block
              <input
                type="number"
                min="1"
                max="30"
                value={evalEpisodes}
                onChange={(e) => setEvalEpisodes(Number(e.target.value))}
              />
            </label>
            <label>
              Reward configuration
              <select
                value={reward}
                onChange={(e) => setReward(e.target.value)}
              >
                <option value="sparse">Sparse · turnover objective</option>
                <option value="moderate">
                  Moderate · time and small pass reward
                </option>
                <option value="heavy">
                  Heavy · educational shaping comparison
                </option>
              </select>
            </label>
            <label>
              Compute device
              <select
                value={device}
                onChange={(e) => setDevice(e.target.value)}
              >
                <option value="cpu">CPU</option>
                <option value="auto">Automatic · CUDA / MPS / CPU</option>
                <option value="mps">Apple Silicon MPS</option>
                <option value="cuda">CUDA</option>
              </select>
            </label>
            <button
              className="primary"
              disabled={busy || running}
              onClick={() => void begin()}
            >
              Start training run
            </button>
            <p>
              One team updates at a time. Opponents mix latest, historical and
              scripted policies. Geometry and receiving delays create pressure;
              passing direction earns no bonus.
            </p>
            <h3>Resume a saved session</h3>
            <label>
              Compatible session
              <select
                value={resume}
                onChange={(e) => {
                  setResume(e.target.value);
                  const s = sessions.find((s) => s.id === e.target.value);
                  if (s) setSteps((n) => Math.max(n, s.steps + 2048));
                }}
              >
                <option value="">
                  {job.resume
                    ? "Use current run session"
                    : "Choose a v2 session"}
                </option>
                {sessions.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.id.split("/")[0]} · G{s.generation} ·{" "}
                    {s.steps.toLocaleString()} steps
                  </option>
                ))}
              </select>
            </label>
            <button
              disabled={busy || running || !(resume || job.resume)}
              onClick={() => void begin(true)}
            >
              Resume compatible session
            </button>
            <p>
              Restores both optimizers, RNG state, population and original
              settings. The total-step target must exceed the saved step count.
              Actor-only and v1 weights cannot silently resume v2 training.
            </p>
          </section>
          <section className="training-results">
            <div className="section-head">
              <h2>Live training worker</h2>
              <span className="status-badge">{job.status}</span>
            </div>
            {current ? (
              <>
                <progress max={current.target} value={current.steps} />
                <p>
                  {current.steps.toLocaleString()} /{" "}
                  {current.target.toLocaleString()} decisions · curriculum{" "}
                  {current.curriculum_stage}
                </p>
                <div className="data-row">
                  <span>Learning team</span>
                  <b>{current.learning_team}</b>
                </div>
                <div className="data-row">
                  <span>Frozen team</span>
                  <b>{current.frozen_team}</b>
                </div>
                <div className="data-row">
                  <span>Sampled opponent</span>
                  <b className="wrap-id">
                    {short(current.sampled_opponent.id)}
                  </b>
                </div>
                {current.latest && (
                  <>
                    <div className="data-row">
                      <span>Entropy / approximate KL</span>
                      <b>
                        {current.latest.entropy.toFixed(3)} /{" "}
                        {current.latest.approx_kl.toFixed(5)}
                      </b>
                    </div>
                    <h3>Latest fixed-benchmark evaluation</h3>
                    <p>
                      {current.latest.evaluation.mean_passes.toFixed(1)} mean
                      passes ·{" "}
                      {current.latest.evaluation.mean_duration.toFixed(1)}s
                      possession ·{" "}
                      {(current.latest.evaluation.success_rate * 100).toFixed(
                        0,
                      )}
                      % possession success
                    </p>
                    <h3>Reward components in the last rollout</h3>
                    {Object.entries(current.latest.reward_components).map(
                      ([k, v]) => (
                        <div className="data-row" key={k}>
                          <span>{k.replaceAll("_", " ")}</span>
                          <b>{v.toFixed(3)}</b>
                        </div>
                      ),
                    )}
                  </>
                )}
              </>
            ) : (
              <div className="empty">
                <h3>Ready for a measured experiment</h3>
                <p>
                  Run a smoke test to verify the pipeline. It does not establish
                  tactical improvement.
                </p>
              </div>
            )}
            <div className="file-controls">
              <button
                disabled={busy || job.status !== "running"}
                onClick={() => void control("paused")}
              >
                Pause & save session
              </button>
              <button
                disabled={busy || job.status !== "running"}
                onClick={() => void control("cancelled")}
              >
                Stop & save session
              </button>
            </div>
            {["pausing", "stopping"].includes(job.status) && (
              <p role="status">
                Finishing the current PPO block and saving a resumable session…
              </p>
            )}
            {job.error && <pre className="error">{job.error}</pre>}
            <h3>Opponent pool composition</h3>
            {poolEntries.length ? (
              <div className="pool-list">
                {poolEntries.map((entry) => (
                  <div className="pool-row" key={entry.id}>
                    <span>
                      {entry.team === "possession" ? "P" : "D"} · {entry.kind}
                    </span>
                    <strong>{short(entry.id)}</strong>
                    <small>
                      G{entry.generation}{" "}
                      {entry.steps !== undefined
                        ? `· ${entry.steps} steps`
                        : ""}
                    </small>
                  </div>
                ))}
              </div>
            ) : (
              <p>
                No active population yet. Scripted opponents and genuine initial
                weights enter the pool when training starts.
              </p>
            )}
            <p className="panel-note">
              Sampling weights: 40% latest, 35% historical, 25% scripted.
              Historical sampling retains weak and early policies and gives
              evaluated best snapshots extra probability. Checkpoint files are
              immutable.
            </p>
          </section>
        </div>
      )}
      {tab === "Training" && (
        <section className="lab-history">
          <div className="section-head">
            <h2>Recorded adaptation</h2>
            <label>
              Training run
              <select
                value={selectedRun?.id || ""}
                onChange={(e) => setRun(e.target.value)}
              >
                {runs.map((r) => (
                  <option key={r.id}>{r.id}</option>
                ))}
              </select>
            </label>
            <button
              onClick={() => void refresh().catch((e) => setError(String(e)))}
            >
              Refresh populations
            </button>
          </div>
          {history.length ? (
            <>
              <TeamChart history={history} metric="mean_step_reward" />
              <TeamChart history={history} metric="evaluation" />
              <p>
                Fixed benchmark curves use held-out seeds. Lower possession
                duration is better for defenders; higher is better for
                possession. Reward changes across opponents do not prove either
                team improved.
              </p>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Generation</th>
                      <th>Learning</th>
                      <th>Opponent</th>
                      <th>Steps</th>
                      <th>Fixed passes</th>
                      <th>Fixed duration</th>
                      <th>Historical duration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.slice(-20).map((r) => (
                      <tr key={r.steps}>
                        <td>{r.generation}</td>
                        <td>{r.team}</td>
                        <td>{short(r.opponent.id)}</td>
                        <td>{r.steps}</td>
                        <td>{r.evaluation.mean_passes.toFixed(1)}</td>
                        <td>{r.evaluation.mean_duration.toFixed(1)}s</td>
                        <td>
                          {r.historical_evaluation.mean_duration.toFixed(1)}s
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p>No adversarial history recorded yet.</p>
          )}
        </section>
      )}
      {tab !== "Training" && (
        <section className="matchup-panel">
          <div className="section-head">
            <h2>
              {tab === "Matchups"
                ? "Checkpoint versus checkpoint"
                : "Football experiments"}
            </h2>
            <span className="status-badge">Matched held-out seeds</span>
          </div>
          <div className="comparison-selectors">
            <label>
              Population run
              <select
                value={selectedRun?.id || ""}
                onChange={(e) => {
                  setRun(e.target.value);
                  setGeneration("all");
                }}
              >
                {runs.map((r) => (
                  <option key={r.id}>{r.id}</option>
                ))}
              </select>
            </label>
            <label>
              Recorded generation
              <select
                value={generation}
                onChange={(e) => chooseGeneration(e.target.value)}
              >
                <option value="all">All actual generations</option>
                {generations.map((g) => (
                  <option key={g} value={g}>
                    Generation {g}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Replay initial seed
              <input
                type="number"
                min="0"
                value={replaySeed}
                onChange={(e) => setReplaySeed(Number(e.target.value))}
              />
            </label>
            <label>
              Evaluation episodes
              <input
                type="number"
                min="1"
                max="20"
                value={Math.min(evalEpisodes, 20)}
                onChange={(e) => setEvalEpisodes(Number(e.target.value))}
              />
            </label>
          </div>
          <p>
            {generations.length
              ? `Recorded generations: ${generations.join(", ")}. ${[5, 10]
                  .filter((g) => !generations.includes(g))
                  .map((g) => `Generation ${g} has not been recorded.`)
                  .join(" ")}`
              : "No v2 neural checkpoints yet. Scripted versus scripted remains available."}{" "}
            Legacy weights remain accessible in Watch AI Learn.
          </p>
          <div className="policy-pickers">
            <PolicyPicker
              team="P"
              title="Possession policies"
              list={possession}
              picks={picksP}
              generation={generation}
              onToggle={toggle}
            />
            <PolicyPicker
              team="D"
              title="Defending policies"
              list={defending}
              picks={picksD}
              generation={generation}
              onToggle={toggle}
            />
          </div>
          <div className="file-controls">
            <button
              className="primary"
              onClick={() =>
                onWatch(
                  picksP[0] === "scripted" ? "" : picksP[0],
                  picksD[0] === "scripted" ? "" : picksD[0],
                  replaySeed,
                )
              }
            >
              Watch selected policy pair
            </button>
            {tab === "Matchups" && (
              <button disabled={busy} onClick={() => void evaluate()}>
                Evaluate matchup matrix
              </button>
            )}
          </div>
          {tab === "Experiments" && (
            <div className="experiment-controls">
              <label>
                Question to test
                <select
                  value={experiment}
                  onChange={(e) => setExperiment(e.target.value)}
                >
                  <option value="geometry">
                    A · Legacy circle versus compact grids
                  </option>
                  <option value="coordination">
                    B · Chase versus cover versus learned pressing
                  </option>
                  <option value="adaptation">
                    C · Possession against stronger opponents
                  </option>
                  <option value="evolution">
                    D · Adversarial generation matchups
                  </option>
                  <option value="rewards">
                    E · Reward-design policy comparison
                  </option>
                </select>
              </label>
              <button disabled={busy} onClick={() => void evaluate(experiment)}>
                Run measured experiment
              </button>
              <p>
                {experiment === "rewards"
                  ? "Select real policies from runs with different reward settings. This compares their weights under the same current evaluation conditions; it does not retrain or invent a reward ablation."
                  : experiment === "geometry"
                    ? "Reports the actual configuration of each legacy/grid condition. Defender speed remains 2.3m/s; ball and receiving mechanics also differ between versions."
                    : "Select the actual policies to compare. Every result comes from matched simulated episodes."}
              </p>
            </div>
          )}
          {busy && (
            <div className="empty" role="status">
              Evaluating actual simulator episodes…
            </div>
          )}
          {!busy && results && (
            <>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Pair / condition</th>
                      <th>Mean passes</th>
                      <th>Duration</th>
                      <th>Interceptions / episode</th>
                      <th>Pressure</th>
                      <th>Adjacent passes</th>
                      <th>Success</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r, i) => (
                      <tr key={i}>
                        <td>
                          {r.label ||
                            `${short(r.possession || "scripted")} vs ${short(r.defending || "scripted")}`}
                        </td>
                        <td>{(r.mean_passes ?? r.passes ?? 0).toFixed(2)}</td>
                        <td>
                          {(r.mean_duration ?? r.duration ?? 0).toFixed(2)}s
                        </td>
                        <td>
                          {(
                            r.interception_frequency ?? r.interceptions
                          )?.toFixed(1) ?? "n/a"}
                        </td>
                        <td>{r.average_pressure?.toFixed(3) ?? "n/a"}</td>
                        <td>
                          {r.adjacent_pass_fraction !== undefined
                            ? `${(r.adjacent_pass_fraction * 100).toFixed(0)}%`
                            : "n/a"}
                        </td>
                        <td>
                          {r.success_rate !== undefined
                            ? `${(r.success_rate * 100).toFixed(0)}%`
                            : "n/a"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p>
                Evaluation seeds: {results.seeds.join(", ")}. Possession success
                means surviving 15 seconds OR completing 10 passes. Continuous
                metrics remain visible. Outcomes across one training seed do not
                establish general improvement.
              </p>
              {results.note && <p>{results.note}</p>}
              {[...new Set(rows.flatMap((r) => r.flags || []))].map((flag) => (
                <p className="panel-note" key={flag}>
                  {flag}
                </p>
              ))}
              {rows
                .filter((r) => r.pass_direction_bins)
                .map((r, i) => (
                  <div className="direction-distribution" key={i}>
                    <h3>
                      {r.label || short(r.possession || "scripted")} · pass
                      directions
                    </h3>
                    <div>
                      {r.pass_direction_bins!.map((n, j) => (
                        <span
                          key={j}
                          style={{
                            height: `${8 + (n / Math.max(1, ...r.pass_direction_bins!)) * 65}px`,
                          }}
                          title={`${j * 45 - 180}° to ${j * 45 - 135}°: ${n} passes`}
                        >
                          <small>{n}</small>
                        </span>
                      ))}
                    </div>
                    <small>
                      Eight equal angle bins, -180° to 180°. Frequency is a
                      diagnostic, not a tactical reward.
                    </small>
                  </div>
                ))}
            </>
          )}
          {!results && !busy && (
            <p className="panel-note">
              Run an evaluation to populate this table. Missing generations and
              results are never synthesized.
            </p>
          )}
        </section>
      )}
    </div>
  );
}
function PolicyPicker({
  team,
  title,
  list,
  picks,
  generation,
  onToggle,
}: {
  team: "P" | "D";
  title: string;
  list: Policy[];
  picks: string[];
  generation: string;
  onToggle: (id: string, team: "P" | "D") => void;
}) {
  return (
    <fieldset>
      <legend>
        {title} <small>up to four</small>
      </legend>
      <label className="check">
        <input
          type="checkbox"
          checked={picks.includes("scripted")}
          onChange={() => onToggle("scripted", team)}
        />
        Scripted benchmark
      </label>
      {list
        .filter(
          (p) =>
            generation === "all" || (p.generation ?? 0) <= Number(generation),
        )
        .map((p) => (
          <label className="check" key={p.id}>
            <input
              type="checkbox"
              checked={picks.includes(p.id)}
              onChange={() => onToggle(p.id, team)}
            />
            <span>
              {policyName(p)}
              <small>
                Training reward: {p.config.reward} · {p.scenario}
              </small>
            </span>
          </label>
        ))}
      {!list.length && <p>No v2 checkpoints for this team.</p>}
    </fieldset>
  );
}
function TeamChart({
  history,
  metric,
}: {
  history: LabMetric[];
  metric: "mean_step_reward" | "evaluation";
}) {
  const value = (r: LabMetric) =>
    metric === "evaluation" ? r.evaluation.mean_duration : r.mean_step_reward;
  const values = history.map(value),
    lo = Math.min(0, ...values),
    hi = Math.max(0.01, ...values),
    range = hi - lo || 1,
    maxSteps = Math.max(1, ...history.map((r) => r.steps));
  return (
    <div className="team-chart">
      <h3>
        {metric === "evaluation"
          ? "Possession duration against fixed scripted opponents"
          : "Actual mean team reward per sampled decision"}
      </h3>
      <svg
        viewBox="0 0 900 230"
        role="img"
        aria-label={
          metric === "evaluation"
            ? "Fixed benchmark duration by training team"
            : "Recorded possession and defending reward curves"
        }
      >
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line
              x1="55"
              y1={190 - t * 160}
              x2="880"
              y2={190 - t * 160}
              stroke="#3b493c"
            />
            <text x="43" y={195 - t * 160} textAnchor="end">
              {(lo + t * range).toFixed(2)}
            </text>
          </g>
        ))}
        {["possession", "pressing"].map((team, i) => {
          const rows = history.filter((r) => r.team === team);
          return (
            <g key={team}>
              <polyline
                fill="none"
                stroke={i ? "#e9947c" : "#d8e9b0"}
                strokeWidth="2"
                points={rows
                  .map(
                    (r) =>
                      `${55 + (r.steps / maxSteps) * 825},${190 - ((value(r) - lo) / range) * 160}`,
                  )
                  .join(" ")}
              />
              {rows.map((r) => (
                <circle
                  key={r.steps}
                  r="3"
                  cx={55 + (r.steps / maxSteps) * 825}
                  cy={190 - ((value(r) - lo) / range) * 160}
                  fill={i ? "#e9947c" : "#d8e9b0"}
                >
                  <title>
                    {team} · G{r.generation} · {r.steps} steps:{" "}
                    {value(r).toFixed(3)}
                  </title>
                </circle>
              ))}
            </g>
          );
        })}
        <text x="55" y="220">
          0
        </text>
        <text x="880" y="220" textAnchor="end">
          {maxSteps.toLocaleString()} environment decisions
        </text>
      </svg>
      <div className="chart-keys">
        <span>Possession</span>
        <span>Defending</span>
      </div>
    </div>
  );
}
