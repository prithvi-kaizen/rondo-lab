export type Vec = [number, number];
export interface Player {
  id: number;
  team: string;
  position: Vec;
  velocity: Vec;
  facing: number;
  angular_velocity: number;
  action: number;
  intention: string;
  control: string;
  probabilities: number[] | null;
  value: number | null;
  previous_action?: number;
  target?: Vec;
  pressure?: number;
  ball_distance?: number;
  receiving_remaining?: number;
  region?: [Vec, Vec];
  last_interception_attempt?: { time: number; point: Vec; eta: number } | null;
  intercept_path?: { point: Vec; eta: number; feasible: boolean };
}
export interface Ball {
  position: Vec;
  velocity: Vec;
  carrier: number | null;
  previous_carrier: number | null;
  target: number | null;
  origin: Vec | null;
  in_flight: boolean;
}
export interface Frame {
  time: number;
  players: Player[];
  ball: Ball;
  passes: number;
  attempts: number;
  hold: number;
  reward_components?: Record<string, number>;
}
export interface Event {
  type: string;
  time: number;
  player?: number;
  target?: number;
  origin?: number;
  position?: Vec;
}
export type Config = Record<string, number | string>;
export interface Policy {
  id: string;
  team: string;
  steps: number;
  stage: string;
  scenario: string;
  mode: string;
  seed: number;
  config: Config;
  note?: string;
  generation?: number;
  team_generation?: number;
  environment_version?: number;
  compatibility?: string;
  lineage?: string;
}
export interface Replay {
  schema_version: 1 | 2;
  episode_id: string;
  seed: number;
  config: Config;
  policy: Partial<Policy> & { kind?: string; name?: string };
  opponent?: Partial<Policy> | null;
  frames: Frame[];
  events: Event[];
  metrics: Record<string, number | string | number[]>;
  decisions: {
    time: number;
    actions: Record<string, number>;
    observations: Record<string, number[]>;
    masks: Record<string, number[]>;
  }[];
}
export interface Evaluation {
  mean_passes: number;
  pass_completion: number;
  mean_duration: number;
  interceptions: number;
  passes_ci95_normal: number;
  episodes: number;
}
export interface Metric {
  steps: number;
  team: string;
  mean_step_reward: number;
  policy_loss: number;
  value_loss: number;
  entropy: number;
  approx_kl: number;
  explained_variance: number | null;
  evaluation: Evaluation;
}
export interface Experiment {
  id: string;
  metadata: Record<string, unknown>;
  history: Metric[];
  population?: {
    entries: {
      id: string;
      team: string;
      kind: string;
      generation: number;
      strategy?: string;
      steps?: number;
    }[];
  };
}
export interface TrainingStatus {
  id?: string;
  status: string;
  error?: string;
  progress?: {
    steps: number;
    target: number;
    updates: number;
    episodes: number;
    latest: Metric;
  } | null;
}
export const labels = [
  "Hold",
  "Pass to 1",
  "Pass to 2",
  "Pass to 3",
  "Pass to 4",
  "Pass to 5",
  "Pass to 6",
  "Adjust left in region",
  "Adjust right in region",
  "Support angle",
  "Avoid pressure",
  "Recover anchor",
];
export const defenderLabels = [
  "Press",
  "Cover",
  "Block 1",
  "Block 2",
  "Block 3",
  "Block 4",
  "Block 5",
  "Block 6",
  "Recover",
  "Pursue ball",
  "Hold shape",
  "Close receiver",
];
export async function api<T>(
  path: string,
  body?: unknown,
  method?: string,
): Promise<T> {
  const r = await fetch("/api" + path, {
    method: method || (body ? "POST" : "GET"),
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    const error = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : JSON.stringify(error.detail),
    );
  }
  return r.json();
}
