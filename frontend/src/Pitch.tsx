import { useId } from "react";
import type { Frame, Replay, Vec } from "./types";
export const layerNames = {
  lanes: "Passing lanes",
  trajectory: "Ball trajectory",
  facing: "Body orientation",
  pressure: "Defensive pressure",
  velocity: "Velocity arrows",
  zones: "Interception reach",
  trails: "Movement trails",
  spacing: "Player spacing",
  options: "Available options",
  network: "Passing network",
  heatmap: "Movement heatmap",
  interceptions: "Interception map",
};
export type Layers = Record<keyof typeof layerNames, boolean>;
export const defaultLayers: Layers = {
  lanes: true,
  trajectory: true,
  facing: true,
  pressure: false,
  velocity: false,
  zones: false,
  trails: false,
  spacing: false,
  options: false,
  network: false,
  heatmap: false,
  interceptions: false,
};
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
const mixVec = (a: Vec, b: Vec, t: number): Vec => [
  mix(a[0], b[0], t),
  mix(a[1], b[1], t),
];
export function atTime(replay: Replay, time: number): Frame {
  const frames = replay.frames;
  let lo = 0,
    hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (frames[mid].time <= time) lo = mid;
    else hi = mid - 1;
  }
  const a = frames[lo],
    b = frames[Math.min(lo + 1, frames.length - 1)];
  const t =
    b.time === a.time
      ? 0
      : Math.max(0, Math.min(1, (time - a.time) / (b.time - a.time)));
  return {
    ...a,
    ball: { ...a.ball, position: mixVec(a.ball.position, b.ball.position, t) },
    players: a.players.map((p, i) => {
      const q = b.players[i];
      const turn = Math.atan2(
        Math.sin(q.facing - p.facing),
        Math.cos(q.facing - p.facing),
      );
      return {
        ...p,
        position: mixVec(p.position, q.position, t),
        facing: p.facing + turn * t,
        velocity: mixVec(p.velocity, q.velocity, t),
      };
    }),
  };
}
const distance = (a: Vec, b: Vec) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const laneDistance = (p: Vec, a: Vec, b: Vec) => {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  const t = Math.max(
    0,
    Math.min(
      1,
      ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1),
    ),
  );
  return distance(p, [a[0] + t * dx, a[1] + t * dy]);
};
export function Pitch({
  replay,
  time,
  layers,
  selected,
  onSelect,
  compact = false,
}: {
  replay: Replay;
  time: number;
  layers: Layers;
  selected: number | null;
  onSelect: (id: number) => void;
  compact?: boolean;
}) {
  const uid = useId().replace(/:/g, "");
  const frame = atTime(replay, time),
    w = Number(replay.config.width),
    h = Number(replay.config.height);
  const carrier = frame.ball.carrier,
    ball = frame.ball.position;
  const recent = replay.frames
    .filter((f) => f.time <= time && f.time > time - 2)
    .filter((_, i) => i % 3 === 0);
  const pastEvents = replay.events.filter((e) => e.time <= time);
  const network = new Map<string, number>();
  pastEvents
    .filter((e) => e.type === "reception" && e.origin !== undefined)
    .forEach((e) => {
      const key = [e.origin!, e.player!].sort().join("-");
      network.set(key, (network.get(key) || 0) + 1);
    });
  return (
    <svg
      className={"pitch " + (compact ? "compact" : "")}
      viewBox={`-3 -2.5 ${w + 6} ${h + 5}`}
      role="group"
      aria-label="Animated top-down 6 versus 2 rondo pitch"
    >
      <defs>
        <linearGradient id={`${uid}turf`} x2="0" y2="1">
          <stop stopColor="#234c3b" />
          <stop offset="1" stopColor="#193e31" />
        </linearGradient>
        <radialGradient id={`${uid}pressure`}>
          <stop stopColor="#ed735f" stopOpacity=".25" />
          <stop offset="1" stopColor="#ed735f" stopOpacity="0" />
        </radialGradient>
        <filter
          id={`${uid}shadow`}
          x="-60%"
          y="-60%"
          width="220%"
          height="240%"
        >
          <feDropShadow
            dx="0"
            dy=".09"
            stdDeviation=".07"
            floodColor="#081d13"
            floodOpacity=".65"
          />
        </filter>
        <marker
          id={`${uid}arrow`}
          markerWidth="5"
          markerHeight="5"
          refX="4"
          refY="2.5"
          orient="auto"
        >
          <path
            d="M0,0 L5,2.5 L0,5"
            fill="none"
            stroke="#e3e9d2"
            strokeWidth=".8"
          />
        </marker>
      </defs>
      <rect
        x="-3"
        y="-2.5"
        width={w + 6}
        height={h + 5}
        rx=".18"
        fill={`url(#${uid}turf)`}
      />
      {Array.from({ length: 10 }, (_, i) => (
        <rect
          key={i}
          x={-3 + ((w + 6) / 10) * i}
          y="-2.5"
          width={(w + 6) / 10}
          height={h + 5}
          fill={i % 2 ? "#ffffff" : "#000000"}
          opacity=".025"
        />
      ))}
      <g stroke="#bad3bd" strokeWidth=".035" fill="none" opacity=".48">
        <rect x="-2.2" y="-1.6" width={w + 4.4} height={h + 3.2} />
        <path d={`M${w / 2},-1.6 V${h + 1.6}`} />
        <circle cx={w / 2} cy={h / 2} r="2.2" />
        <circle cx={w / 2} cy={h / 2} r=".06" fill="#bad3bd" />
        <path
          d={`M-2.2,${h / 2 - 3.4} H.1 V${h / 2 + 3.4} H-2.2 M${w + 2.2},${h / 2 - 3.4} H${w - 0.1} V${h / 2 + 3.4} H${w + 2.2}`}
        />
      </g>
      <rect
        x="0"
        y="0"
        width={w}
        height={h}
        fill="#8ab87a"
        fillOpacity=".025"
        stroke="#d4dfb5"
        strokeWidth=".045"
        strokeDasharray=".16 .16"
        opacity=".65"
      />
      {[
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
      ].map(([x, y], i) => (
        <g key={i} transform={`translate(${x} ${y})`}>
          <ellipse cy=".05" rx=".14" ry=".075" fill="#11271f" />
          <path d="M-.11,.04 L0,-.16 L.11,.04 Z" fill="#e6bb79" />
        </g>
      ))}
      <text x="0" y="-.45" className="pitch-label">
        {w} × {h} m · training area
      </text>
      <text x={w} y={h + 0.75} textAnchor="end" className="pitch-label">
        6 possession / 2 pressing
      </text>
      {layers.heatmap &&
        replay.frames
          .filter((f) => f.time <= time)
          .filter((_, i) => i % 12 === 0)
          .map((f, i) => (
            <g key={i}>
              {f.players.map((p) => (
                <circle
                  key={p.id}
                  cx={p.position[0]}
                  cy={p.position[1]}
                  r=".32"
                  fill={p.id < 6 ? "#c4d99d" : "#f18875"}
                  opacity=".035"
                />
              ))}
            </g>
          ))}
      {layers.network &&
        [...network].map(([key, n]) => {
          const [a, b] = key.split("-").map(Number);
          return (
            <line
              key={key}
              x1={frame.players[a].position[0]}
              y1={frame.players[a].position[1]}
              x2={frame.players[b].position[0]}
              y2={frame.players[b].position[1]}
              stroke="#d7e8ae"
              opacity=".5"
              strokeWidth={0.025 + Math.min(n, 10) * 0.02}
            />
          );
        })}
      {layers.spacing &&
        frame.players.slice(0, 6).map((p, i) => {
          const q = frame.players[(i + 1) % 6];
          return (
            <g key={i}>
              <line
                x1={p.position[0]}
                y1={p.position[1]}
                x2={q.position[0]}
                y2={q.position[1]}
                stroke="#c5cbb9"
                strokeWidth=".025"
                strokeDasharray=".1 .14"
              />
              <text
                x={(p.position[0] + q.position[0]) / 2}
                y={(p.position[1] + q.position[1]) / 2}
                className="distance-label"
              >
                {distance(p.position, q.position).toFixed(1)}m
              </text>
            </g>
          );
        })}
      {layers.pressure &&
        frame.players
          .slice(6)
          .map((p) => (
            <circle
              key={p.id}
              cx={p.position[0]}
              cy={p.position[1]}
              r="2.5"
              fill={`url(#${uid}pressure)`}
            />
          ))}
      {layers.zones &&
        frame.players
          .slice(6)
          .map((p) => (
            <circle
              key={p.id}
              cx={p.position[0]}
              cy={p.position[1]}
              r={Number(replay.config.interception_radius)}
              fill="#f18470"
              fillOpacity=".12"
              stroke="#f4a08f"
              strokeWidth=".025"
              strokeDasharray=".07 .09"
            />
          ))}
      {(layers.lanes || layers.options) &&
        carrier !== null &&
        carrier < 6 &&
        frame.players
          .slice(0, 6)
          .filter((p) => p.id !== carrier)
          .map((p) => {
            const clearance = Math.min(
              ...frame.players
                .slice(6)
                .map((d) => laneDistance(d.position, ball, p.position)),
            );
            const blocked =
              clearance < Number(replay.config.interception_radius) + 0.2;
            return (
              <g key={p.id}>
                <line
                  x1={ball[0]}
                  y1={ball[1]}
                  x2={p.position[0]}
                  y2={p.position[1]}
                  stroke={blocked ? "#f09985" : "#d3dfb3"}
                  strokeWidth=".035"
                  strokeDasharray={blocked ? ".08 .16" : ".18 .18"}
                  opacity={blocked ? 0.65 : 0.38}
                />
                {layers.options && (
                  <text
                    x={p.position[0]}
                    y={p.position[1] - 0.8}
                    textAnchor="middle"
                    className="distance-label"
                  >
                    {blocked ? "blocked" : "open"}
                  </text>
                )}
              </g>
            );
          })}
      {layers.trails &&
        frame.players.map((p) => (
          <polyline
            key={p.id}
            points={recent
              .map((f) => f.players[p.id].position.join(","))
              .join(" ")}
            stroke={p.id < 6 ? "#e8e6d6" : "#ec8c76"}
            strokeWidth=".055"
            fill="none"
            opacity=".3"
          />
        ))}
      {layers.trajectory && (
        <polyline
          points={recent.map((f) => f.ball.position.join(",")).join(" ")}
          stroke="#f6e7b0"
          strokeWidth=".05"
          fill="none"
          opacity=".7"
          strokeLinecap="round"
        />
      )}
      {layers.trajectory && frame.ball.in_flight && frame.ball.origin && (
        <line
          x1={frame.ball.origin[0]}
          y1={frame.ball.origin[1]}
          x2={ball[0]}
          y2={ball[1]}
          stroke="#fff2b6"
          strokeWidth=".04"
          opacity=".65"
        />
      )}
      {layers.interceptions &&
        pastEvents
          .filter((e) => e.type === "interception" || e.type === "tackle")
          .map((e, i) => (
            <g
              key={i}
              transform={`translate(${e.position![0]} ${e.position![1]})`}
              stroke="#ffad91"
              strokeWidth=".06"
            >
              <path d="M-.15,-.15 L.15,.15 M-.15,.15 L.15,-.15" />
            </g>
          ))}
      {frame.players.map((p) => {
        const [x, y] = p.position,
          speed = Math.hypot(...p.velocity),
          selectedPlayer = selected === p.id;
        const stride =
          Math.sin(time * 14 + p.id) * Math.min(speed / 3, 1) * 0.075;
        return (
          <g
            key={p.id}
            transform={`translate(${x} ${y})`}
            role="button"
            tabIndex={0}
            aria-label={`Inspect ${p.id < 6 ? "possession player" : "defender"} ${p.id + 1}`}
            onClick={() => onSelect(p.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect(p.id);
              }
            }}
            className="player"
          >
            <circle className="player-hit" r=".8" fill="transparent" />
            {carrier === p.id && (
              <circle
                r=".55"
                fill="none"
                stroke="#d9e9ad"
                strokeWidth=".035"
                opacity=".8"
              />
            )}
            {selectedPlayer && (
              <circle
                r=".66"
                fill="none"
                stroke="#f5ecc4"
                strokeWidth=".045"
                strokeDasharray=".12 .09"
              />
            )}
            {layers.velocity && speed > 0.1 && (
              <line
                x1="0"
                y1="0"
                x2={p.velocity[0] * 0.5}
                y2={p.velocity[1] * 0.5}
                stroke="#e3e9d2"
                strokeWidth=".04"
                markerEnd={`url(#${uid}arrow)`}
              />
            )}
            <g
              transform={`rotate(${(p.facing * 180) / Math.PI})`}
              filter={`url(#${uid}shadow)`}
            >
              <ellipse
                cx={-0.19 + stride}
                cy="-.13"
                rx=".11"
                ry=".07"
                fill="#101f1a"
              />
              <ellipse
                cx={-0.19 - stride}
                cy=".13"
                rx=".11"
                ry=".07"
                fill="#101f1a"
              />
              <path
                d="M-.24,-.18 Q-.24,-.35 -.08,-.36 L.12,-.25 Q.28,-.15 .28,0 Q.28,.15 .12,.25 L-.08,.36 Q-.24,.35 -.24,.18 Z"
                fill={p.id < 6 ? "#efeedd" : "#b84b3c"}
                stroke={p.id < 6 ? "#ffffff" : "#f2a18c"}
                strokeWidth=".025"
              />
              <circle cx=".22" cy="0" r=".12" fill="#b99b78" />
              {layers.facing && (
                <path
                  d="M.4,-.09 L.57,0 L.4,.09"
                  fill="none"
                  stroke="#f1e7bf"
                  strokeWidth=".04"
                />
              )}
            </g>
            <text
              y=".09"
              x="-.05"
              textAnchor="middle"
              className="shirt-number"
              fill={p.id < 6 ? "#1e372e" : "#fff5e8"}
            >
              {p.id + 1}
            </text>
            {!compact && (
              <text y=".9" textAnchor="middle" className="player-label">
                {p.id < 6 ? `P${p.id + 1}` : `D${p.id - 5}`}
              </text>
            )}
          </g>
        );
      })}
      <g
        transform={`translate(${ball[0]} ${ball[1]})`}
        className="ball"
        filter={`url(#${uid}shadow)`}
      >
        <circle r=".135" fill="#faf8e9" stroke="#203a2d" strokeWidth=".018" />
        <path
          d="M0,-.07 L.067,-.02 L.042,.06 L-.042,.06 L-.067,-.02 Z"
          fill="#28392f"
          transform={`rotate(${time * Number(Math.hypot(...frame.ball.velocity)) * 40})`}
        />
      </g>
    </svg>
  );
}
