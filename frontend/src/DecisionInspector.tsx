import type { Frame, Player, Replay } from "./types";
import { labels, defenderLabels } from "./types";
export default function DecisionInspector({
  replay,
  frame,
  player,
}: {
  replay: Replay;
  frame: Frame;
  player: Player;
}) {
  const decision = replay.decisions.filter((d) => d.time <= frame.time).at(-1);
  const observation = decision?.observations[String(player.id)],
    mask = decision?.masks[String(player.id)];
  const previous = replay.frames
    .filter((f) => f.time <= (decision?.time ?? 0))
    .at(-1);
  const reward = previous?.reward_components
    ? Object.values(previous.reward_components).reduce((a, b) => a + b, 0) *
      (player.id < 6 ? 1 : -1)
    : null;
  const actionNames =
    player.id < 6
      ? replay.config.environment_version === 1
        ? labels.map((s, i) =>
            i === 7 ? "Clockwise" : i === 8 ? "Counterclockwise" : s,
          )
        : labels
      : defenderLabels;
  return (
    <section className="decision-inspector">
      <h3>Recorded decision input</h3>
      {observation ? (
        <>
          <p>
            Decision at {(decision?.time ?? 0).toFixed(2)}s ·{" "}
            {observation.length} normalized features. Future opponent actions
            are excluded.
          </p>
          <div className="data-row">
            <span>Own normalized position</span>
            <b>
              {observation
                .slice(0, 2)
                .map((v) => v.toFixed(3))
                .join(", ")}
            </b>
          </div>
          <div className="data-row">
            <span>Relative ball position</span>
            <b>
              {observation
                .slice(6, 8)
                .map((v) => v.toFixed(3))
                .join(", ")}
            </b>
          </div>
        </>
      ) : (
        <p>Observation recording is unavailable in this legacy file.</p>
      )}
      <div className="data-row">
        <span>
          {player.id < 6
            ? "Current proximity pressure"
            : "Team pressure at ball"}
        </span>
        <b>{player.pressure?.toFixed(3) ?? "Unavailable"}</b>
      </div>
      <div className="data-row">
        <span>Ball distance</span>
        <b>
          {player.ball_distance?.toFixed(2) ??
            Math.hypot(
              player.position[0] - frame.ball.position[0],
              player.position[1] - frame.ball.position[1],
            ).toFixed(2)}{" "}
          m
        </b>
      </div>
      {player.id < 6 ? (
        <>
          <div className="data-row">
            <span>Receiving preparation left</span>
            <b>{player.receiving_remaining?.toFixed(2) ?? "Unavailable"} s</b>
          </div>
          <h3>Available passing targets</h3>
          <p>
            {mask
              ? mask
                  .slice(1, 7)
                  .map((v, i) => (v ? `P${i + 1}` : null))
                  .filter(Boolean)
                  .join(", ") || "No pass available at this decision."
              : "Mask unavailable."}
          </p>
        </>
      ) : (
        <>
          <div className="data-row">
            <span>Targeted lane</span>
            <b>
              {player.action >= 2 && player.action <= 7
                ? `Ball to P${player.action - 1}`
                : "No selected receiver lane"}
            </b>
          </div>
          <div className="data-row">
            <span>Controller target</span>
            <b>
              {player.target?.map((n) => n.toFixed(2)).join(", ") ??
                "Unavailable"}
            </b>
          </div>
          <h3>Last feasible interception attempt</h3>
          <p>
            {player.last_interception_attempt
              ? `${player.last_interception_attempt.time.toFixed(2)}s · target (${player.last_interception_attempt.point.map((n) => n.toFixed(2)).join(", ")}) · ${player.last_interception_attempt.eta.toFixed(2)}s predicted arrival. An attempt does not guarantee contact.`
              : "No feasible interception attempt recorded yet."}
          </p>
        </>
      )}
      <div className="data-row">
        <span>Previous action</span>
        <b>
          {player.previous_action !== undefined
            ? actionNames[player.previous_action]
            : "Unavailable"}
        </b>
      </div>
      <div className="data-row">
        <span>Previous completed interval reward</span>
        <b>{reward !== null ? reward.toFixed(3) : "Unavailable"}</b>
      </div>
    </section>
  );
}
