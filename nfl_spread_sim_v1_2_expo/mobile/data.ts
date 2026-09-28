import { APP_DATA } from "./generatedData";
import type { AppSnapshot, ForwardPick, GameView, MarketPick, MarketType } from "./types";

export { APP_DATA };
export type { ForwardPick, GameView, MarketPick, MarketType };

export const pct = (x: number | null | undefined, digits = 1) =>
  x == null || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(digits)}%`;

export const signed = (x: number | null | undefined, digits = 1) =>
  x == null || Number.isNaN(x)
    ? "—"
    : `${x >= 0 ? "+" : ""}${x.toFixed(digits)}`;

export const oddsText = (x: number | null | undefined) =>
  x == null || Number.isNaN(x) ? "—" : `${x > 0 ? "+" : ""}${x.toFixed(0)}`;

export const marketText = (market: MarketType) => {
  if (market === "moneyline") return "MONEYLINE";
  if (market === "spread") return "SPREAD";
  return "TOTAL";
};

export const lineText = (
  market: MarketType,
  pick: string,
  line: number
) => {
  if (market === "moneyline") return pick.toUpperCase();
  if (market === "spread") return `${pick} ${signed(line)}`;
  return `${pick.toUpperCase()} ${line.toFixed(1)}`;
};

export const getGame = (id: string): GameView | undefined =>
  APP_DATA.games.find((g) => g.id === id) as GameView | undefined;

const SNAPSHOT = APP_DATA as AppSnapshot;

export const currentWeek = SNAPSHOT.currentWeek ?? SNAPSHOT.week;

export const availableWeeks = () => {
  const configured = SNAPSHOT.availableWeeks ?? [];
  if (configured.length) return [...configured].sort((a, b) => a - b);
  return [...new Set(APP_DATA.games.map((g) => g.week))].sort((a, b) => a - b);
};

export const hasStarted = (g: GameView, nowMs = Date.now()) => {
  if (!g.kickoffAt) return false;
  return new Date(g.kickoffAt).getTime() <= nowMs;
};

export const gamesForWeek = (week: number) =>
  APP_DATA.games.filter((g) => g.week === week);

export const upcomingGames = (nowMs = Date.now(), week?: number) =>
  APP_DATA.games.filter(
    (g) => (week == null || g.week === week) && !hasStarted(g, nowMs)
  );

const allPicks = (g: GameView) =>
  [g.spread, g.total, g.moneyline].filter((p): p is MarketPick => Boolean(p));

export const upcomingQualifiedPicks = (nowMs = Date.now(), week?: number) =>
  upcomingGames(nowMs, week)
    .flatMap((g) =>
      allPicks(g)
        .filter((p) => p.qualifies)
        .map((pick) => ({ game: g, pick }))
    )
    .sort((a, b) => (b.pick.edge ?? -999) - (a.pick.edge ?? -999));

export const bestQualifiedPick = (g: GameView) =>
  allPicks(g)
    .filter((p) => p.qualifies)
    .sort((a, b) => (b.edge ?? -999) - (a.edge ?? -999))[0] ?? null;

export const bestAngle = (g: GameView) =>
  allPicks(g).sort((a, b) => (b.edge ?? -999) - (a.edge ?? -999))[0] ?? null;

export const projectedMoneylineWinner = (g: GameView) => {
  if (g.modelMargin == null || Number.isNaN(g.modelMargin)) return null;
  if (Math.abs(g.modelMargin) < 0.01) {
    return { team: null, label: "Projected toss-up", margin: 0 };
  }
  const homeFavored = g.modelMargin > 0;
  const team = homeFavored ? g.home : g.away;
  return {
    team,
    label: `${team} projected winner`,
    margin: Math.abs(g.modelMargin),
  };
};

export const projectedScore = (g: GameView) => {
  if (g.modelMargin == null || g.modelTotal == null) return null;
  return {
    away: (g.modelTotal - g.modelMargin) / 2,
    home: (g.modelTotal + g.modelMargin) / 2,
  };
};

export const kickoffText = (g: GameView) => {
  if (!g.kickoffAt) return "Kickoff TBD";
  const d = new Date(g.kickoffAt);
  return d.toLocaleString("en-US", {
    weekday: "short",
    hour: "numeric",
    minute: "2-digit",
  });
};

const ESPN: Record<string, string> = {
  ARI: "ari", ATL: "atl", BAL: "bal", BUF: "buf", CAR: "car", CHI: "chi",
  CIN: "cin", CLE: "cle", DAL: "dal", DEN: "den", DET: "det", GB: "gb",
  HOU: "hou", IND: "ind", JAX: "jax", JAC: "jax", KC: "kc", LV: "lv", LAC: "lac",
  LA: "lar", MIA: "mia", MIN: "min", NE: "ne", NO: "no", NYG: "nyg",
  NYJ: "nyj", PHI: "phi", PIT: "pit", SEA: "sea", SF: "sf", TB: "tb",
  TEN: "ten", WAS: "wsh",
};

export const teamLogo = (team: string) => {
  const key = ESPN[team] ?? team.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/nfl/500/${key}.png`;
};

export const shortGame = (id: string) => {
  const p = id.split("_");
  return p.length >= 4 ? `${p[2]} @ ${p[3]}` : id;
};

export const resultColor = (result: string | null | undefined) => {
  const r = result?.toLowerCase();
  if (r === "win") return "win";
  if (r === "loss") return "loss";
  if (r === "push") return "push";
  return "pending";
};
