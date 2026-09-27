import { APP_DATA } from "./data";
import type { AppSnapshot } from "./types";
import type { TrackedBet } from "./components/TrackedBetsContext";

export type BetResult = "win" | "loss" | "push" | "pending";

export type SettledTrackedBet = TrackedBet & {
  result: BetResult;
  profitUnits: number | null;
  actualMargin: number | null;
  actualTotal: number | null;
  awayScore: number | null;
  homeScore: number | null;
};

const SNAPSHOT = APP_DATA as AppSnapshot;

const gameResult = (gameId: string) => {
  const direct = SNAPSHOT.gameResults?.find((x) => x.gameId === gameId);
  if (direct) return direct;

  return APP_DATA.forwardPicks.find(
    (x) =>
      x.gameId === gameId &&
      (x.actualMargin != null || x.actualTotal != null)
  );
};

export const profitForOdds = (odds: number, stakeUnits = 1) => {
  if (odds > 0) return stakeUnits * (odds / 100);
  if (odds < 0) return stakeUnits * (100 / Math.abs(odds));
  return 0;
};

export const settleTrackedBet = (bet: TrackedBet): SettledTrackedBet => {
  const r = gameResult(bet.gameId);

  if (!r) {
    return {
      ...bet,
      result: "pending",
      profitUnits: null,
      actualMargin: null,
      actualTotal: null,
      awayScore: null,
      homeScore: null,
    };
  }

  let result: BetResult = "pending";

  if (bet.market === "total" && r.actualTotal != null) {
    const side = bet.pick.toUpperCase();
    if (r.actualTotal === bet.line) {
      result = "push";
    } else if (side === "UNDER") {
      result = r.actualTotal < bet.line ? "win" : "loss";
    } else if (side === "OVER") {
      result = r.actualTotal > bet.line ? "win" : "loss";
    }
  }

  if (bet.market === "spread" && r.actualMargin != null) {
    let pickedMargin: number | null = null;
    if (bet.pick === bet.home) pickedMargin = r.actualMargin;
    else if (bet.pick === bet.away) pickedMargin = -r.actualMargin;

    if (pickedMargin != null) {
      const cover = pickedMargin + bet.line;
      if (Math.abs(cover) < 1e-9) result = "push";
      else result = cover > 0 ? "win" : "loss";
    }
  }

  if (bet.market === "moneyline" && r.actualMargin != null) {
    let pickedMargin: number | null = null;
    if (bet.pick === bet.home) pickedMargin = r.actualMargin;
    else if (bet.pick === bet.away) pickedMargin = -r.actualMargin;

    if (pickedMargin != null) {
      if (Math.abs(pickedMargin) < 1e-9) result = "push";
      else result = pickedMargin > 0 ? "win" : "loss";
    }
  }

  const profitUnits =
    result === "win"
      ? profitForOdds(bet.odds, bet.stakeUnits)
      : result === "loss"
        ? -bet.stakeUnits
        : result === "push"
          ? 0
          : null;

  return {
    ...bet,
    result,
    profitUnits,
    actualMargin: r.actualMargin,
    actualTotal: r.actualTotal,
    awayScore: r.awayScore,
    homeScore: r.homeScore,
  };
};

export const trackedSummary = (bets: TrackedBet[]) => {
  const settled = bets.map(settleTrackedBet);
  const wins = settled.filter((x) => x.result === "win").length;
  const losses = settled.filter((x) => x.result === "loss").length;
  const pushes = settled.filter((x) => x.result === "push").length;
  const pending = settled.filter((x) => x.result === "pending").length;
  const units = settled.reduce((sum, x) => sum + (x.profitUnits ?? 0), 0);
  const risked = settled
    .filter((x) => x.result !== "pending")
    .reduce((sum, x) => sum + x.stakeUnits, 0);

  return {
    settled,
    wins,
    losses,
    pushes,
    pending,
    units,
    risked,
    roi: risked > 0 ? units / risked : 0,
  };
};
