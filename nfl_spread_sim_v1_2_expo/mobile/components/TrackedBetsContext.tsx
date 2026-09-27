import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  createContext,
  PropsWithChildren,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

import type { MarketType } from "../types";

export type TrackedBet = {
  id: string;
  gameId: string;
  away: string;
  home: string;
  kickoffAt: string | null;
  market: MarketType;
  pick: string;
  line: number;
  odds: number;
  book: string;
  edge: number | null;
  selectedProbability: number | null;
  breakEven: number | null;
  stakeUnits: number;
  placedAt: string;
};

export type TrackedBetInput = Omit<TrackedBet, "id" | "placedAt">;

type TrackerValue = {
  trackedBets: TrackedBet[];
  ready: boolean;
  logBet: (bet: TrackedBetInput) => Promise<void>;
  removeBet: (id: string) => Promise<void>;
  isTracked: (bet: TrackedBetInput) => boolean;
};

const STORAGE_KEY = "nfl-sim-my-actual-bets-v1";
const Context = createContext<TrackerValue | null>(null);

export const trackedBetId = (bet: TrackedBetInput) =>
  [
    bet.gameId,
    bet.market,
    bet.pick,
    Number(bet.line).toFixed(2),
    Number(bet.odds).toFixed(0),
    bet.book,
  ].join("|");

export function TrackedBetsProvider({ children }: PropsWithChildren) {
  const [trackedBets, setTrackedBets] = useState<TrackedBet[]>([]);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let alive = true;

    AsyncStorage.getItem(STORAGE_KEY)
      .then((raw) => {
        if (!alive || !raw) return;
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) setTrackedBets(parsed);
      })
      .catch(() => {})
      .finally(() => {
        if (alive) setReady(true);
      });

    return () => {
      alive = false;
    };
  }, []);

  const persist = async (next: TrackedBet[]) => {
    setTrackedBets(next);
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  };

  const logBet = async (bet: TrackedBetInput) => {
    const id = trackedBetId(bet);
    if (trackedBets.some((x) => x.id === id)) return;

    await persist([
      {
        ...bet,
        id,
        placedAt: new Date().toISOString(),
      },
      ...trackedBets,
    ]);
  };

  const removeBet = async (id: string) => {
    await persist(trackedBets.filter((x) => x.id !== id));
  };

  const isTracked = (bet: TrackedBetInput) =>
    trackedBets.some((x) => x.id === trackedBetId(bet));

  const value = useMemo(
    () => ({ trackedBets, ready, logBet, removeBet, isTracked }),
    [trackedBets, ready]
  );

  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export function useTrackedBets() {
  const value = useContext(Context);
  if (!value) {
    throw new Error("useTrackedBets must be used inside TrackedBetsProvider");
  }
  return value;
}
