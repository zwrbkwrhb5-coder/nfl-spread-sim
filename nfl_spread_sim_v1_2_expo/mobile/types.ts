export type MarketType = "spread" | "total" | "moneyline";

export type QuoteOption = {
  line: number;
  odds: number;
  book: string;
  breakEven: number | null;
  edge: number | null;
  quoteAgeMinutes: number | null;
  selected: boolean;
};

export type MarketPick = {
  market: MarketType;
  pick: string;
  line: number;
  odds: number;
  book: string;
  rawProbability: number | null;
  calibratedProbability: number | null;
  selectedProbability: number | null;
  breakEven: number | null;
  edge: number | null;
  minimumEdge: number | null;
  qualifies: boolean;
  quoteAgeMinutes: number | null;
  oddsBoard: QuoteOption[];
};

export type ShadowTotal = {
  currentConsensusTotal: number;
  shadowTotal: number;
  gap: number;
  direction: "OVER" | "UNDER" | "PASS";
  booksInConsensus: number;
  timingMatchedToTraining: boolean;
  shadowOnly: boolean;
};

export type GameView = {
  id: string;
  season: number;
  week: number;
  away: string;
  home: string;
  kickoffAt: string | null;
  awayQB: string | null;
  homeQB: string | null;
  modelMargin: number | null;
  modelTotal: number | null;
  spread: MarketPick | null;
  total: MarketPick | null;
  moneyline?: MarketPick | null;
  shadowTotal: ShadowTotal | null;
  qualifies: boolean;
  bestEdge: number | null;
};


export type GameResult = {
  gameId: string;
  actualMargin: number | null;
  actualTotal: number | null;
  awayScore: number | null;
  homeScore: number | null;
};

export type ForwardPick = {
  gameId: string;
  market: string;
  pick: string;
  line: number;
  odds: number;
  book: string;
  edge: number | null;
  selectedProbability: number | null;
  breakEven: number | null;
  placed: boolean;
  stakeUnits: number | null;
  closeConsensusLine: number | null;
  closeBestLine: number | null;
  clvPointsVsConsensus: number | null;
  clvPointsVsBest: number | null;
  result: string | null;
  settlementProfitUnits: number | null;
  status: string;
  closeNote: string | null;
  actualMargin: number | null;
  actualTotal: number | null;
  awayScore: number | null;
  homeScore: number | null;
};

export type HistoricalMetric = {
  market: MarketType | "other";
  games: number;
  marketMae: number;
  marketCalibratedMae: number;
  baseModelMae: number;
  stackMae: number;
  marketRmse: number;
  marketCalibratedRmse: number;
  baseModelRmse: number;
  stackRmse: number;
};

export type PerformanceSummary = {
  qualifiedCount: number;
  settledCount: number;
  wins: number;
  losses: number;
  pushes: number;
  forwardUnits: number;
  placedCount: number;
  clvCount: number;
  positiveClvCount: number;
  averageClvPoints: number | null;
};

export type AppSnapshot = {
  generatedAt: string;
  season: number;
  week: number;
  currentWeek?: number;
  nextWeek?: number;
  availableWeeks?: number[];
  games: GameView[];
  gameResults?: GameResult[];
  forwardPicks: ForwardPick[];
  historicalMetrics: HistoricalMetric[];
  performance: PerformanceSummary;
};
