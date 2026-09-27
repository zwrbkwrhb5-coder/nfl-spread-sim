import { router, useLocalSearchParams } from "expo-router";
import {
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { useTrackedBets } from "../../components/TrackedBetsContext";
import {
  getGame,
  lineText,
  marketText,
  oddsText,
  pct,
  projectedMoneylineWinner,
  projectedScore,
  signed,
  teamLogo,
} from "../../data";
import { C } from "../../theme";
import type { GameView, MarketPick } from "../../types";

function MarketCard({ p, game }: { p: MarketPick; game: GameView }) {
  const { logBet, isTracked } = useTrackedBets();

  const quotes = p.oddsBoard.length
    ? p.oddsBoard.slice(0, 2)
    : [{
        line: p.line,
        odds: p.odds,
        book: p.book,
        edge: p.edge,
        breakEven: p.breakEven,
      }];

  return (
    <View style={[s.marketCard, p.qualifies && { borderColor: C.gold }]}> 
      <View style={s.marketTop}>
        <View>
          <Text style={s.marketLabel}>{marketText(p.market)}</Text>
          <Text style={s.marketPick}>
            {lineText(p.market, p.pick, p.line)}
          </Text>
        </View>

        <View style={[s.badge, { backgroundColor: p.qualifies ? C.greenSoft : C.panel3 }]}> 
          <Text style={[s.badgeText, { color: p.qualifies ? C.green : C.muted }]}> 
            {p.qualifies ? "QUALIFIED" : "PASS"}
          </Text>
        </View>
      </View>

      <View style={s.metrics}>
        <View style={s.metric}>
          <Text style={s.metricLabel}>MODEL PROB</Text>
          <Text style={s.metricValue}>{pct(p.selectedProbability)}</Text>
        </View>

        <View style={s.metric}>
          <Text style={s.metricLabel}>BREAK-EVEN</Text>
          <Text style={s.metricValue}>{pct(p.breakEven)}</Text>
        </View>

        <View style={s.metric}>
          <Text style={s.metricLabel}>EDGE</Text>
          <Text style={[s.metricValue, { color: p.qualifies ? C.green : C.muted }]}> 
            {signed((p.edge ?? 0) * 100, 2)}%
          </Text>
        </View>
      </View>

      <View style={s.oddsBoard}>
        <View style={s.oddsHead}>
          <Text style={s.oddsTitle}>TOP 2 OPTIONS</Text>
          <Text style={s.oddsSub}>CHOOSE WHAT YOU ACTUALLY BET</Text>
        </View>

        {quotes.map((q, qi) => {
          const bet = {
            gameId: game.id,
            away: game.away,
            home: game.home,
            kickoffAt: game.kickoffAt,
            market: p.market,
            pick: p.pick,
            line: q.line,
            odds: q.odds,
            book: q.book,
            edge: q.edge ?? p.edge,
            selectedProbability: p.selectedProbability,
            breakEven: q.breakEven ?? p.breakEven,
            stakeUnits: 1,
          };
          const tracked = isTracked(bet);

          return (
            <View
              key={`${q.book}-${q.line}-${q.odds}-${qi}`}
              style={[s.oddsRow, qi === 0 && s.oddsRowSelected]}
            >
              <Text style={[s.oddsBook, qi === 0 && { color: C.gold }]}> 
                {qi === 0 ? "★ #1 " : "#2 "}{q.book}
              </Text>

              <View style={s.quoteLine}>
                <Text style={s.oddsLine}>{lineText(p.market, p.pick, q.line)}</Text>
                <Text style={[s.oddsNumber, qi === 0 && { color: C.gold }]}> 
                  {oddsText(q.odds)}
                </Text>
                <Text style={[s.oddsEdge, qi === 0 && { color: C.green }]}> 
                  {q.edge == null ? "—" : `${signed(q.edge * 100, 2)}% edge`}
                </Text>
              </View>

              <Pressable
                disabled={tracked}
                onPress={() => logBet(bet)}
                style={[s.betButton, tracked && s.betButtonLogged]}
              >
                <Text style={[s.betButtonText, tracked && s.betButtonTextLogged]}>
                  {tracked ? "✓ BET LOGGED" : "I BET THIS • TRACK 1U"}
                </Text>
              </Pressable>
            </View>
          );
        })}

        <Text style={s.oddsHelp}>
          Simply viewing a model pick never adds it to your personal tracking.
        </Text>
      </View>
    </View>
  );
}

export default function GameDetail() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const g = getGame(String(id));
  if (!g) return null;

  const score = projectedScore(g);
  const mlWinner = projectedMoneylineWinner(g);

  return (
    <SafeAreaView style={s.page} edges={["top", "left", "right"]}>
      <ScrollView
        style={s.page}
        contentContainerStyle={s.content}
        contentInsetAdjustmentBehavior="never"
      >
        <View style={s.topBar}>
          <Pressable
            onPress={() => router.back()}
            style={s.back}
            hitSlop={10}
            accessibilityRole="button"
            accessibilityLabel="Go back"
          >
            <Text style={s.backText}>‹</Text>
          </Pressable>

          <Text style={s.brand}>
            NFL <Text style={{ color: C.gold }}>SIM</Text>
          </Text>

          <View style={s.topSpacer} />
        </View>

        <View style={s.match}>
          <View style={s.team}>
            <Image source={{ uri: teamLogo(g.away) }} style={s.heroLogo} />
            <Text style={s.teamName}>{g.away}</Text>
          </View>

          <Text style={s.at}>@</Text>

          <View style={s.team}>
            <Image source={{ uri: teamLogo(g.home) }} style={s.heroLogo} />
            <Text style={s.teamName}>{g.home}</Text>
          </View>
        </View>

        {score && (
          <View style={s.scoreCard}>
            <Text style={s.scoreTitle}>MODEL PROJECTED SCORE</Text>
            <Text style={s.score}>
              {g.away} {score.away.toFixed(0)} – {score.home.toFixed(0)} {g.home}
            </Text>
            <Text style={s.total}>Projected total {g.modelTotal?.toFixed(1)}</Text>
          </View>
        )}

        {mlWinner && (
          <View style={s.mlWinnerCard}>
            <Text style={s.mlWinnerLabel}>PROJECTED MONEYLINE WINNER</Text>
            <View style={s.mlWinnerRow}>
              <View style={{ flex: 1 }}>
                <Text style={s.mlWinnerTeam}>
                  {mlWinner.team ?? "TOSS-UP"}
                </Text>
                <Text style={s.mlWinnerSub}>
                  {mlWinner.team
                    ? `Projected by ${mlWinner.margin.toFixed(1)} points`
                    : "Projected margin is essentially even"}
                </Text>
              </View>

              {g.moneyline && (
                <View style={s.mlBestQuote}>
                  <Text style={s.mlBestQuoteLabel}>BEST AVAILABLE</Text>
                  <Text style={s.mlBestQuoteValue}>
                    {g.moneyline.pick} {oddsText(g.moneyline.oddsBoard[0]?.odds ?? g.moneyline.odds)}
                  </Text>
                  <Text style={s.mlBestQuoteBook}>
                    {g.moneyline.oddsBoard[0]?.book ?? g.moneyline.book}
                  </Text>
                </View>
              )}
            </View>
            {!g.moneyline && (
              <Text style={s.mlNoOdds}>
                Winner projection is available from the score model; moneyline odds are not present in this export yet.
              </Text>
            )}
          </View>
        )}

        <Text style={s.section}>CURRENT MARKETS</Text>
        {g.spread && <MarketCard p={g.spread} game={g} />}
        {g.total && <MarketCard p={g.total} game={g} />}
        {g.moneyline && <MarketCard p={g.moneyline} game={g} />}
      </ScrollView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  page: { flex: 1, backgroundColor: C.bg },
  content: { padding: 17, paddingBottom: 40 },

  topBar: {
    minHeight: 54,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 4,
  },
  back: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: C.panel2,
    borderWidth: 1,
    borderColor: C.line,
    alignItems: "center",
    justifyContent: "center",
  },
  backText: {
    color: C.text,
    fontSize: 30,
    lineHeight: 32,
    marginTop: -2,
  },

  brand: {
    color: C.text,
    fontSize: 24,
    fontWeight: "900",
    fontStyle: "italic",
    textAlign: "center",
  },
  topSpacer: {
    width: 44,
    height: 44,
  },

  match: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    marginTop: 25,
    gap: 16,
  },
  team: { alignItems: "center" },
  heroLogo: { width: 100, height: 100, resizeMode: "contain" },
  teamName: { color: C.text, fontSize: 18, fontWeight: "900", marginTop: 4 },
  at: { color: C.gold, fontSize: 23, fontWeight: "900" },

  scoreCard: {
    backgroundColor: C.panel,
    borderWidth: 1,
    borderColor: C.gold,
    borderRadius: 17,
    padding: 15,
    marginTop: 18,
    alignItems: "center",
  },
  scoreTitle: { color: C.muted, fontSize: 8, fontWeight: "900" },
  score: { color: C.text, fontSize: 25, fontWeight: "900", marginTop: 6 },
  total: { color: C.gold, fontSize: 11, fontWeight: "800", marginTop: 4 },

  mlWinnerCard: {
    backgroundColor: "#151309",
    borderWidth: 1,
    borderColor: C.gold,
    borderRadius: 17,
    padding: 15,
    marginTop: 10,
  },
  mlWinnerLabel: { color: C.gold, fontSize: 8, fontWeight: "900", letterSpacing: 0.8 },
  mlWinnerRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
    marginTop: 8,
  },
  mlWinnerTeam: { color: C.text, fontSize: 28, fontWeight: "900" },
  mlWinnerSub: { color: C.muted, fontSize: 11, marginTop: 3 },
  mlBestQuote: { alignItems: "flex-end" },
  mlBestQuoteLabel: { color: C.dim, fontSize: 7, fontWeight: "900" },
  mlBestQuoteValue: { color: C.gold, fontSize: 14, fontWeight: "900", marginTop: 3 },
  mlBestQuoteBook: { color: C.muted, fontSize: 9, marginTop: 2 },
  mlNoOdds: {
    color: C.dim,
    fontSize: 9,
    lineHeight: 14,
    borderTopWidth: 1,
    borderTopColor: C.line,
    marginTop: 10,
    paddingTop: 9,
  },

  section: {
    color: C.muted,
    fontSize: 9,
    fontWeight: "900",
    letterSpacing: 1.2,
    marginTop: 21,
    marginBottom: 8,
  },

  marketCard: {
    backgroundColor: C.panel,
    borderWidth: 1,
    borderColor: C.line,
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
  },
  marketTop: { flexDirection: "row", justifyContent: "space-between" },
  marketLabel: { color: C.gold, fontSize: 8, fontWeight: "900" },
  marketPick: { color: C.text, fontSize: 20, fontWeight: "900", marginTop: 3 },

  badge: {
    borderRadius: 99,
    paddingHorizontal: 8,
    paddingVertical: 5,
    height: 26,
  },
  badgeText: { fontSize: 8, fontWeight: "900" },

  metrics: { flexDirection: "row", gap: 8, marginTop: 14 },
  metric: {
    flex: 1,
    backgroundColor: C.panel2,
    borderRadius: 10,
    padding: 10,
  },
  metricLabel: { color: C.dim, fontSize: 7, fontWeight: "900" },
  metricValue: { color: C.text, fontSize: 14, fontWeight: "900", marginTop: 3 },

  oddsBoard: {
    backgroundColor: "#090C0F",
    borderRadius: 13,
    borderWidth: 1,
    borderColor: "#2B3137",
    padding: 10,
    marginTop: 12,
  },
  oddsHead: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingBottom: 6,
  },
  oddsTitle: { color: C.gold, fontSize: 8, fontWeight: "900" },
  oddsSub: { color: C.dim, fontSize: 7, fontWeight: "900" },

  oddsRow: {
    borderTopWidth: 1,
    borderTopColor: "#22282E",
    paddingVertical: 10,
  },
  oddsRowSelected: {
    backgroundColor: "#171409",
    borderRadius: 9,
    paddingHorizontal: 8,
  },
  oddsBook: { color: C.text, fontSize: 11, fontWeight: "900" },
  quoteLine: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    marginTop: 5,
    flexWrap: "wrap",
  },
  oddsLine: { color: C.text, fontSize: 12, fontWeight: "900" },
  oddsNumber: { color: C.text, fontSize: 13, fontWeight: "900" },
  oddsEdge: { color: C.muted, fontSize: 10, fontWeight: "900" },

  betButton: {
    minHeight: 44,
    marginTop: 9,
    borderRadius: 9,
    borderWidth: 1,
    borderColor: C.gold,
    backgroundColor: C.gold,
    alignItems: "center",
    justifyContent: "center",
  },
  betButtonLogged: {
    backgroundColor: C.greenSoft,
    borderColor: C.green,
  },
  betButtonText: { color: "#17120A", fontSize: 9, fontWeight: "900" },
  betButtonTextLogged: { color: C.green },

  oddsHelp: { color: C.dim, fontSize: 8, lineHeight: 13, marginTop: 8 },
});
