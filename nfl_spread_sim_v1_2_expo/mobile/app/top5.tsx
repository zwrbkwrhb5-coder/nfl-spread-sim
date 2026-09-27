import { useState } from "react";
import {
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import BottomNav from "../components/BottomNav";
import BrandHeader from "../components/BrandHeader";
import { useTrackedBets } from "../components/TrackedBetsContext";
import WeekSelector from "../components/WeekSelector";
import {
  availableWeeks,
  currentWeek,
  kickoffText,
  lineText,
  marketText,
  oddsText,
  pct,
  signed,
  teamLogo,
  upcomingQualifiedPicks,
} from "../data";
import { C } from "../theme";

export default function Picks() {
  const weeks = availableWeeks();
  const [selectedWeek, setSelectedWeek] = useState(currentWeek);
  const modelPicks = upcomingQualifiedPicks(Date.now(), selectedWeek);
  const { logBet, isTracked } = useTrackedBets();

  return (
    <SafeAreaView style={s.page}>
      <ScrollView contentContainerStyle={s.content}>
        <BrandHeader />
        <Text style={s.title}>Best Bets</Text>
        <Text style={s.sub}>
          Spread, total, and moneyline picks. Only bets you tap “I BET THIS” on are tracked.
        </Text>

        <Text style={s.weekCaption}>CHOOSE WEEK</Text>
        <WeekSelector
          weeks={weeks}
          selectedWeek={selectedWeek}
          currentWeek={currentWeek}
          onSelect={setSelectedWeek}
        />

        {modelPicks.length === 0 && (
          <View style={s.empty}>
            <Text style={s.emptyTitle}>No Week {selectedWeek} qualifiers yet</Text>
            <Text style={s.emptyText}>
              The model will not lower its edge threshold just to create a pick.
            </Text>
          </View>
        )}

        {modelPicks.map((item, index) => {
          const logoTeam = item.pick.market === "total" ? item.game.home : item.pick.pick;
          return (
            <View
              key={`${item.game.id}-${item.pick.market}`}
              style={[s.card, index === 0 && s.heroCard]}
            >
              <View style={s.rank}>
                <Text style={s.rankText}>
                  {index === 0 ? "♛  #1 MODEL PICK" : `#${index + 1} MODEL PICK`}
                </Text>
              </View>

              <View style={s.pickHeader}>
                <Image source={{ uri: teamLogo(logoTeam) }} style={s.logo} />
                <View style={{ flex: 1 }}>
                  <Text style={s.pickTitle}>
                    {lineText(item.pick.market, item.pick.pick, item.pick.line)}
                  </Text>
                  <Text style={s.vs}>
                    {item.game.away} @ {item.game.home} • {kickoffText(item.game)}
                  </Text>
                </View>
                <View style={s.marketPill}>
                  <Text style={s.marketPillText}>{marketText(item.pick.market)}</Text>
                </View>
              </View>

              <View style={s.metrics}>
                <View style={s.metric}>
                  <Text style={s.metricLabel}>EDGE</Text>
                  <Text style={[s.metricValue, { color: C.gold }]}>
                    {signed((item.pick.edge ?? 0) * 100, 2)}%
                  </Text>
                </View>
                <View style={s.metric}>
                  <Text style={s.metricLabel}>MODEL</Text>
                  <Text style={s.metricValue}>{pct(item.pick.selectedProbability)}</Text>
                </View>
                <View style={s.metric}>
                  <Text style={s.metricLabel}>BREAK EVEN</Text>
                  <Text style={s.metricValue}>{pct(item.pick.breakEven)}</Text>
                </View>
              </View>

              <View style={s.topTwoWrap}>
                <Text style={s.topTwoTitle}>TOP 2 AVAILABLE OPTIONS</Text>

                {(item.pick.oddsBoard.length
                  ? item.pick.oddsBoard.slice(0, 2)
                  : [{
                      line: item.pick.line,
                      odds: item.pick.odds,
                      book: item.pick.book,
                      edge: item.pick.edge,
                      breakEven: item.pick.breakEven,
                    }]
                ).map((q, qi) => {
                  const bet = {
                    gameId: item.game.id,
                    away: item.game.away,
                    home: item.game.home,
                    kickoffAt: item.game.kickoffAt,
                    market: item.pick.market,
                    pick: item.pick.pick,
                    line: q.line,
                    odds: q.odds,
                    book: q.book,
                    edge: q.edge ?? item.pick.edge,
                    selectedProbability: item.pick.selectedProbability,
                    breakEven: q.breakEven ?? item.pick.breakEven,
                    stakeUnits: 1,
                  };
                  const tracked = isTracked(bet);

                  return (
                    <View
                      key={`${q.book}-${q.line}-${q.odds}-${qi}`}
                      style={[s.optionCard, qi === 0 && s.optionCardBest]}
                    >
                      <View style={s.optionRow}>
                        <View style={s.optionRank}>
                          <Text style={s.optionRankText}>
                            {qi === 0 ? "★ #1 BEST" : "#2 NEXT BEST"}
                          </Text>
                        </View>

                        <View style={{ flex: 1, minWidth: 0 }}>
                          <Text style={s.optionBook}>{q.book}</Text>
                          <Text style={s.optionLine}>
                            {lineText(item.pick.market, item.pick.pick, q.line)}
                          </Text>
                        </View>

                        <View style={s.optionRight}>
                          <Text style={[s.optionOdds, qi === 0 && { color: C.gold }]}> 
                            {oddsText(q.odds)}
                          </Text>
                          <Text style={[s.optionEdge, qi === 0 && { color: C.green }]}> 
                            {q.edge == null ? "—" : `${signed(q.edge * 100, 2)}% edge`}
                          </Text>
                        </View>
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

                <Text style={s.shopHelp}>
                  Only bets you personally log with this button count in Results and Performance.
                </Text>
              </View>
            </View>
          );
        })}
      </ScrollView>

      <BottomNav active="picks" />
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  page: { flex: 1, backgroundColor: C.bg },
  content: { padding: 16, paddingBottom: 100 },

  weekCaption: { color: C.dim, fontSize: 9, fontWeight: "800", marginBottom: 7 },

  title: { color: C.text, fontSize: 36, fontWeight: "900", marginTop: 21 },
  sub: {
    color: C.muted,
    fontSize: 13,
    lineHeight: 18,
    marginTop: 2,
    marginBottom: 15,
  },

  card: {
    backgroundColor: C.panel,
    borderWidth: 1,
    borderColor: C.line,
    borderRadius: 18,
    padding: 15,
    marginBottom: 11,
  },
  heroCard: {
    borderColor: C.gold,
    shadowColor: C.gold,
    shadowOpacity: 0.22,
    shadowRadius: 9,
  },
  rank: {
    alignSelf: "flex-start",
    backgroundColor: C.gold,
    borderRadius: 99,
    paddingHorizontal: 11,
    paddingVertical: 6,
    marginBottom: 12,
  },
  rankText: { color: "#17120A", fontSize: 9, fontWeight: "900" },

  pickHeader: { flexDirection: "row", alignItems: "center", gap: 12 },
  logo: { width: 56, height: 56, resizeMode: "contain" },
  pickTitle: { color: C.text, fontSize: 23, fontWeight: "900" },
  vs: { color: C.muted, fontSize: 10, marginTop: 3 },
  marketPill: {
    backgroundColor: C.panel2,
    borderWidth: 1,
    borderColor: C.line,
    borderRadius: 99,
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  marketPillText: { color: C.gold, fontSize: 8, fontWeight: "900" },

  metrics: {
    flexDirection: "row",
    backgroundColor: C.panel2,
    borderRadius: 13,
    marginTop: 14,
    padding: 12,
  },
  metric: { flex: 1 },
  metricLabel: { color: C.dim, fontSize: 7, fontWeight: "900" },
  metricValue: { color: C.text, fontSize: 13, fontWeight: "900", marginTop: 3 },

  topTwoWrap: {
    backgroundColor: C.panel2,
    borderRadius: 13,
    padding: 10,
    marginTop: 12,
  },
  topTwoTitle: {
    color: C.muted,
    fontSize: 8,
    fontWeight: "900",
    letterSpacing: 1,
    marginBottom: 7,
  },

  optionCard: {
    backgroundColor: "#0D1115",
    borderWidth: 1,
    borderColor: C.line,
    borderRadius: 11,
    padding: 10,
    marginBottom: 8,
  },
  optionCardBest: {
    backgroundColor: "#171409",
    borderColor: C.gold,
  },
  optionRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 9,
  },
  optionRank: {
    minWidth: 68,
    borderRadius: 99,
    backgroundColor: C.panel2,
    paddingHorizontal: 8,
    paddingVertical: 6,
  },
  optionRankText: { color: C.muted, fontSize: 7, fontWeight: "900" },
  optionBook: { color: C.text, fontSize: 11, fontWeight: "900" },
  optionLine: { color: C.text, fontSize: 13, fontWeight: "900", marginTop: 3 },
  optionRight: { alignItems: "flex-end" },
  optionOdds: { color: C.text, fontSize: 14, fontWeight: "900" },
  optionEdge: { color: C.muted, fontSize: 9, fontWeight: "900", marginTop: 4 },

  betButton: {
    minHeight: 42,
    borderRadius: 9,
    borderWidth: 1,
    borderColor: C.gold,
    backgroundColor: C.gold,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 10,
  },
  betButtonLogged: {
    backgroundColor: C.greenSoft,
    borderColor: C.green,
  },
  betButtonText: { color: "#17120A", fontSize: 9, fontWeight: "900" },
  betButtonTextLogged: { color: C.green },
  shopHelp: { color: C.dim, fontSize: 8, lineHeight: 13, marginTop: 4 },

  empty: {
    backgroundColor: C.panel,
    borderWidth: 1,
    borderColor: C.line,
    borderRadius: 16,
    padding: 18,
  },
  emptyTitle: { color: C.text, fontSize: 17, fontWeight: "900" },
  emptyText: { color: C.muted, fontSize: 11, lineHeight: 17, marginTop: 6 },
});
