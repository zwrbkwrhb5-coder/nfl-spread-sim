import {
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
import { lineText, marketText, oddsText, signed } from "../data";
import { C } from "../theme";
import { trackedSummary } from "../tracking";

export default function Results() {
  const { trackedBets, removeBet } = useTrackedBets();
  const summary = trackedSummary(trackedBets);

  return (
    <SafeAreaView style={s.page}>
      <ScrollView contentContainerStyle={s.content}>
        <BrandHeader />
        <Text style={s.title}>My Tracking</Text>
        <Text style={s.sub}>
          Only bets you tapped “I BET THIS” on are counted here.
        </Text>

        <View style={s.summary}>
          <View style={s.sumCell}>
            <Text style={s.sumLabel}>RECORD</Text>
            <Text style={[s.sumValue, { color: C.gold }]}>
              {summary.wins}-{summary.losses}
            </Text>
          </View>

          <View style={s.vline} />

          <View style={s.sumCell}>
            <Text style={s.sumLabel}>UNITS</Text>
            <Text
              style={[
                s.sumValue,
                { color: summary.units >= 0 ? C.green : C.red },
              ]}
            >
              {signed(summary.units, 2)}u
            </Text>
          </View>

          <View style={s.vline} />

          <View style={s.sumCell}>
            <Text style={s.sumLabel}>PENDING</Text>
            <Text style={s.sumValue}>{summary.pending}</Text>
          </View>
        </View>

        <Text style={s.note}>
          Current tracking stake: 1 unit per logged bet.
        </Text>

        {summary.settled.length === 0 && (
          <View style={s.empty}>
            <Text style={s.emptyTitle}>No bets tracked yet</Text>
            <Text style={s.emptyText}>
              Go to Picks and tap “I BET THIS” only after you actually place a wager.
            </Text>
          </View>
        )}

        {summary.settled.map((bet) => {
          const fg =
            bet.result === "win"
              ? C.green
              : bet.result === "loss"
                ? C.red
                : bet.result === "push"
                  ? C.gray
                  : C.blue;

          return (
            <View key={bet.id} style={[s.betCard, { borderLeftColor: fg }]}> 
              <View style={s.betHead}>
                <View>
                  <Text style={[s.status, { color: fg }]}> 
                    {bet.result.toUpperCase()}
                  </Text>
                  <Text style={s.match}>{bet.away} @ {bet.home}</Text>
                </View>

                <Text style={[s.profit, { color: fg }]}> 
                  {bet.profitUnits == null
                    ? "PENDING"
                    : `${signed(bet.profitUnits, 2)}u`}
                </Text>
              </View>

              <View style={s.pickRow}>
                <Text style={s.pick}>
                  {lineText(bet.market, bet.pick, bet.line)}
                </Text>
                <View style={s.marketTag}>
                  <Text style={s.marketTagText}>{marketText(bet.market)}</Text>
                </View>
              </View>

              <View style={s.quoteBox}>
                <View>
                  <Text style={s.metaLabel}>SPORTSBOOK</Text>
                  <Text style={s.metaValue}>{bet.book}</Text>
                </View>

                <View>
                  <Text style={s.metaLabel}>ODDS</Text>
                  <Text style={[s.metaValue, { color: C.gold }]}> 
                    {oddsText(bet.odds)}
                  </Text>
                </View>

                <View>
                  <Text style={s.metaLabel}>EDGE WHEN LOGGED</Text>
                  <Text style={s.metaValue}>
                    {bet.edge == null ? "—" : `${signed(bet.edge * 100, 2)}%`}
                  </Text>
                </View>
              </View>

              {bet.awayScore != null && bet.homeScore != null && (
                <Text style={s.final}>
                  Final: {bet.away} {bet.awayScore.toFixed(0)} – {bet.homeScore.toFixed(0)} {bet.home}
                </Text>
              )}

              <Pressable
                onPress={() => removeBet(bet.id)}
                style={s.remove}
              >
                <Text style={s.removeText}>REMOVE FROM MY TRACKER</Text>
              </Pressable>
            </View>
          );
        })}
      </ScrollView>

      <BottomNav active="results" />
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  page: { flex: 1, backgroundColor: C.bg },
  content: { padding: 16, paddingBottom: 100 },

  title: { color: C.text, fontSize: 36, fontWeight: "900", marginTop: 21 },
  sub: {
    color: C.muted,
    fontSize: 13,
    lineHeight: 18,
    marginTop: 2,
    marginBottom: 14,
  },

  summary: {
    flexDirection: "row",
    borderWidth: 1,
    borderColor: C.gold,
    borderRadius: 15,
    paddingVertical: 15,
    backgroundColor: "#11110E",
  },
  sumCell: { flex: 1, alignItems: "center" },
  sumLabel: { color: C.muted, fontSize: 8, fontWeight: "900" },
  sumValue: { color: C.text, fontSize: 22, fontWeight: "900", marginTop: 4 },
  vline: { width: 1, backgroundColor: "#3A3528" },

  note: {
    color: C.dim,
    fontSize: 9,
    marginTop: 8,
    marginBottom: 13,
  },

  betCard: {
    backgroundColor: C.panel,
    borderWidth: 1,
    borderColor: C.line,
    borderLeftWidth: 4,
    borderRadius: 15,
    padding: 14,
    marginBottom: 10,
  },
  betHead: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
  },
  status: { fontSize: 9, fontWeight: "900" },
  match: { color: C.muted, fontSize: 10, marginTop: 3 },
  profit: { fontSize: 16, fontWeight: "900" },

  pickRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 9,
    gap: 10,
  },
  pick: { color: C.text, fontSize: 21, fontWeight: "900", flex: 1 },
  marketTag: {
    backgroundColor: C.panel2,
    borderWidth: 1,
    borderColor: C.line,
    borderRadius: 99,
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  marketTagText: { color: C.gold, fontSize: 8, fontWeight: "900" },

  quoteBox: {
    flexDirection: "row",
    justifyContent: "space-between",
    backgroundColor: C.panel2,
    borderRadius: 10,
    padding: 10,
    marginTop: 10,
  },
  metaLabel: { color: C.dim, fontSize: 7, fontWeight: "900" },
  metaValue: { color: C.text, fontSize: 10, fontWeight: "900", marginTop: 3 },

  final: { color: C.muted, fontSize: 10, marginTop: 9 },

  remove: {
    minHeight: 40,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#51242A",
    backgroundColor: "#1D0C10",
    marginTop: 11,
    alignItems: "center",
    justifyContent: "center",
  },
  removeText: { color: C.red, fontSize: 8, fontWeight: "900" },

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
