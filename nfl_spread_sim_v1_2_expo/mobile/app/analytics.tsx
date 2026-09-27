import {
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import BottomNav from "../components/BottomNav";
import BrandHeader from "../components/BrandHeader";
import { useTrackedBets } from "../components/TrackedBetsContext";
import { signed } from "../data";
import { C } from "../theme";
import { trackedSummary } from "../tracking";

export default function Analytics() {
  const { trackedBets } = useTrackedBets();
  const sData = trackedSummary(trackedBets);

  const graded = sData.wins + sData.losses;
  const winRate = graded ? sData.wins / graded : 0;

  const spreadBets = sData.settled.filter(
    (x) => x.market === "spread" && x.result !== "pending"
  );
  const totalBets = sData.settled.filter(
    (x) => x.market === "total" && x.result !== "pending"
  );
  const moneylineBets = sData.settled.filter(
    (x) => x.market === "moneyline" && x.result !== "pending"
  );

  const record = (xs: typeof sData.settled) => {
    const w = xs.filter((x) => x.result === "win").length;
    const l = xs.filter((x) => x.result === "loss").length;
    return `${w}-${l}`;
  };

  return (
    <SafeAreaView style={s.page}>
      <ScrollView contentContainerStyle={s.content}>
        <BrandHeader />
        <Text style={s.title}>Performance</Text>
        <Text style={s.sub}>Your manually logged bets only</Text>

        <View style={s.summary}>
          <View style={s.cell}>
            <Text style={s.label}>WIN RATE</Text>
            <Text style={[s.big, { color: C.gold }]}> 
              {graded ? `${(winRate * 100).toFixed(1)}%` : "—"}
            </Text>
          </View>

          <View style={s.vline} />

          <View style={s.cell}>
            <Text style={s.label}>UNITS</Text>
            <Text style={[s.big, { color: sData.units >= 0 ? C.green : C.red }]}> 
              {signed(sData.units, 2)}u
            </Text>
          </View>

          <View style={s.vline} />

          <View style={s.cell}>
            <Text style={s.label}>TRACKED</Text>
            <Text style={s.big}>{trackedBets.length}</Text>
          </View>
        </View>

        <View style={s.card}>
          <Text style={s.cardTitle}>My Betting Record</Text>

          <View style={s.row}>
            <Text style={s.rowLabel}>Wins</Text>
            <Text style={[s.rowValue, { color: C.green }]}>{sData.wins}</Text>
          </View>
          <View style={s.row}>
            <Text style={s.rowLabel}>Losses</Text>
            <Text style={[s.rowValue, { color: C.red }]}>{sData.losses}</Text>
          </View>
          <View style={s.row}>
            <Text style={s.rowLabel}>Pushes</Text>
            <Text style={s.rowValue}>{sData.pushes}</Text>
          </View>
          <View style={s.row}>
            <Text style={s.rowLabel}>Pending</Text>
            <Text style={[s.rowValue, { color: C.blue }]}>{sData.pending}</Text>
          </View>
          <View style={s.row}>
            <Text style={s.rowLabel}>ROI</Text>
            <Text style={[s.rowValue, { color: sData.roi >= 0 ? C.green : C.red }]}> 
              {sData.risked ? `${signed(sData.roi * 100, 1)}%` : "—"}
            </Text>
          </View>
        </View>

        <View style={s.two}>
          <View style={s.marketCard}>
            <Text style={s.marketTitle}>SPREAD</Text>
            <Text style={s.marketRecord}>{record(spreadBets)}</Text>
            <Text style={s.marketSub}>{spreadBets.length} graded</Text>
          </View>

          <View style={s.marketCard}>
            <Text style={s.marketTitle}>TOTALS</Text>
            <Text style={s.marketRecord}>{record(totalBets)}</Text>
            <Text style={s.marketSub}>{totalBets.length} graded</Text>
          </View>
        </View>

        <View style={s.marketCardSolo}>
          <Text style={s.marketTitle}>MONEYLINE</Text>
          <Text style={s.marketRecord}>{record(moneylineBets)}</Text>
          <Text style={s.marketSub}>{moneylineBets.length} graded</Text>
        </View>

        <View style={s.rule}>
          <Text style={s.ruleTitle}>TRACKING RULE</Text>
          <Text style={s.ruleText}>
            A model recommendation does not affect this record. It appears here only after you personally tap “I BET THIS.”
          </Text>
        </View>
      </ScrollView>

      <BottomNav active="performance" />
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  page: { flex: 1, backgroundColor: C.bg },
  content: { padding: 16, paddingBottom: 100 },

  title: { color: C.text, fontSize: 36, fontWeight: "900", marginTop: 21 },
  sub: { color: C.muted, fontSize: 14, marginTop: 1, marginBottom: 14 },

  summary: {
    flexDirection: "row",
    borderWidth: 1,
    borderColor: C.gold,
    borderRadius: 15,
    paddingVertical: 15,
    backgroundColor: "#11110E",
  },
  cell: { flex: 1, alignItems: "center" },
  label: { color: C.muted, fontSize: 8, fontWeight: "900" },
  big: { color: C.text, fontSize: 22, fontWeight: "900", marginTop: 4 },
  vline: { width: 1, backgroundColor: "#3A3528" },

  card: {
    marginTop: 12,
    backgroundColor: C.panel,
    borderWidth: 1,
    borderColor: C.line,
    borderRadius: 15,
    padding: 14,
  },
  cardTitle: { color: C.text, fontSize: 14, fontWeight: "900", marginBottom: 5 },
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    borderTopWidth: 1,
    borderTopColor: C.line,
    paddingVertical: 11,
  },
  rowLabel: { color: C.muted, fontSize: 11, fontWeight: "700" },
  rowValue: { color: C.text, fontSize: 13, fontWeight: "900" },

  two: { flexDirection: "row", gap: 9, marginTop: 10 },
  marketCard: {
    flex: 1,
    backgroundColor: C.panel,
    borderWidth: 1,
    borderColor: C.line,
    borderRadius: 15,
    padding: 13,
  },
  marketCardSolo: {
    backgroundColor: C.panel,
    borderWidth: 1,
    borderColor: C.line,
    borderRadius: 15,
    padding: 13,
    marginTop: 10,
  },
  marketTitle: { color: C.gold, fontSize: 8, fontWeight: "900" },
  marketRecord: { color: C.text, fontSize: 24, fontWeight: "900", marginTop: 5 },
  marketSub: { color: C.muted, fontSize: 9, marginTop: 2 },

  rule: {
    backgroundColor: "#151309",
    borderWidth: 1,
    borderColor: C.gold,
    borderRadius: 14,
    padding: 13,
    marginTop: 10,
  },
  ruleTitle: { color: C.gold, fontSize: 9, fontWeight: "900" },
  ruleText: { color: C.text, fontSize: 10, lineHeight: 16, marginTop: 5 },
});
