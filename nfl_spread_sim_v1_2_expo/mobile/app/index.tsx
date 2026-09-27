import { Link } from "expo-router";
import { useEffect, useState } from "react";
import {
  FlatList,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import BottomNav from "../components/BottomNav";
import BrandHeader from "../components/BrandHeader";
import WeekSelector from "../components/WeekSelector";
import {
  APP_DATA,
  availableWeeks,
  bestAngle,
  currentWeek,
  kickoffText,
  lineText,
  marketText,
  oddsText,
  projectedMoneylineWinner,
  projectedScore,
  signed,
  teamLogo,
  upcomingGames,
} from "../data";
import { C } from "../theme";
import type { GameView } from "../types";

function GameCard({ game }: { game: GameView }) {
  const pick = bestAngle(game);
  const score = projectedScore(game);
  const mlWinner = projectedMoneylineWinner(game);
  const qualifies = Boolean(pick?.qualifies);

  return (
    <Link href={{ pathname: "/game/[id]", params: { id: game.id } }} asChild>
      <Pressable
        style={StyleSheet.flatten([
          s.gameCard,
          qualifies ? s.gameCardQualified : null,
        ])}
      >
        <View style={s.gameTop}>
          <View style={s.teams}>
            <Image source={{ uri: teamLogo(game.away) }} style={s.logo} />
            <View style={{ flex: 1 }}>
              <Text style={s.gameName}>{game.away} @ {game.home}</Text>
              <Text style={s.kickoff}>{kickoffText(game)}</Text>
            </View>
            <Image source={{ uri: teamLogo(game.home) }} style={s.logo} />
          </View>

          <View
            style={[
              s.status,
              { backgroundColor: qualifies ? C.greenSoft : C.panel3 },
            ]}
          >
            <Text
              style={[
                s.statusText,
                { color: qualifies ? C.green : C.muted },
              ]}
            >
              {qualifies ? "QUALIFIED" : "PASS"}
            </Text>
          </View>
        </View>

        {score && (
          <View style={s.projection}>
            <View>
              <Text style={s.projLabel}>PROJECTED SCORE</Text>
              <Text style={s.projScore}>
                {game.away} {score.away.toFixed(0)}  –  {score.home.toFixed(0)} {game.home}
              </Text>
            </View>
            <View style={s.totalBox}>
              <Text style={s.totalLabel}>TOTAL</Text>
              <Text style={s.totalValue}>{game.modelTotal?.toFixed(1)}</Text>
            </View>
          </View>
        )}

        {mlWinner && (
          <View style={s.mlProjection}>
            <View style={{ flex: 1 }}>
              <Text style={s.mlLabel}>PROJECTED MONEYLINE WINNER</Text>
              <Text style={s.mlWinner}>{mlWinner.team ?? "TOSS-UP"}</Text>
              <Text style={s.mlSub}>
                {mlWinner.team
                  ? `Model projects ${mlWinner.team} by ${mlWinner.margin.toFixed(1)} points`
                  : "Model margin is essentially even"}
              </Text>
            </View>
            {game.moneyline && (
              <View style={s.mlOdds}>
                <Text style={s.mlOddsLabel}>BEST ML</Text>
                <Text style={s.mlOddsValue}>
                  {game.moneyline.pick} {oddsText(game.moneyline.oddsBoard[0]?.odds ?? game.moneyline.odds)}
                </Text>
              </View>
            )}
          </View>
        )}

        {pick && (
          <View style={s.pickRow}>
            <View style={{ flex: 1, paddingRight: 10 }}>
              <Text style={s.pickLabel}>
                {qualifies ? "MODEL PICK" : "BEST ANGLE"} • {marketText(pick.market)}
              </Text>
              <Text style={s.pickText}>{lineText(pick.market, pick.pick, pick.line)}</Text>
              <Text style={s.book}>
                ★ #1 {pick.oddsBoard[0]?.book ?? pick.book} • {oddsText(pick.oddsBoard[0]?.odds ?? pick.odds)}
              </Text>
              {pick.oddsBoard[1] && (
                <Text style={s.book}>
                  #2 {pick.oddsBoard[1].book} • {oddsText(pick.oddsBoard[1].odds)}
                </Text>
              )}
            </View>
            <View style={s.edgeBox}>
              <Text style={s.edgeLabel}>EDGE</Text>
              <Text style={[s.edgeValue, { color: qualifies ? C.green : C.muted }]}>
                {signed((pick.edge ?? 0) * 100, 2)}%
              </Text>
            </View>
          </View>
        )}
      </Pressable>
    </Link>
  );
}

export default function Home() {
  const weeks = availableWeeks();
  const [selectedWeek, setSelectedWeek] = useState(currentWeek);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(timer);
  }, []);

  const games = upcomingGames(now, selectedWeek);
  const qualifiers = games.filter((g) => g.qualifies).length;

  return (
    <SafeAreaView style={s.page}>
      <FlatList
        data={games}
        keyExtractor={(x) => x.id}
        renderItem={({ item }) => <GameCard game={item} />}
        ItemSeparatorComponent={() => <View style={{ height: 11 }} />}
        contentContainerStyle={s.content}
        ListHeaderComponent={
          <>
            <BrandHeader />
            <Text style={s.title}>Dashboard</Text>
            <Text style={s.sub}>Current + next week model predictions</Text>

            <Text style={s.weekCaption}>CHOOSE WEEK</Text>
            <WeekSelector
              weeks={weeks}
              selectedWeek={selectedWeek}
              currentWeek={currentWeek}
              onSelect={setSelectedWeek}
            />

            <View style={s.summary}>
              <View style={s.summaryItem}>
                <Text style={s.summaryLabel}>UPCOMING</Text>
                <Text style={s.summaryValue}>{games.length}</Text>
              </View>
              <View style={s.vline} />
              <View style={s.summaryItem}>
                <Text style={s.summaryLabel}>QUALIFIED</Text>
                <Text style={[s.summaryValue, { color: C.green }]}>{qualifiers}</Text>
              </View>
              <View style={s.vline} />
              <View style={s.summaryItem}>
                <Text style={s.summaryLabel}>VIEWING</Text>
                <Text style={[s.summaryValue, { color: C.gold }]}>W{selectedWeek}</Text>
              </View>
            </View>

            <View style={s.sectionRow}>
              <Text style={s.sectionTitle}>Week {selectedWeek} Games</Text>
              <Text style={s.sectionCount}>{games.length} GAMES</Text>
            </View>

            {games.length === 0 && (
              <View style={s.empty}>
                <Text style={s.emptyTitle}>No upcoming games</Text>
                <Text style={s.emptyText}>
                  If this is next week, refresh the model export after its odds are available.
                </Text>
              </View>
            )}
          </>
        }
      />

      <BottomNav active="home" />
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  page: { flex: 1, backgroundColor: C.bg },
  content: { padding: 16, paddingBottom: 100 },

  weekCaption: { color: C.dim, fontSize: 9, fontWeight: "800", marginBottom: 7 },

  title: { color: C.text, fontSize: 34, fontWeight: "900", marginTop: 21 },
  sub: { color: C.muted, fontSize: 14, marginTop: 1, marginBottom: 14 },

  summary: {
    flexDirection: "row",
    borderWidth: 1,
    borderColor: C.lineGold,
    borderRadius: 16,
    paddingVertical: 15,
    marginBottom: 23,
    backgroundColor: "#10100D",
  },
  summaryItem: { flex: 1, alignItems: "center" },
  summaryLabel: { color: C.muted, fontSize: 8, fontWeight: "800" },
  summaryValue: { color: C.text, fontSize: 22, fontWeight: "900", marginTop: 4 },
  vline: { width: 1, backgroundColor: "#353126" },

  sectionRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 9,
  },
  sectionTitle: { color: C.text, fontSize: 21, fontWeight: "900" },
  sectionCount: { color: C.muted, fontSize: 9, fontWeight: "800" },

  gameCard: {
    backgroundColor: C.panel,
    borderWidth: 1,
    borderColor: C.line,
    borderRadius: 18,
    padding: 14,
  },
  gameCardQualified: {
    borderColor: "#68532A",
    shadowColor: C.gold,
    shadowOpacity: 0.12,
    shadowRadius: 7,
  },
  gameTop: { flexDirection: "row", alignItems: "center", gap: 8 },
  teams: { flexDirection: "row", alignItems: "center", gap: 8, flex: 1 },
  logo: { width: 44, height: 44, resizeMode: "contain" },
  gameName: { color: C.text, fontSize: 17, fontWeight: "900" },
  kickoff: { color: C.muted, fontSize: 10, marginTop: 2 },
  status: { paddingHorizontal: 8, paddingVertical: 5, borderRadius: 99 },
  statusText: { fontSize: 8, fontWeight: "900" },

  projection: {
    marginTop: 13,
    backgroundColor: C.panel2,
    borderRadius: 12,
    padding: 11,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  projLabel: { color: C.dim, fontSize: 7, fontWeight: "900" },
  projScore: { color: C.text, fontSize: 14, fontWeight: "900", marginTop: 3 },
  totalBox: { alignItems: "flex-end" },
  totalLabel: { color: C.dim, fontSize: 7, fontWeight: "900" },
  totalValue: { color: C.gold, fontSize: 16, fontWeight: "900", marginTop: 2 },

  mlProjection: {
    marginTop: 9,
    backgroundColor: "#151309",
    borderWidth: 1,
    borderColor: C.lineGold,
    borderRadius: 12,
    padding: 11,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 10,
  },
  mlLabel: { color: C.gold, fontSize: 7, fontWeight: "900" },
  mlWinner: { color: C.text, fontSize: 20, fontWeight: "900", marginTop: 3 },
  mlSub: { color: C.muted, fontSize: 9, marginTop: 2 },
  mlOdds: { alignItems: "flex-end" },
  mlOddsLabel: { color: C.dim, fontSize: 7, fontWeight: "900" },
  mlOddsValue: { color: C.gold, fontSize: 12, fontWeight: "900", marginTop: 3 },

  pickRow: {
    marginTop: 12,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-end",
  },
  pickLabel: { color: C.dim, fontSize: 7, fontWeight: "900" },
  pickText: { color: C.text, fontSize: 22, fontWeight: "900", marginTop: 3 },
  book: { color: C.muted, fontSize: 10, marginTop: 4 },
  edgeBox: { alignItems: "flex-end" },
  edgeLabel: { color: C.dim, fontSize: 7, fontWeight: "900" },
  edgeValue: { color: C.text, fontSize: 21, fontWeight: "900", marginTop: 2 },

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
