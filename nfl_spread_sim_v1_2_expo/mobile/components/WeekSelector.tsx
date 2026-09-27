import { Pressable, StyleSheet, Text, View } from "react-native";
import { C } from "../theme";

export default function WeekSelector({
  weeks,
  selectedWeek,
  currentWeek,
  onSelect,
}: {
  weeks: number[];
  selectedWeek: number;
  currentWeek: number;
  onSelect: (week: number) => void;
}) {
  if (weeks.length <= 1) return null;

  return (
    <View style={s.wrap}>
      {weeks.map((week) => {
        const active = week === selectedWeek;
        const label =
          week === currentWeek
            ? `WEEK ${week} • CURRENT`
            : week === currentWeek + 1
              ? `WEEK ${week} • NEXT`
              : `WEEK ${week}`;

        return (
          <Pressable
            key={week}
            onPress={() => onSelect(week)}
            style={[s.button, active && s.buttonActive]}
            accessibilityRole="button"
            accessibilityState={{ selected: active }}
          >
            <Text style={[s.text, active && s.textActive]}>{label}</Text>
          </Pressable>
        );
      })}
    </View>
  );
}

const s = StyleSheet.create({
  wrap: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 14,
  },
  button: {
    flex: 1,
    minHeight: 44,
    borderRadius: 11,
    borderWidth: 1,
    borderColor: C.line,
    backgroundColor: C.panel,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 8,
  },
  buttonActive: {
    backgroundColor: C.gold,
    borderColor: C.gold,
  },
  text: {
    color: C.muted,
    fontSize: 8,
    fontWeight: "900",
  },
  textActive: {
    color: "#17120A",
  },
});
