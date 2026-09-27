import { StyleSheet, Text, View } from "react-native";
import { C } from "../theme";

export default function BrandHeader({
  right = "This Week",
}: {
  right?: string;
}) {
  return (
    <View style={s.row}>
      <Text style={s.brand}>
        NFL <Text style={s.gold}>SIM</Text>
      </Text>
      <View style={s.pill}>
        <Text style={s.pillText}>▣  {right}⌄</Text>
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  brand: {
    color: C.text,
    fontSize: 24,
    fontWeight: "900",
    fontStyle: "italic",
  },
  gold: { color: C.gold },
  pill: {
    borderWidth: 1,
    borderColor: "#37404A",
    borderRadius: 10,
    paddingHorizontal: 11,
    paddingVertical: 8,
    backgroundColor: "#101419",
  },
  pillText: {
    color: C.text,
    fontWeight: "700",
    fontSize: 11,
  },
});
