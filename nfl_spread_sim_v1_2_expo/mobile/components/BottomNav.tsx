import { Link } from "expo-router";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { C } from "../theme";

type Tab = "home" | "picks" | "results" | "performance";

const tabs = [
  { key: "home" as const, label: "Home", icon: "⌂", href: "/" as const },
  { key: "picks" as const, label: "Picks", icon: "▤", href: "/top5" as const },
  { key: "results" as const, label: "Results", icon: "▥", href: "/performance" as const },
  { key: "performance" as const, label: "Performance", icon: "↗", href: "/analytics" as const },
];

export default function BottomNav({ active }: { active: Tab }) {
  return (
    <View style={s.nav}>
      {tabs.map((tab) => {
        const selected = active === tab.key;
        return (
          <Link href={tab.href} key={tab.key} asChild>
            <Pressable
              style={StyleSheet.flatten([s.tab, selected ? s.activeTab : null])}
            >
              <Text style={[s.icon, selected && s.activeText]}>{tab.icon}</Text>
              <Text style={[s.label, selected && s.activeText]}>{tab.label}</Text>
            </Pressable>
          </Link>
        );
      })}
    </View>
  );
}

const s = StyleSheet.create({
  nav: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    height: 76,
    paddingBottom: 8,
    backgroundColor: "rgba(7,9,10,0.98)",
    borderTopWidth: 1,
    borderTopColor: "#22272C",
    flexDirection: "row",
  },
  tab: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 3,
  },
  activeTab: {
    borderTopWidth: 2,
    borderTopColor: C.gold,
  },
  icon: {
    color: C.dim,
    fontSize: 23,
    fontWeight: "900",
  },
  label: {
    color: C.dim,
    fontSize: 9,
    fontWeight: "700",
  },
  activeText: {
    color: C.gold,
  },
});
