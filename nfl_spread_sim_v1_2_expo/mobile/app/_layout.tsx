import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { TrackedBetsProvider } from "../components/TrackedBetsContext";
import { AutoUpdater } from "../components/AutoUpdater";

export default function Layout() {
  return (
    <TrackedBetsProvider>
      <AutoUpdater />

      <StatusBar style="light" />

      <Stack
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: "#050708" },
          animation: "fade",
        }}
      />
    </TrackedBetsProvider>
  );
}
