import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { TrackedBetsProvider } from "../components/TrackedBetsContext";

export default function Layout() {
  return (
    <TrackedBetsProvider>
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
