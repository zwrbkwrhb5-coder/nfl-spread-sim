import { useCallback, useEffect, useRef } from "react";
import { AppState } from "react-native";
import * as Updates from "expo-updates";

const MIN_CHECK_INTERVAL_MS = 5 * 60 * 1000;

export function AutoUpdater() {
  const checkingRef = useRef(false);
  const lastCheckRef = useRef(0);

  const checkForUpdate = useCallback(async () => {
    if (__DEV__ || !Updates.isEnabled || checkingRef.current) {
      return;
    }

    const now = Date.now();

    if (now - lastCheckRef.current < MIN_CHECK_INTERVAL_MS) {
      return;
    }

    checkingRef.current = true;
    lastCheckRef.current = now;

    try {
      const update = await Updates.checkForUpdateAsync();

      if (update.isAvailable) {
        console.log("NFL SIM update available. Downloading...");

        await Updates.fetchUpdateAsync();

        console.log("NFL SIM update downloaded. Reloading...");

        await Updates.reloadAsync();
      }
    } catch (error) {
      console.log("NFL SIM update check skipped:", error);
    } finally {
      checkingRef.current = false;
    }
  }, []);

  useEffect(() => {
    void checkForUpdate();

    const subscription = AppState.addEventListener("change", (state) => {
      if (state === "active") {
        void checkForUpdate();
      }
    });

    return () => {
      subscription.remove();
    };
  }, [checkForUpdate]);

  return null;
}
