/**
 * App root — bootstraps the session then renders FeedScreen.
 *
 * useSession handles UUID generation/retrieval and user registration.
 * While the session is initialising, a splash-style loading screen is shown.
 */

import { StatusBar } from "expo-status-bar";
import React from "react";
import { ActivityIndicator, StyleSheet, Text, View } from "react-native";

import { FeedScreen } from "./src/screens/FeedScreen";
import { useSession } from "./src/hooks/useSession";

export default function App(): React.ReactElement {
  const { userId, ready } = useSession();

  if (!ready) {
    return (
      <View style={styles.splash}>
        <Text style={styles.splashTitle}>MedClip-pipeline</Text>
        <ActivityIndicator
          size="small"
          color="#c4867a"
          style={{ marginTop: 24 }}
        />
        <StatusBar style="dark" />
      </View>
    );
  }

  if (!userId) {
    // Edge case: ready but no userId (storage failure).
    return (
      <View style={styles.splash}>
        <Text style={styles.errorText}>
          Failed to start session.{"\n"}Please restart the app.
        </Text>
        <StatusBar style="dark" />
      </View>
    );
  }

  return (
    <>
      <FeedScreen userId={userId} />
      <StatusBar style="dark" />
    </>
  );
}

const styles = StyleSheet.create({
  splash: {
    flex: 1,
    backgroundColor: "#f2ede4",
    alignItems: "center",
    justifyContent: "center",
  },
  splashTitle: {
    fontSize: 14,
    fontWeight: "700",
    letterSpacing: 5,
    color: "#5a4a42",
    textTransform: "uppercase",
  },
  errorText: {
    fontSize: 15,
    color: "#a08c84",
    textAlign: "center",
    lineHeight: 22,
  },
});
