/**
 * useSession — anonymous UUID session management.
 *
 * Behaviour:
 *  - First launch: generates a UUIDv4, persists it to AsyncStorage, calls
 *    POST /users to create the user row in PostgreSQL, then sets ready=true.
 *  - Subsequent launches: reads the stored UUID, skips registration,
 *    sets ready=true immediately.
 *
 * Returns { userId, ready } — FeedScreen waits until ready===true.
 */

import AsyncStorage from "@react-native-async-storage/async-storage";
import { useEffect, useState } from "react";
import * as Crypto from "expo-crypto";

import { createUser, verifyUser } from "../api/client";

const STORAGE_KEY = "MedClip-pipeline_user_id";

interface Session {
  userId: string | null;
  ready: boolean;
}

export function useSession(): Session {
  const [userId, setUserId] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function initSession(): Promise<void> {
      try {
        const stored = await AsyncStorage.getItem(STORAGE_KEY);

        if (stored) {
          const exists = await verifyUser(stored);
          if (exists) {
            if (!cancelled) {
              setUserId(stored);
              setReady(true);
            }
            return;
          }
          // Stored UUID no longer exists in backend (e.g. after a DB reset) — re-register.
          await AsyncStorage.removeItem(STORAGE_KEY);
        }

        // First launch — register with backend and persist the returned UUID.
        let newId: string;
        try {
          const res = await createUser();
          newId = res.user_id;
        } catch (err) {
          console.warn("[useSession] POST /users failed:", err);
          newId = Crypto.randomUUID(); // fallback: local UUID
        }

        await AsyncStorage.setItem(STORAGE_KEY, newId);

        if (!cancelled) {
          setUserId(newId);
          setReady(true);
        }
      } catch (err) {
        console.error("[useSession] Failed to initialise session:", err);
        if (!cancelled) setReady(true); // unblock UI even on storage failure
      }
    }

    initSession();
    return () => {
      cancelled = true;
    };
  }, []);

  return { userId, ready };
}
