/**
 * Typed API client for the MedClip-pipeline FastAPI backend.
 *
 * Base URL is read from the EXPO_PUBLIC_API_URL environment variable,
 * falling back to localhost for local development.
 */

const BASE_URL =
  (process.env.EXPO_PUBLIC_API_URL ?? "http://localhost:8000").replace(
    /\/$/,
    ""
  );

// ---------------------------------------------------------------------------
// Shared types (mirror Phase 3 Pydantic schemas)
// ---------------------------------------------------------------------------

export interface FeedItem {
  id: number;
  title: string;
  price: string;
  currency: string;
  brand: string | null;
  size: string | null;
  image_url: string;
  category: string;
  vector: number[];
  source: "exploit" | "explore";
}

export type SwipeDirection = "right" | "left";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function apiFetch<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      "bypass-tunnel-reminder": "true",
    },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`API ${options?.method ?? "GET"} ${path} → ${response.status}`);
  }
  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

/** Register a new anonymous user and return their UUID. */
export async function createUser(): Promise<{ user_id: string }> {
  return apiFetch<{ user_id: string }>("/users", { method: "POST" });
}

/**
 * Verify that a user UUID still exists in the backend.
 * Returns true if the user exists, false if 404 (e.g. after a DB reset).
 */
export async function verifyUser(userId: string): Promise<boolean> {
  try {
    await apiFetch<{ user_id: string }>(`/users/${userId}`);
    return true;
  } catch {
    return false;
  }
}

/**
 * Record a swipe action. Fire-and-forget — callers should NOT await this
 * so the UI never blocks on network latency.
 */
export function swipe(
  userId: string,
  item: FeedItem,
  direction: SwipeDirection
): void {
  const action = direction === "right" ? "like" : "dislike";
  apiFetch("/swipe", {
    method: "POST",
    body: JSON.stringify({
      user_id: userId,
      item_id: item.id,
      category: item.category,
      swiped_item_vector: item.vector,
      action,
    }),
  }).catch((err) => console.warn("[swipe] background request failed:", err));
}

/** Fetch the next batch of 20 personalised items for the user. */
export async function getFeed(
  userId: string,
  category?: string
): Promise<FeedItem[]> {
  const cat =
    category && category.toLowerCase() !== "all"
      ? category.toLowerCase()
      : undefined;
  const qs = cat ? `&category=${encodeURIComponent(cat)}` : "";
  const data = await apiFetch<{ user_id: string; items: FeedItem[] }>(
    `/feed?user_id=${userId}${qs}`
  );
  return data.items;
}
