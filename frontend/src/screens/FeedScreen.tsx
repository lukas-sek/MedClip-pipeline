/**
 * FeedScreen — the main swipe deck screen.
 *
 * Responsibilities:
 *  - Loads the initial 20-item batch from GET /feed on mount.
 *  - Renders a horizontal CategoryChips bar to filter by garment category.
 *  - Renders a swipeable card deck using react-native-deck-swiper.
 *  - Fires POST /swipe on each gesture (fire-and-forget, no UI block).
 *  - When 5 cards remain, prefetches the next batch in the background
 *    and appends it to the deck — guaranteeing zero-latency infinite scroll.
 *  - Shows a brief toast ("Liked!" / "Passed") after each swipe.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Animated,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import Swiper from "react-native-deck-swiper";

import { type FeedItem, getFeed, swipe } from "../api/client";
import {
  type Category,
  CategoryChips,
} from "../components/CategoryChips";
import { SwipeCard } from "../components/SwipeCard";

const PREFETCH_THRESHOLD = 5;
const TOAST_DURATION_MS = 900;

interface FeedScreenProps {
  userId: string;
}

export function FeedScreen({ userId }: FeedScreenProps): React.ReactElement {
  const [cards, setCards] = useState<FeedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [prefetching, setPrefetching] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<Category>("All");
  // Incremented on every category switch to force-remount the Swiper and
  // avoid stale card index / gesture state from the previous deck.
  const [deckKey, setDeckKey] = useState(0);
  const toastOpacity = useRef(new Animated.Value(0)).current;
  const swiperRef = useRef<Swiper<FeedItem>>(null);

  // -------------------------------------------------------------------------
  // Feed loading
  // -------------------------------------------------------------------------
  const loadFeed = useCallback(
    (category: Category): void => {
      setLoading(true);
      setCards([]);
      getFeed(userId, category)
        .then((items) => setCards(items))
        .catch((err) => console.error("[FeedScreen] feed load failed:", err))
        .finally(() => setLoading(false));
    },
    [userId]
  );

  useEffect(() => {
    loadFeed(selectedCategory);
  }, [userId, selectedCategory, loadFeed]);

  // -------------------------------------------------------------------------
  // Category selection
  // -------------------------------------------------------------------------
  const handleCategoryChange = useCallback((category: Category): void => {
    setSelectedCategory(category);
    setDeckKey((k) => k + 1);
  }, []);

  // -------------------------------------------------------------------------
  // Toast helper
  // -------------------------------------------------------------------------
  const showToast = useCallback(
    (msg: string): void => {
      setToastMsg(msg);
      toastOpacity.setValue(1);
      Animated.timing(toastOpacity, {
        toValue: 0,
        duration: TOAST_DURATION_MS,
        delay: 400,
        useNativeDriver: true,
      }).start(() => setToastMsg(null));
    },
    [toastOpacity]
  );

  // -------------------------------------------------------------------------
  // Prefetch logic
  // -------------------------------------------------------------------------
  const maybePrefetch = useCallback(
    (remainingCount: number): void => {
      if (remainingCount <= PREFETCH_THRESHOLD && !prefetching) {
        setPrefetching(true);
        getFeed(userId, selectedCategory)
          .then((newItems) => {
            setCards((prev) => {
              // Deduplicate by id before appending.
              const existingIds = new Set(prev.map((c) => c.id));
              const fresh = newItems.filter((i) => !existingIds.has(i.id));
              return [...prev, ...fresh];
            });
          })
          .catch((err) =>
            console.warn("[FeedScreen] prefetch failed:", err)
          )
          .finally(() => setPrefetching(false));
      }
    },
    [userId, selectedCategory, prefetching]
  );

  // -------------------------------------------------------------------------
  // Swipe handlers
  // -------------------------------------------------------------------------
  const handleSwiped = useCallback(
    (cardIndex: number, direction: "right" | "left"): void => {
      const item = cards[cardIndex];
      if (!item) return;

      // Fire-and-forget — UI must not block.
      // Always uses item.category from the payload so vectors stay accurate
      // regardless of which category filter view is active.
      swipe(userId, item, direction);

      if (direction === "right") {
        showToast("Liked! ♥");
      } else {
        showToast("Passed");
      }

      const remaining = cards.length - cardIndex - 1;
      maybePrefetch(remaining);
    },
    [cards, userId, showToast, maybePrefetch]
  );

  const handleSwipedRight = useCallback(
    (cardIndex: number): void => handleSwiped(cardIndex, "right"),
    [handleSwiped]
  );

  const handleSwipedLeft = useCallback(
    (cardIndex: number): void => handleSwiped(cardIndex, "left"),
    [handleSwiped]
  );

  const handleSwipedAll = useCallback((): void => {
    getFeed(userId, selectedCategory)
      .then((items) => setCards((prev) => [...prev, ...items]))
      .catch((err) =>
        console.warn("[FeedScreen] onSwipedAll fetch failed:", err)
      );
  }, [userId, selectedCategory]);

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>MedClip-pipeline</Text>
        <Text style={styles.headerSub}>vintage · curated · yours</Text>
      </View>

      {/* Category filter */}
      <CategoryChips
        selected={selectedCategory}
        onSelect={handleCategoryChange}
      />

      {/* Card deck */}
      <View style={styles.deckContainer}>
        {loading ? (
          <ActivityIndicator size="large" color="#c4867a" />
        ) : cards.length > 0 ? (
          <Swiper
            key={deckKey}
            ref={swiperRef}
            cards={cards}
            renderCard={(item) =>
              item ? <SwipeCard item={item} /> : <View />
            }
            onSwipedRight={handleSwipedRight}
            onSwipedLeft={handleSwipedLeft}
            onSwipedAll={handleSwipedAll}
            cardIndex={0}
            backgroundColor="transparent"
            stackSize={3}
            stackSeparation={12}
            animateOverlayLabelsOpacity
            overlayLabels={{
              right: {
                title: "LIKE",
                style: {
                  label: {
                    backgroundColor: "transparent",
                    color: "#4caf50",
                    fontSize: 32,
                    fontWeight: "800",
                    borderWidth: 3,
                    borderColor: "#4caf50",
                    borderRadius: 8,
                    padding: 8,
                  },
                  wrapper: {
                    flexDirection: "column",
                    alignItems: "flex-start",
                    justifyContent: "flex-start",
                    marginTop: 30,
                    marginLeft: 20,
                  },
                },
              },
              left: {
                title: "PASS",
                style: {
                  label: {
                    backgroundColor: "transparent",
                    color: "#f44336",
                    fontSize: 32,
                    fontWeight: "800",
                    borderWidth: 3,
                    borderColor: "#f44336",
                    borderRadius: 8,
                    padding: 8,
                  },
                  wrapper: {
                    flexDirection: "column",
                    alignItems: "flex-end",
                    justifyContent: "flex-start",
                    marginTop: 30,
                    marginRight: 20,
                  },
                },
              },
            }}
          />
        ) : (
          <View style={styles.centered}>
            <Text style={styles.emptyText}>No more items right now.</Text>
            <Text style={styles.emptySubText}>Check back soon!</Text>
          </View>
        )}
      </View>

      {/* Action buttons */}
      <View style={styles.actions}>
        <TouchableOpacity
          style={[styles.actionBtn, styles.dislikeBtn]}
          onPress={() => swiperRef.current?.swipeLeft()}
          activeOpacity={0.8}
        >
          <Text style={styles.actionIcon}>✕</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionBtn, styles.likeBtn]}
          onPress={() => swiperRef.current?.swipeRight()}
          activeOpacity={0.8}
        >
          <Text style={styles.actionIcon}>♥</Text>
        </TouchableOpacity>
      </View>

      {/* Toast */}
      {toastMsg && (
        <Animated.View style={[styles.toast, { opacity: toastOpacity }]}>
          <Text style={styles.toastText}>{toastMsg}</Text>
        </Animated.View>
      )}

      {/* Background prefetch indicator */}
      {prefetching && (
        <ActivityIndicator
          size="small"
          color="#c4867a"
          style={styles.prefetchIndicator}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f2ede4",
    alignItems: "center",
  },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  header: {
    paddingTop: 60,
    paddingBottom: 8,
    alignItems: "center",
  },
  headerTitle: {
    fontSize: 13,
    fontWeight: "700",
    letterSpacing: 4,
    color: "#5a4a42",
    textTransform: "uppercase",
  },
  headerSub: {
    fontSize: 11,
    color: "#a08c84",
    letterSpacing: 1.5,
    marginTop: 2,
  },
  deckContainer: {
    flex: 1,
    width: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
  actions: {
    flexDirection: "row",
    justifyContent: "center",
    gap: 40,
    paddingBottom: 48,
    paddingTop: 12,
  },
  actionBtn: {
    width: 64,
    height: 64,
    borderRadius: 32,
    alignItems: "center",
    justifyContent: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.12,
    shadowRadius: 8,
    elevation: 6,
  },
  dislikeBtn: {
    backgroundColor: "#fff",
    borderWidth: 2,
    borderColor: "#f44336",
  },
  likeBtn: {
    backgroundColor: "#fff",
    borderWidth: 2,
    borderColor: "#4caf50",
  },
  actionIcon: {
    fontSize: 26,
  },
  toast: {
    position: "absolute",
    top: "45%",
    alignSelf: "center",
    backgroundColor: "rgba(0,0,0,0.65)",
    paddingHorizontal: 24,
    paddingVertical: 10,
    borderRadius: 24,
  },
  toastText: {
    color: "#fff",
    fontSize: 15,
    fontWeight: "600",
    letterSpacing: 0.5,
  },
  emptyText: {
    fontSize: 18,
    color: "#5a4a42",
    fontWeight: "600",
  },
  emptySubText: {
    fontSize: 14,
    color: "#a08c84",
    marginTop: 6,
  },
  prefetchIndicator: {
    position: "absolute",
    bottom: 120,
  },
});
