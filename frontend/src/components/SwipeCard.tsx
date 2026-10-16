/**
 * SwipeCard — renders a single fashion item card.
 *
 * Layout:
 *   - Full-bleed item image
 *   - Gradient overlay fading from transparent → dark at the bottom
 *   - Brand name (bold white), size badge, price badge over the gradient
 */

import { LinearGradient } from "expo-linear-gradient";
import React from "react";
import { Dimensions, Image, StyleSheet, Text, View } from "react-native";

import type { FeedItem } from "../api/client";

const { width: SCREEN_WIDTH } = Dimensions.get("window");

// Card is 90% of screen width; height is 4:5 ratio (portrait fashion photos)
const CARD_WIDTH = SCREEN_WIDTH * 0.9;
const CARD_HEIGHT = CARD_WIDTH * 1.25;

interface SwipeCardProps {
  item: FeedItem;
}

export function SwipeCard({ item }: SwipeCardProps): React.ReactElement {
  const priceLabel = item.price
    ? `${parseFloat(item.price).toFixed(2)} ${item.currency}`
    : null;

  return (
    <View style={styles.card}>
      <Image
        source={{ uri: item.image_url }}
        style={styles.image}
        resizeMode="cover"
      />

      {/* Bottom gradient overlay */}
      <LinearGradient
        colors={["transparent", "rgba(0,0,0,0.75)"]}
        style={styles.gradient}
      >
        <View style={styles.metaRow}>
          {/* Brand */}
          {item.brand ? (
            <Text style={styles.brand} numberOfLines={1}>
              {item.brand}
            </Text>
          ) : (
            <Text style={styles.brand} numberOfLines={1}>
              {item.title}
            </Text>
          )}

          <View style={styles.badgeRow}>
            {/* Size badge */}
            {item.size && (
              <View style={styles.badge}>
                <Text style={styles.badgeText}>{item.size}</Text>
              </View>
            )}

            {/* Price badge */}
            {priceLabel && (
              <View style={[styles.badge, styles.priceBadge]}>
                <Text style={styles.badgeText}>{priceLabel}</Text>
              </View>
            )}
          </View>
        </View>
      </LinearGradient>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    width: CARD_WIDTH,
    height: CARD_HEIGHT,
    borderRadius: 20,
    overflow: "hidden",
    backgroundColor: "#e8e0d5",
    // Subtle shadow for depth
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.18,
    shadowRadius: 16,
    elevation: 10,
  },
  image: {
    width: "100%",
    height: "100%",
  },
  gradient: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    height: CARD_HEIGHT * 0.4,
    justifyContent: "flex-end",
    paddingHorizontal: 18,
    paddingBottom: 18,
  },
  metaRow: {
    gap: 8,
  },
  brand: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "700",
    letterSpacing: 0.3,
  },
  badgeRow: {
    flexDirection: "row",
    gap: 8,
    flexWrap: "wrap",
  },
  badge: {
    backgroundColor: "rgba(255,255,255,0.22)",
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.35)",
  },
  priceBadge: {
    backgroundColor: "rgba(255,255,255,0.9)",
  },
  badgeText: {
    color: "#fff",
    fontSize: 12,
    fontWeight: "600",
  },
});
