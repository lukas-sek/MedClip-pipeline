import React from "react";
import { ScrollView, StyleSheet, Text, TouchableOpacity } from "react-native";

export const CATEGORIES = [
  "All",
  "Jackets",
  "Pants",
  "Shirts",
  "Shoes",
  "Hats",
  "Accessories",
] as const;

export type Category = (typeof CATEGORIES)[number];

interface Props {
  selected: Category;
  onSelect: (category: Category) => void;
}

export function CategoryChips({
  selected,
  onSelect,
}: Props): React.ReactElement {
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      style={styles.scroll}
      contentContainerStyle={styles.content}
    >
      {CATEGORIES.map((cat) => {
        const isActive = cat === selected;
        return (
          <TouchableOpacity
            key={cat}
            activeOpacity={0.7}
            style={[styles.chip, isActive && styles.chipActive]}
            onPress={() => onSelect(cat)}
          >
            <Text style={[styles.label, isActive && styles.labelActive]}>
              {cat}
            </Text>
          </TouchableOpacity>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 0,
  },
  content: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    gap: 8,
  },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 20,
    backgroundColor: "#ede8e1",
    borderWidth: 1.5,
    borderColor: "transparent",
  },
  chipActive: {
    backgroundColor: "#c4867a",
    borderColor: "#c4867a",
  },
  label: {
    fontSize: 13,
    fontWeight: "500",
    color: "#8a7570",
    letterSpacing: 0.3,
  },
  labelActive: {
    color: "#fff",
    fontWeight: "700",
  },
});
