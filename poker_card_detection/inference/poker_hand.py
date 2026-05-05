from collections import Counter
from typing import NamedTuple

RANK_VALUES = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


class Card(NamedTuple):
    rank: str
    suit: str


def parse_card(label: str) -> Card:
    suit = label[-1]
    rank = label[:-1]
    return Card(rank=rank, suit=suit)


def classify_hand(labels: list[str]) -> str:
    if len(labels) < 5:
        return f"Incomplete hand ({len(labels)} cards detected)"

    cards = [parse_card(label) for label in labels[:5]]
    ranks = [card.rank for card in cards]
    suits = [card.suit for card in cards]
    values = sorted(RANK_VALUES.get(rank, 0) for rank in ranks)

    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)

    is_flush = len(set(suits)) == 1
    is_straight = len(set(values)) == 5 and (max(values) - min(values) == 4)

    if set(values) == {14, 2, 3, 4, 5}:
        is_straight = True

    if is_straight and is_flush:
        return (
            "Royal Flush"
            if min(v for v in values if v != 1) >= 10
            else "Straight Flush"
        )
    if counts[0] == 4:
        return "Four of a Kind"
    if counts[0] == 3 and counts[1] == 2:
        return "Full House"
    if is_flush:
        return "Flush"
    if is_straight:
        return "Straight"
    if counts[0] == 3:
        return "Three of a Kind"
    if counts[0] == 2 and counts[1] == 2:
        return "Two Pair"
    if counts[0] == 2:
        return "One Pair"
    return "High Card"
