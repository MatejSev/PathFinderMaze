"""
core/engine.py — základní blackjack engine (karty, box, hodnota ruky,
basic strategy). Sdílené všemi hráči i prostředím.

Pravidla: 6 balíčků, penetrace 75 %, S17 (volitelně H17), blackjack 3:2,
double na 2 karty, split, insurance. Ranky 0..12 = 2..10,J,Q,K,A.
"""

import numpy as np

RANK_VALUE = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])
HILO_TAG   = np.array([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1])
ACE = 12

A_STAND, A_HIT, A_DOUBLE, A_SPLIT = 0, 1, 2, 3


def hand_value(cards):
    """Vrátí (hodnota, is_soft) pro seznam rank-indexů."""
    total = int(RANK_VALUE[cards].sum())
    aces = int(np.sum(np.array(cards) == ACE))
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total, (aces > 0 and total <= 21)


class Shoe:
    """Box s několika balíčky, Hi-Lo running countem a cut card."""

    def __init__(self, num_decks=6, penetration=0.75, rng=None):
        self.num_decks = num_decks
        self.total = 52 * num_decks
        self.cut = int(self.total * penetration)
        self.rng = rng or np.random.default_rng()
        self.counts = None
        self.running = 0
        self.dealt = 0
        self.shuffle()

    def shuffle(self):
        self.counts = np.full(13, 4 * self.num_decks, dtype=np.int64)
        self.running = 0
        self.dealt = 0

    def needs_shuffle(self):
        return self.dealt >= self.cut

    def draw(self):
        total = self.counts.sum()
        if total == 0:
            self.shuffle()
            total = self.counts.sum()
        r = self.rng.integers(total)
        rank = int(np.searchsorted(np.cumsum(self.counts), r, side='right'))
        self.counts[rank] -= 1
        self.dealt += 1
        self.running += HILO_TAG[rank]
        return rank

    def decks_left(self):
        return max((self.total - self.dealt) / 52.0, 0.25)

    def true_count(self):
        return self.running / self.decks_left()


# ---------------------------------------------------------------------------
# Basic strategy (6D, S17, DAS) — přesná tabulka
# ---------------------------------------------------------------------------

def basic_strategy(cards, dealer_up_rank, legal_mask):
    """Vrátí akci (A_STAND/HIT/DOUBLE/SPLIT) dle basic strategy.
    legal_mask: pole bool délky 4."""
    up = int(RANK_VALUE[dealer_up_rank])
    val, soft = hand_value(cards)
    n = len(cards)
    can_double = legal_mask[A_DOUBLE]
    can_split = legal_mask[A_SPLIT]

    def D(fallback):
        return A_DOUBLE if can_double else fallback

    if can_split and n == 2 and RANK_VALUE[cards[0]] == RANK_VALUE[cards[1]]:
        pv = int(RANK_VALUE[cards[0]])
        if cards[0] == ACE:
            return A_SPLIT
        if pv == 10:
            return A_STAND
        if pv == 9:
            return A_SPLIT if up in (2, 3, 4, 5, 6, 8, 9) else A_STAND
        if pv == 8:
            return A_SPLIT
        if pv == 7:
            return A_SPLIT if up <= 7 else A_HIT
        if pv == 6:
            return A_SPLIT if up <= 6 else A_HIT
        if pv == 5:
            return D(A_HIT) if up <= 9 else A_HIT
        if pv == 4:
            return A_SPLIT if up in (5, 6) else A_HIT
        if pv in (2, 3):
            return A_SPLIT if up <= 7 else A_HIT

    if soft:
        if val >= 20:
            return A_STAND
        if val == 19:
            return D(A_STAND) if up == 6 else A_STAND
        if val == 18:
            if up in (2, 3, 4, 5, 6):
                return D(A_STAND)
            if up in (7, 8):
                return A_STAND
            return A_HIT
        if val == 17:
            return D(A_HIT) if up in (3, 4, 5, 6) else A_HIT
        if val in (15, 16):
            return D(A_HIT) if up in (4, 5, 6) else A_HIT
        if val in (13, 14):
            return D(A_HIT) if up in (5, 6) else A_HIT
        return A_HIT

    if val >= 17:
        return A_STAND
    if val >= 13:
        return A_STAND if up <= 6 else A_HIT
    if val == 12:
        return A_STAND if up in (4, 5, 6) else A_HIT
    if val == 11:
        return D(A_HIT)
    if val == 10:
        return D(A_HIT) if up <= 9 else A_HIT
    if val == 9:
        return D(A_HIT) if up in (3, 4, 5, 6) else A_HIT
    return A_HIT
