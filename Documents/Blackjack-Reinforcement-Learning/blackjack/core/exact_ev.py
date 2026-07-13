"""
dealer_exact.py — přesné rozdělení konečných výsledků krupiéra.

Composition-dependent: bere pravděpodobnosti karet z aktuálního složení
balíčku (bez náhrady během ruky). Žádný šum — čistá pravděpodobnost.

Reprezentace hodnoty karty: 10 kategorií (index 0..9):
  0..7 = karty 2..9
  8    = desítky (10, J, Q, K)   — 16 kusů na balíček
  9    = eso (A)                 — 4 kusy na balíček

Ruka se sleduje jako (total, is_soft), kde total je nejlepší (ne-bustující)
součet a is_soft znamená, že jedno eso se počítá jako 11.

Self-test dole ověřuje bust-rate krupiéra proti známým hodnotám.
"""

import numpy as np
from functools import lru_cache

# body jednotlivých kategorií
POINTS = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
# počet kusů každé kategorie na JEDEN balíček
PER_DECK = np.array([4, 4, 4, 4, 4, 4, 4, 4, 16, 4])


def add_card(total, soft, pts):
    """Přidá kartu o hodnotě pts do ruky (total, soft). Vrátí (total, soft)."""
    total += pts
    if pts == 11:
        soft = True
    if total > 21 and soft:
        total -= 10          # eso z 11 na 1
        soft = False
    return total, soft


class DealerExact:
    def __init__(self, h17=False):
        self.h17 = h17

    def outcome_dist(self, up_pts, counts):
        """Rozdělení konečných výsledků krupiéra začínajícího up-kartou o
        hodnotě up_pts, z balíčku `counts` (10 kategorií, už BEZ up-karty).
        Vrací np.array [P17, P18, P19, P20, P21, Pbust] (bez BJ – to řeší
        volající zvlášť; zde up karta není BJ situace)."""
        counts = counts.astype(np.float64)

        memo = {}

        def rec(total, soft, cnts_key, cnts):
            # krupiérovo pravidlo
            if total > 21:
                return np.array([0, 0, 0, 0, 0, 1.0])
            if total >= 17:
                if total == 17 and soft and self.h17:
                    pass  # táhne dál
                else:
                    idx = total - 17
                    out = np.zeros(6)
                    out[idx] = 1.0
                    return out
            key = (total, soft, cnts_key)
            if key in memo:
                return memo[key]
            tot_cards = cnts.sum()
            if tot_cards <= 0:
                # nemá z čeho táhnout: ber jako stojí na total (nemělo by nastat)
                out = np.zeros(6)
                out[min(max(total - 17, 0), 4)] = 1.0
                return out
            res = np.zeros(6)
            for ci in range(10):
                c = cnts[ci]
                if c <= 0:
                    continue
                p = c / tot_cards
                nc = cnts.copy()
                nc[ci] -= 1
                nt, ns = add_card(total, soft, POINTS[ci])
                res += p * rec(nt, ns, tuple(nc.astype(int)), nc)
            memo[key] = res
            return res

        soft0 = (up_pts == 11)
        return rec(up_pts, soft0, tuple(counts.astype(int)), counts)


# ---------------------------------------------------------------------------
# Self-test: bust rate krupiéra podle up-karty (známé hodnoty, S17, ~inf balíček)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    de = DealerExact(h17=False)
    # velký balíček ~ nekonečný (aproximace pravděpodobností kolody)
    big = PER_DECK.astype(float) * 1000
    print("Bust-rate krupiéra podle up-karty (S17):")
    print("  up | P(bust) | očekávané (přibl.)")
    expected = {2: .354, 3: .375, 4: .402, 5: .429, 6: .421,
                7: .262, 8: .245, 9: .231, 10: .214, 11: .115}
    for up in range(2, 12):
        counts = big.copy()
        # odeber up-kartu z balíčku
        if up == 10:
            counts[8] -= 1
        elif up == 11:
            counts[9] -= 1
        else:
            counts[up - 2] -= 1
        dist = de.outcome_dist(up, counts)
        pbust = dist[5]
        print(f"  {up:>2} | {pbust:.3f}   | {expected[up]:.3f}")
    print("\nMá-li to sedět (±0.005), krupiérův engine je správný.")


# ===== Round EV =====
BJ_PAYOUT = 1.5


class RoundEV:
    def __init__(self, h17=False):
        self.h17 = h17

    def compute(self, counts):
        counts = np.asarray(counts, dtype=np.float64)
        tot = counts.sum()
        if tot < 4:
            return -0.005
        p = tuple(counts / tot)

        @lru_cache(maxsize=None)
        def dealer_rec(total, soft):
            if total > 21:
                return (0., 0., 0., 0., 0., 1.)
            if total >= 17:
                if total == 17 and soft and self.h17:
                    pass
                else:
                    out = [0.] * 6
                    out[total - 17] = 1.
                    return tuple(out)
            res = [0.] * 6
            for ci in range(10):
                pc = p[ci]
                if pc <= 0:
                    continue
                nt, ns = _add(total, soft, POINTS[ci])
                sub = dealer_rec(nt, ns)
                for j in range(6):
                    res[j] += pc * sub[j]
            return tuple(res)

        def stand_ev(ptotal, up_pts, no_bj=False):
            dist = dealer_rec(up_pts, up_pts == 11)
            dist = list(dist)
            if no_bj:
                # odečti pravděpodobnost dealer-BJ z '21' kýble a přenormuj,
                # protože dealer-BJ je řešen zvlášť (peek) — hráč sem vstupuje
                # jen když dealer BJ NEMÁ.
                pbj = _dealer_bj_prob(up_pts, p)
                if pbj > 0 and dist[4] >= pbj:
                    dist[4] -= pbj
                    s = sum(dist)
                    if s > 0:
                        dist = [x / s for x in dist]
            ev = 0.0
            dtot = (17, 18, 19, 20, 21)
            for i in range(5):
                pr = dist[i]
                if pr <= 0:
                    continue
                if ptotal > dtot[i]:
                    ev += pr
                elif ptotal < dtot[i]:
                    ev -= pr
            ev += dist[5]
            return ev

        @lru_cache(maxsize=None)
        def play_ev(ptotal, psoft, up_pts, can_double):
            if ptotal > 21:
                return -1.0
            no_bj = up_pts in (10, 11)
            best = stand_ev(ptotal, up_pts, no_bj)
            hv = 0.0
            for ci in range(10):
                pc = p[ci]
                if pc <= 0:
                    continue
                nt, ns = _add(ptotal, psoft, POINTS[ci])
                hv += pc * play_ev(nt, ns, up_pts, False)
            if hv > best:
                best = hv
            if can_double:
                dv = 0.0
                for ci in range(10):
                    pc = p[ci]
                    if pc <= 0:
                        continue
                    nt, ns = _add(ptotal, psoft, POINTS[ci])
                    if nt > 21:
                        dv += pc * -2.0
                    else:
                        dv += pc * 2.0 * stand_ev(nt, up_pts, no_bj)
                if dv > best:
                    best = dv
            return best

        ev = 0.0
        for du in range(10):
            pdu = p[du]
            if pdu <= 0:
                continue
            up_pts = POINTS[du]
            dbj = _dealer_bj_prob(up_pts, p)
            for pa in range(10):
                ppa = p[pa]
                if ppa <= 0:
                    continue
                for pb in range(10):
                    ppb = p[pb]
                    if ppb <= 0:
                        continue
                    prob = pdu * ppa * ppb
                    ptotal, psoft = _two_card(POINTS[pa], POINTS[pb])
                    if ptotal == 21:      # hráčův blackjack
                        ev += prob * (BJ_PAYOUT * (1 - dbj))
                    else:
                        ev_play = play_ev(ptotal, psoft, up_pts, True)
                        ev += prob * (dbj * (-1.0) + (1 - dbj) * ev_play)
        return ev


def _add(total, soft, pts):
    total += pts
    if pts == 11:
        soft = True
    if total > 21 and soft:
        total -= 10
        soft = False
    return total, soft


def _two_card(p1, p2):
    total = p1 + p2
    soft = (p1 == 11 or p2 == 11)
    if total > 21 and soft:
        total -= 10
    return total, soft


def _dealer_bj_prob(up_pts, p):
    if up_pts == 11:
        return p[8]
    if up_pts == 10:
        return p[9]
    return 0.0


