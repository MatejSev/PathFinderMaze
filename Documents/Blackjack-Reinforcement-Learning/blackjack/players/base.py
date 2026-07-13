"""
players/base.py — společné rozhraní všech hráčů.

Každý hráč (počítací metoda i RL agent) implementuje:
  - bet(bet_obs, env) -> int   : velikost sázky 1..MAX_BET_UNITS
  - insurance(env) -> bool     : vzít pojištění?  (default: ne)
  - play(cards, up, mask, env) -> akce : jak hrát ruku (default: basic strategy)

Počítací metody se LIŠÍ POUZE v bet() — hrají všechny basic strategy, aby
porovnání izolovalo KVALITU SÁZENÍ (počítání), ne hraní. RL agent může mít
vlastní i play(), ale ve výchozím srovnání také hraje basic strategy, aby
byl fér vůči ostatním.

Sázecí pravidlo `bet_from_true_count` je sdílené, aby se metody lišily jen
tím, JAK count počítají, ne jak z něj dělají sázku.
"""

from blackjack.core.engine import basic_strategy, RANK_VALUE

MIN_BET = 1
MAX_BET = 20

# realistické sázení: sázka roste PLYNULE se sílou countu, jako u skutečného
# hráče (ne skokově min/max). Parametry odpovídají běžnému counting spreadu.
BREAK_EVEN_TC = 1.0     # pod tímto TC sází hráč minimum
RAMP = 2.0              # o kolik jednotek přidá na každý bod TC nad break-even


def bet_from_edge_strength(strength, min_u=MIN_BET, max_u=MAX_BET,
                           break_even=BREAK_EVEN_TC, ramp=RAMP):
    """Plynulá sázka podle síly výhody (typicky true count, u Perfectu
    ekvivalentní TC z přesného EV). Čím silnější count, tím větší sázka,
    postupně — jako reálný hráč, který zvyšuje sázku s rostoucím countem.

      strength <= break_even   -> minimum
      výš                      -> min + ramp*(strength-break_even), ořez na max
    """
    if strength <= break_even:
        return min_u
    u = min_u + ramp * (strength - break_even)
    return int(min(max_u, max(min_u, round(u))))


# zpětně kompatibilní alias
def bet_from_true_count(tc, min_u=MIN_BET, max_u=MAX_BET):
    return bet_from_edge_strength(tc, min_u, max_u)


class Player:
    """Základní hráč: sází vždy minimum, hraje basic strategy. Slouží jako
    baseline a jako předek pro ostatní."""

    name = "Base"

    def bet(self, bet_obs, env):
        return MIN_BET

    def insurance(self, env):
        return False

    def play(self, cards, dealer_up_rank, legal_mask, env):
        return basic_strategy(cards, dealer_up_rank, legal_mask)
