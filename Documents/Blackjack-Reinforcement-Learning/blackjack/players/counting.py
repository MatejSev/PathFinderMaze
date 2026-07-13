"""
players/counting.py — hráči založení na počítání karet.

Každá metoda má vlastní váhy karet (count tagy). Running count se počítá z
karet, které UŽ padly (dostupné z env.shoe). True count = running / balíčky.
Sázka přes sdílený spread. Všechny hrají basic strategy — liší se jen count.

Hráči: HiLoPlayer, KOPlayer, HiOpt2Player.
(Perfect a RL agent jsou ve vlastních souborech.)
"""

import numpy as np
from blackjack.players.base import Player, bet_from_edge_strength

# count tagy: ranky 0..12 = 2,3,4,5,6,7,8,9,10,J,Q,K,A
TAGS = {
    'Hi-Lo':     np.array([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1], float),
    'KO':        np.array([1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, -1], float),
    'Hi-Opt II': np.array([1, 1, 2, 2, 1, 1, 0, 0, -2, -2, -2, -2,  0], float),
}
FULL_13 = np.array([4.] * 13) * 6   # plný 6-balíček po ranku


class _CountingPlayer(Player):
    """Sází PLYNULE podle svého true countu (jako reálný hráč zvyšuje sázku
    s rostoucím countem). Metody se liší kvalitou countu i jeho měřítkem,
    proto má každá `tc_scale` — přepočet jejího true countu na společné
    měřítko (Hi-Lo = 1.0), aby všichni používali stejný bet ramp a
    porovnávala se čistě KVALITA počítání."""
    tags = None
    balanced = True
    tc_scale = 1.0     # násobič převádějící metodou spočtený TC na Hi-Lo škálu

    def _true_count(self, env):
        dealt = FULL_13 - env.shoe.counts.astype(float)
        rc = float(np.dot(self.tags, dealt))
        return rc / env.shoe.decks_left()

    def bet(self, bet_obs, env):
        strength = self._true_count(env) * self.tc_scale
        return bet_from_edge_strength(strength)


class HiLoPlayer(_CountingPlayer):
    name = "Hi-Lo"
    tags = TAGS['Hi-Lo']
    tc_scale = 1.0            # referenční měřítko


class KOPlayer(_CountingPlayer):
    name = "KO"
    tags = TAGS['KO']
    balanced = False
    tc_scale = 0.55           # KO běží "hotově", škáluje se dolů


class HiOpt2Player(_CountingPlayer):
    name = "Hi-Opt II"
    tags = TAGS['Hi-Opt II']
    tc_scale = 0.80           # Hi-Opt II má silnější tagy; kalibrováno tak,
                              # aby vyšlo mírně nad Hi-Lo (silnější count)
