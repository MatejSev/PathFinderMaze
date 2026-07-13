"""blackjack.players — hráči se společným rozhraním (viz base.Player).

Počítací metody a Perfect jsou pevní (deterministickí) hráči. RL agent
(PPO), Q-learning, DQN a Stealth se učí. Všichni sázejí plynule podle síly
countu a ve výchozím srovnání hrají basic strategy, takže se porovnává
kvalita SÁZENÍ.
"""

from blackjack.players.base import Player
from blackjack.players.counting import HiLoPlayer, KOPlayer, HiOpt2Player
from blackjack.players.perfect import PerfectPlayer
from blackjack.players.rl_agent import RLAgentPlayer
from blackjack.players.qlearning import QLearningPlayer
from blackjack.players.dqn import DQNPlayer
from blackjack.players.stealth import StealthPlayer

__all__ = ["Player", "HiLoPlayer", "KOPlayer", "HiOpt2Player",
           "PerfectPlayer", "RLAgentPlayer", "QLearningPlayer",
           "DQNPlayer", "StealthPlayer"]
