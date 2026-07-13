"""
blackjack.config — centrální nastavení cest a herních pravidel.

Cesty k modelům se odvozují od kořene repozitáře, takže fungují bez ohledu
na to, odkud skript spustíš.
"""

import os

# kořen repozitáře = o dvě úrovně výš než tento soubor (blackjack/config.py)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


def model_path(name):
    """Absolutní cesta k modelu v adresáři models/."""
    return os.path.join(MODELS_DIR, name)


# výchozí názvy modelů
PPO_MODEL = model_path("ppo_agent.pt")
QLEARNING_MODEL = model_path("qlearning.npy")
DQN_MODEL = model_path("dqn.pt")


def stealth_model(stealth_level):
    return model_path(f"stealth_{int(stealth_level * 100)}.pt")


# herní pravidla (výchozí)
NUM_DECKS = 6
PENETRATION = 0.75
H17_DEFAULT = False
BLACKJACK_PAYOUT = 1.5
