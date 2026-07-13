"""
Blackjack – casino-věrná implementace v Pythonu.

Pravidla (odpovídají běžnému kasinovému Blackjacku):
  - 6 balíčků v boxu (shoe), 312 karet.
  - Míchá se, když se dojede k "cut card" umístěné cca 75 % hloubky boxu
    (tj. po projetí ~4,5 balíčku se na konci kola zamíchá).
  - Krupiér stojí na soft 17 (S17). Lze přepnout na H17.
  - Blackjack platí 3:2.
  - Insurance (pojištění) platí 2:1, nabízí se když krupiér ukazuje eso.
  - Double down na libovolné první dvě karty.
  - Split libovolného páru; max 4 ruce po splitu. Rozdělená esa dostanou
    1 kartu a nelze na ně hitovat. Split esa + 10 = 21, ne blackjack.
  - Double po splitu povolen (DAS).
  - 1 až 7 hráčů.
"""

import random
import sys


# ---------------------------------------------------------------------------
# Karty a hodnoty
# ---------------------------------------------------------------------------

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUITS = ['♠', '♥', '♦', '♣']

CARD_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11,
}


class Card:
    __slots__ = ('rank', 'suit')

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    @property
    def value(self):
        return CARD_VALUES[self.rank]

    def __str__(self):
        return f"{self.rank}{self.suit}"


# ---------------------------------------------------------------------------
# Shoe (box s balíčky) a míchání
# ---------------------------------------------------------------------------

class Shoe:
    """Box s několika balíčky karet a cut card, jako v kasinu."""

    def __init__(self, num_decks=6, penetration=0.75):
        self.num_decks = num_decks
        self.penetration = penetration
        self.cards = []
        self.cut_index = 0
        self.needs_shuffle = True
        self._build_and_shuffle()

    def _build_and_shuffle(self):
        self.cards = [Card(r, s) for _ in range(self.num_decks)
                      for s in SUITS for r in RANKS]
        random.shuffle(self.cards)
        total = len(self.cards)
        # Cut card se umístí za ~penetration boxu. Jakmile ji projedeme,
        # na konci kola se zamíchá (u kasin se dohraje běžící kolo).
        self.cut_index = int(total * self.penetration)
        self.needs_shuffle = False
        print(f"\n*** Box byl zamíchán ({self.num_decks} balíčků, "
              f"{total} karet). Cut card za {self.cut_index} kartami. ***")

    def deal(self):
        if not self.cards:
            self._build_and_shuffle()
        card = self.cards.pop()
        # Pokud jsme projeli cut card, zamícháme až po dohrání kola.
        dealt = (self.num_decks * 52) - len(self.cards)
        if dealt >= self.cut_index:
            self.needs_shuffle = True
        return card

    def maybe_reshuffle(self):
        """Volá se na konci kola. Zamíchá, pokud byla projeta cut card."""
        if self.needs_shuffle:
            self._build_and_shuffle()


# ---------------------------------------------------------------------------
# Ruka
# ---------------------------------------------------------------------------

class Hand:
    def __init__(self, bet=0):
        self.cards = []
        self.bet = bet
        self.stood = False
        self.doubled = False
        self.is_split_ace = False
        self.surrendered = False

    def add(self, card):
        self.cards.append(card)

    @property
    def value(self):
        total = sum(c.value for c in self.cards)
        aces = sum(1 for c in self.cards if c.rank == 'A')
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return total

    @property
    def is_soft(self):
        total = sum(c.value for c in self.cards)
        aces = sum(1 for c in self.cards if c.rank == 'A')
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return aces > 0 and total <= 21

    @property
    def is_bust(self):
        return self.value > 21

    @property
    def is_blackjack(self):
        return len(self.cards) == 2 and self.value == 21 and not self.is_split_ace

    @property
    def can_split(self):
        return (len(self.cards) == 2
                and self.cards[0].value == self.cards[1].value)

    def __str__(self):
        cards = ' '.join(str(c) for c in self.cards)
        soft = " (soft)" if self.is_soft and not self.is_bust else ""
        return f"[{cards}] = {self.value}{soft}"


# ---------------------------------------------------------------------------
# Hráč
# ---------------------------------------------------------------------------

class Player:
    def __init__(self, name, bankroll=1000):
        self.name = name
        self.bankroll = bankroll
        self.hands = []
        self.insurance = 0

    def reset_round(self):
        self.hands = []
        self.insurance = 0


# ---------------------------------------------------------------------------
# Vstupní pomocníci
# ---------------------------------------------------------------------------

def prompt_int(msg, lo, hi):
    while True:
        try:
            v = int(input(msg))
            if lo <= v <= hi:
                return v
            print(f"Zadej číslo mezi {lo} a {hi}.")
        except ValueError:
            print("Neplatný vstup, zadej číslo.")


def prompt_choice(msg, choices):
    choices = [c.lower() for c in choices]
    while True:
        v = input(msg).strip().lower()
        if v in choices:
            return v
        print(f"Vyber jednu z možností: {', '.join(choices)}")


def prompt_yes_no(msg):
    return prompt_choice(msg + " (a/n): ", ['a', 'n']) == 'a'


# ---------------------------------------------------------------------------
# Hra
# ---------------------------------------------------------------------------

class Blackjack:
    def __init__(self, players, shoe, dealer_hits_soft_17=False,
                 blackjack_payout=1.5):
        self.players = players
        self.shoe = shoe
        self.h17 = dealer_hits_soft_17
        self.bj_payout = blackjack_payout
        self.dealer = Hand()

    # -- výdej karet -------------------------------------------------------

    def initial_deal(self):
        self.dealer = Hand()
        for p in self.players:
            for h in p.hands:
                h.add(self.shoe.deal())
        self.dealer.add(self.shoe.deal())          # up card
        for p in self.players:
            for h in p.hands:
                h.add(self.shoe.deal())
        self.dealer.add(self.shoe.deal())          # hole card

    def show_table(self, hide_hole=True):
        print("\n" + "=" * 50)
        if hide_hole and len(self.dealer.cards) >= 2:
            print(f"Krupiér: [{self.dealer.cards[0]} ??]")
        else:
            print(f"Krupiér: {self.dealer}")
        for p in self.players:
            for i, h in enumerate(p.hands):
                tag = f" (ruka {i+1})" if len(p.hands) > 1 else ""
                print(f"{p.name}{tag}: {h}  sázka {h.bet}")
        print("=" * 50)

    # -- fáze sázek --------------------------------------------------------

    def betting(self):
        active = []
        for p in self.players:
            if p.bankroll <= 0:
                print(f"{p.name} nemá žádné žetony a vynechává.")
                continue
            print(f"\n{p.name}, máš {p.bankroll} žetonů.")
            bet = prompt_int(f"Tvoje sázka (1-{p.bankroll}): ", 1, p.bankroll)
            p.bankroll -= bet
            h = Hand(bet)
            p.hands = [h]
            active.append(p)
        return active

    # -- insurance ---------------------------------------------------------

    def offer_insurance(self, active):
        if self.dealer.cards[0].rank != 'A':
            return
        print("\nKrupiér ukazuje eso – nabízí se pojištění (insurance).")
        for p in active:
            main = p.hands[0]
            max_ins = min(main.bet // 2, p.bankroll)
            if max_ins <= 0:
                continue
            if prompt_yes_no(f"{p.name}, chceš pojištění (max {max_ins})?"):
                amt = prompt_int(f"Kolik (1-{max_ins}): ", 1, max_ins)
                p.bankroll -= amt
                p.insurance = amt

    def settle_insurance(self, active):
        dealer_bj = self.dealer.is_blackjack
        for p in active:
            if p.insurance > 0:
                if dealer_bj:
                    win = p.insurance * 2
                    p.bankroll += p.insurance + win  # vrátí vklad + 2:1
                    print(f"{p.name}: pojištění vyplaceno +{win}.")
                else:
                    print(f"{p.name}: pojištění prohráno -{p.insurance}.")
        return dealer_bj

    # -- tahy hráče --------------------------------------------------------

    def play_hand(self, player, hand):
        while True:
            if hand.is_split_ace:
                # rozdělená esa: jen 1 karta, konec
                return
            if hand.value == 21:
                hand.stood = True
                return
            if hand.is_bust:
                print(f"{player.name}: přetažení! {hand}")
                return

            options = ['h', 's']
            label = "[H]it, [S]tand"
            can_double = (len(hand.cards) == 2
                          and player.bankroll >= hand.bet)
            can_split = (hand.can_split
                         and player.bankroll >= hand.bet
                         and len(player.hands) < 4)
            if can_double:
                options.append('d')
                label += ", [D]ouble"
            if can_split:
                options.append('p')
                label += ", s[P]lit"

            print(f"\n{player.name}: {hand}")
            choice = prompt_choice(f"{label}: ", options)

            if choice == 'h':
                hand.add(self.shoe.deal())
                print(f"  → {hand}")
            elif choice == 's':
                hand.stood = True
                return
            elif choice == 'd':
                player.bankroll -= hand.bet
                hand.bet *= 2
                hand.doubled = True
                hand.add(self.shoe.deal())
                print(f"  Double! → {hand}")
                hand.stood = True
                return
            elif choice == 'p':
                self.split_hand(player, hand)
                return  # nové ruce se dohrají v hlavní smyčce

    def split_hand(self, player, hand):
        player.bankroll -= hand.bet
        idx = player.hands.index(hand)
        card2 = hand.cards.pop()
        new_hand = Hand(hand.bet)
        new_hand.add(card2)
        is_ace = hand.cards[0].rank == 'A'
        if is_ace:
            hand.is_split_ace = True
            new_hand.is_split_ace = True
        # doplnit po jedné kartě do obou rukou
        hand.add(self.shoe.deal())
        new_hand.add(self.shoe.deal())
        player.hands.insert(idx + 1, new_hand)
        print(f"  Split! {hand}  |  {new_hand}")

    def players_turn(self, active):
        for p in active:
            i = 0
            while i < len(p.hands):
                h = p.hands[i]
                if len(p.hands) > 1:
                    print(f"\n--- {p.name}, ruka {i+1} z {len(p.hands)} ---")
                if not h.is_blackjack:
                    self.play_hand(p, h)
                i += 1

    # -- tah krupiéra ------------------------------------------------------

    def dealer_turn(self, anyone_alive):
        self.show_table(hide_hole=False)
        if not anyone_alive:
            return
        while True:
            v = self.dealer.value
            soft = self.dealer.is_soft
            if v < 17 or (v == 17 and soft and self.h17):
                self.dealer.add(self.shoe.deal())
                print(f"Krupiér táhne → {self.dealer}")
            else:
                break
        if self.dealer.is_bust:
            print("Krupiér se přetáhl!")
        else:
            print(f"Krupiér stojí na {self.dealer.value}.")

    # -- vyhodnocení -------------------------------------------------------

    def settle(self, active):
        dv = self.dealer.value
        dealer_bust = self.dealer.is_bust
        dealer_bj = self.dealer.is_blackjack
        print("\n--- Vyhodnocení ---")
        for p in active:
            for i, h in enumerate(p.hands):
                tag = f" (ruka {i+1})" if len(p.hands) > 1 else ""
                name = f"{p.name}{tag}"
                if h.is_bust:
                    print(f"{name}: prohra (přetažení) -{h.bet}")
                    continue
                if h.is_blackjack and not dealer_bj:
                    win = int(h.bet * self.bj_payout)
                    p.bankroll += h.bet + win
                    print(f"{name}: BLACKJACK! +{win}")
                    continue
                if dealer_bj and not h.is_blackjack:
                    print(f"{name}: prohra (krupiér BJ) -{h.bet}")
                    continue
                if dealer_bj and h.is_blackjack:
                    p.bankroll += h.bet
                    print(f"{name}: push (oba BJ)")
                    continue
                if dealer_bust or h.value > dv:
                    p.bankroll += h.bet * 2
                    print(f"{name}: výhra +{h.bet}")
                elif h.value < dv:
                    print(f"{name}: prohra -{h.bet}")
                else:
                    p.bankroll += h.bet
                    print(f"{name}: push (remíza)")

    # -- jedno kolo --------------------------------------------------------

    def play_round(self):
        for p in self.players:
            p.reset_round()

        active = self.betting()
        if not active:
            return
        self.initial_deal()
        self.show_table(hide_hole=True)

        # Insurance
        self.offer_insurance(active)
        dealer_bj = False
        if self.dealer.cards[0].rank == 'A':
            dealer_bj = self.settle_insurance(active)

        # Peek: pokud krupiér ukazuje eso nebo desítku, kontrola BJ
        if self.dealer.cards[0].value in (10, 11) and self.dealer.is_blackjack:
            print(f"\nKrupiér má blackjack! {self.dealer}")
            self.settle(active)
            self.shoe.maybe_reshuffle()
            return

        # Tahy hráčů
        self.players_turn(active)

        # Je někdo, kdo nepřetáhl a nemá BJ, aby krupiér táhl?
        anyone_alive = any(
            (not h.is_bust) for p in active for h in p.hands
        )
        self.dealer_turn(anyone_alive)
        self.settle(active)
        self.shoe.maybe_reshuffle()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("          BLACKJACK – kasino edice")
    print("=" * 50)

    num = prompt_int("Počet hráčů (1-7): ", 1, 7)
    players = []
    for i in range(num):
        name = input(f"Jméno hráče {i+1}: ").strip() or f"Hráč {i+1}"
        players.append(Player(name, bankroll=1000))

    h17 = prompt_yes_no("Má krupiér táhnout na soft 17 (H17)?")
    shoe = Shoe(num_decks=6, penetration=0.75)
    game = Blackjack(players, shoe, dealer_hits_soft_17=h17)

    round_no = 1
    while True:
        alive = [p for p in players if p.bankroll > 0]
        if not alive:
            print("\nNikdo už nemá žetony. Konec hry.")
            break
        print(f"\n########## KOLO {round_no} ##########")
        game.play_round()
        round_no += 1

        print("\nStav žetonů:")
        for p in players:
            print(f"  {p.name}: {p.bankroll}")

        if not prompt_yes_no("\nHrát další kolo?"):
            break

    print("\nDěkujeme za hru! Konečný stav:")
    for p in players:
        print(f"  {p.name}: {p.bankroll} žetonů")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nHra ukončena.")
        sys.exit(0)
