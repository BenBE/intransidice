"""
Full explanation:
https://singingbanana.com/dice/article.htm


"""
import binascii
from itertools import product
from typing import Callable, TypeVar

import numpy as np

from more_itertools import unique_everseen

from p_tqdm import p_map
from intransidice import graphs

T = TypeVar("T")

syshash = "any"


def cache_or_recompute(item: str, c: Callable[[], T]) -> T:
    item = f"{item}.{syshash}.npy"
    try:
        return np.load(item)
    except IOError:
        val = c()
        np.save(item, val)
        return val


class Die:
    TINDEX = np.uint64
    SIDES = 6
    ALPHABET = "123456"

    @classmethod
    def set_die_type(cls, sides, alphabet):
        global syshash
        cls.SIDES = sides
        cls.ALPHABET = alphabet
        syshash = "{:8x}".format(binascii.crc32(f"{sides}-{alphabet}".encode()))

    @classmethod
    def get_dice_count(cls):
        return len(cls.ALPHABET) ** cls.SIDES

    @classmethod
    def get_die_index(cls, name: str) -> TINDEX:
        S = cls.SIDES
        A = cls.ALPHABET
        B = len(A)
        return sum(A.index(c) * B ** e for e, c in enumerate(name))

    @classmethod
    def get_die_name(cls, index: TINDEX) -> str:
        """ return user-friendly representation of this dice """
        i = int(index)
        S = cls.SIDES
        A = cls.ALPHABET
        B = len(A)
        die = list(A[0] * S)
        for e in range(S):
            die[e] = A[i % B]
            i //= B
        return "".join(reversed(die))

    @classmethod
    def get_die_hash(cls, index: TINDEX) -> bytes:
        """ return minimal unique representation """
        i = int(index)
        S = cls.SIDES
        B = len(cls.ALPHABET)
        die = [0] * S
        for e in range(S):
            die[e] = i % B
            i //= B
        return bytes(sorted(die))

    @classmethod
    def get_unique_dice(cls):
        all_dice = range(cls.get_dice_count())
        unique_dice = unique_everseen(all_dice, key=cls.get_die_hash)
        yield from unique_dice

    @classmethod
    def get_unique_dice_vector(cls):
        return np.fromiter(Die.get_unique_dice(), dtype=Die.TINDEX)

    @staticmethod
    def play_wdl(d1: bytes, d2: bytes):
        """ statistics for playing hyperdice with sides given by d1 and d2 """

        ad1 = np.tile(list(d1), (len(d2), 1)).T
        ad2 = np.tile(list(d2), (len(d1), 1))

        win = np.count_nonzero(ad1 > ad2)
        loss = np.count_nonzero(ad1 < ad2)
        draw = ad1.size - win - loss
        return win, draw, loss


Die.set_die_type(Die.SIDES, Die.ALPHABET)


class WinTable:

    def __init__(self, all_dice: np.ndarray) -> None:
        self.all_dice: np.ndarray = all_dice
        self.tidx2i_cache = {tidx: i for i, tidx in enumerate(self.all_dice)}
        print("WinTable: have", len(self.all_dice), "unique dice")
        self.wdl_table: np.ndarray = cache_or_recompute("x_wins_against_y", self.calc_wintable)
        print("WinTable: have", str(self.wdl_table.shape), "total dice pairs")
        # wins > draw + loss
        self.wins_against: np.ndarray = np.array(
            self.wdl_table[:, :, 0] > self.wdl_table[:, :, 1] + self.wdl_table[:, :, 2], dtype=bool)

    def calc_wintable(self):
        dh = [Die.get_die_hash(d) for d in self.all_dice]

        def play_wdl_line(d):
            return [Die.play_wdl(d, d2) for d2 in dh]

        data = p_map(play_wdl_line, dh)
        table = np.array(data, dtype=np.int8)
        return table

    def tidx2i(self, idx: Die.TINDEX) -> int:
        return self.tidx2i_cache[idx]

    def get_result(self, d1: Die.TINDEX, d2: Die.TINDEX):
        return self.wdl_table[self.tidx2i(d1), self.tidx2i(d2)]

    def beaten_by(self, ref: Die.TINDEX) -> np.ndarray:
        """ list of TIDX that always loose to this dice """
        iref = self.tidx2i(ref)
        row = self.wins_against[iref]
        lst = self.all_dice[row]
        return lst


class DieMaker:

    def __init__(self):
        print("DieMaker Setup:")
        print("  Sides:   %s" % Die.SIDES)
        print("  Values:  (%d) %s" % (len(Die.ALPHABET), Die.ALPHABET))

        self.all_dice: np.array = cache_or_recompute("all_unique_dice", Die.get_unique_dice_vector)
        self.table = WinTable(self.all_dice)

    @staticmethod
    def canonical_ordering(cycle):
        pivot = np.argmax(cycle)
        return tuple(cycle[pivot:] + cycle[:pivot])

    def fixed_cycles(self, dice=3):
        for gcycle in graphs.enumerate_fixed_len_cycles(self.table.wins_against, dice):
            # gcycle is in array indices
            cycle = tuple(self.all_dice[i] for i in gcycle)
            yield cycle

    def reverses_when_double(self, cycles):
        def get_twodice(one: bytes):
            totals = [a + b for a, b in product(one, one)]
            return bytes(sorted(totals))

        def check_cycle(cycle):
            rev = list(reversed(cycle))
            for d1, d2 in zip(rev, rev[1:] + [rev[0]]):
                double1 = get_twodice(Die.get_die_hash(d1))
                double2 = get_twodice(Die.get_die_hash(d2))
                w, d, l = Die.play_wdl(double1, double2)
                d1wins = w > d + l
                if not d1wins:
                    return False
            return True

        for cycle in cycles:
            if check_cycle(cycle):
                yield cycle

    def make(self, dice):

        # forward = self.simple_cycles(max_dice=3, min_dice=3)
        forward = self.fixed_cycles(dice=dice)
        reversible = self.reverses_when_double(forward)
        for cycle in reversible:
            print("\n", " -> ".join(Die.get_die_name(d) for d in cycle), flush=True)

        print("")
