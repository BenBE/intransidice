"""
Full explanation:
https://singingbanana.com/dice/article.htm


"""
import binascii
from itertools import product
from typing import Callable, TypeVar, List

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


def count_iterable(it):
    cnt = 0
    for e in it:
        cnt += 1
        yield e
    print("Iterator Count of" ,repr(it), "=", cnt)


class Die:
    TINDEX = np.uint64
    SIDES = 6
    ALPHABET = "123456"

    @classmethod
    def set_die_type(cls, sides, alphabet):
        global syshash
        cls.SIDES = sides
        cls.ALPHABET = alphabet
        syshash = "s{:02d}a{:02d}.{:8x}".format(sides, len(alphabet), binascii.crc32(f"{sides}-{alphabet}".encode()))

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
        B = len(cls.ALPHABET)
        def gen_unique_dice(base, digit, remaining):
            nonlocal B
            if not remaining:
                yield base
                return
            base *= B
            for d in range(digit, B):
                yield from gen_unique_dice(base + d, d, remaining - 1)

        all_dice = gen_unique_dice(0, 0, cls.SIDES)
        unique_dice = unique_everseen(all_dice, key=cls.get_die_hash)
        yield from unique_dice

    @classmethod
    def get_unique_dice_vector(cls):
        return np.fromiter(Die.get_unique_dice(), dtype=Die.TINDEX)

    @classmethod
    def get_two_dice_hash(cls, one: bytes):
        totals = [a + b for a, b in product(one, one)]
        return bytes(sorted(totals))

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


class DiceHashDG:
    """"
    This class holds the results and the "Win" directed graph for a set of dice hashes, having
    played every dice against every other.
    Dice need not be actual single dice, dice_hashes can also contain "virtual" dice of multiple throws
    """

    def __init__(self, topic: str, dice_hashes: List[bytes]) -> None:
        self.dice_hashes = {d: i for i, d in enumerate(dice_hashes)}
        self.results_sum = len(dice_hashes[0]) ** 2
        print("DiceHashDG: ", topic, "playing all", len(self.dice_hashes), "dice states with", self.results_sum, "outcomes each pair")
        self.results: np.ndarray = cache_or_recompute(f"{topic}_x_wins_against_y", self.calc_results)
        self.graph = self.results_to_graph(self.results)
        print("DiceHashDG: ", topic, "graph contains a total of", self.graph.size, "outcomes")

    def results_to_graph(self, results: np.ndarray):
        # wins > draw + loss
        return np.array(results > self.results_sum / 2, dtype=bool)

    def calc_results(self):
        # tabulate all dice pairs
        dh = list(self.dice_hashes.keys())

        def play_wdl_line(d1):
            # save some space by only including wins and converting in inner loop
            return np.array([Die.play_wdl(d1, d2)[0] for d2 in dh], dtype=np.uint16)

        data = p_map(play_wdl_line, dh)
        table = np.array(data)
        return table


class WinTable:

    def __init__(self, all_dice: np.ndarray) -> None:
        self.all_dice: np.ndarray = all_dice
        self.tidx2i_cache = {tidx: i for i, tidx in enumerate(self.all_dice)}
        print("WinTable: have", len(self.all_dice), "unique dice")
        hashes_one = [Die.get_die_hash(d) for d in self.all_dice]
        self.throwone = DiceHashDG("single", hashes_one)
        self.throwtwo = DiceHashDG("double", [Die.get_two_dice_hash(d) for d in hashes_one])

    def tidx2i(self, idx: Die.TINDEX) -> int:
        return self.tidx2i_cache[idx]

    def get_result(self, d1: Die.TINDEX, d2: Die.TINDEX):
        id1, id2 = self.tidx2i(d1), self.tidx2i(d2)
        w = self.throwone.results[id1, id2]
        l = self.throwone.results[id2, id1]
        d = self.throwone.results_sum - w - l
        return w, d, l

    def beaten_by(self, ref: Die.TINDEX) -> np.ndarray:
        """ list of TIDX that always loose to this dice """
        iref = self.tidx2i(ref)
        row = self.throwone.graph[iref]
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

    def graph_cycles_2(self, dice):
        """ Return tuples in array index format """
        yield from graphs.enumerate_fixed_len_cycles(self.table.throwone.graph, dice)

    def graph_cycles_filter_reversed_double(self, gcycles):
        lut2 = self.table.throwtwo.graph

        def check_cycle_lut(cycle):
            # check if clockwise on doubles is a loss everytime
            for d1, d2 in zip(cycle, (*cycle[1:], cycle[0])):
                d2wins = lut2[d2, d1]
                if not d2wins:
                    return False
            return True

        for gcycle in gcycles:
            if check_cycle_lut(gcycle):
                yield gcycle

    def graph_cycles_to_tidx(self, gcycles):
        for gcycle in gcycles:
            yield tuple(self.all_dice[i] for i in gcycle)

    def graph_cycles(self, dice):
        gr1 = self.table.throwone.graph
        gr2 = self.table.throwtwo.graph
        # subgraph that contains only links that are a>b with 1 and b>1 with 2
        # FIXME: is that provably true? seems waaaay to simple. but true for 6-10-{3,4,5}...
        cyclable_graph = gr1 & gr2.T
        # only use nodes that can be a loop in this subgraph
        cyclable_2_rev = graphs.adjacency_where_can_cycle(cyclable_graph)
        yield from graphs.enumerate_chains(cyclable_graph, dice, cyclable_2_rev, only_cycles=True)

    def make(self, dice):
        forward_g = self.graph_cycles(dice)
        reversible_g = self.graph_cycles_filter_reversed_double(forward_g)
        reversible = self.graph_cycles_to_tidx(reversible_g)
        cnt = 0
        for cycle in reversible:
            cnt += 1
            print(" -> ".join(Die.get_die_name(d) for d in cycle), flush=True)
        print("total:", cnt)
        print("")
