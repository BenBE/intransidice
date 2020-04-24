try:
    import intransidice
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__) + '/..')

import argparse

from intransidice import Die, DieMaker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Intransitive Dice sets.')
    parser.add_argument('--dice', type=int, default=3,
                        help='Number of dice in the set to make')
    parser.add_argument('--sides', type=int,default=Die.SIDES,
                        help='Number of sides of the dice')
    parser.add_argument('--alphabet', type=str, default=Die.ALPHABET,
                        help='labels of the sides in ascending order')
    parser.add_argument('--filter-coinlike', type=bool, default=False,
                        help='only return die that are coinlike')
    parser.add_argument('--eval', type=str, nargs=2, default=None,
                        help='Evaluate a pair of dice')

    args = parser.parse_args()

    Die.set_die_type(args.sides, args.alphabet)
    maker = DieMaker()
    if args.eval is not None:
        maker.evaluate(*args.eval)
    else:
        maker.make(args.dice, args.filter_coinlike)
