import argparse
import sys

from intransidice import Die, DieMaker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Intransitive Dice sets.')
    parser.add_argument('--dice', type=int, default=5,
                        help='Number of dice in the set to make')
    parser.add_argument('--sides', type=int,default=Die.SIDES,
                        help='Number of sides of the dice')
    parser.add_argument('--alphabet', type=str, default=Die.ALPHABET,
                        help='labels of the sides in ascending order')

    args = parser.parse_args()

    Die.set_die_type(args.sides, args.alphabet)
    maker = DieMaker()
    maker.make(args.dice)
