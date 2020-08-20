import random
import numpy as np
import time

def generate_bit_strings(l, w):
    return np.random.randint(low=0, high=2, size=[l, w])


def get_parities(bit_strings):

    s = np.sum(bit_strings, axis=1)

    return ((s%2 == 0)*1).reshape([-1, 1])


def generate_data(l, w):

    bit_strings = generate_bit_strings(l, w)
    parities = get_parities(bit_strings)

    assert bit_strings.shape[0] == parities.shape[0]

    return bit_strings, parities


def generate_rand_data():
    pass


def main():

    t = time.time()
    bit_strings, parities = generate_data(10, 5)
    print(time.time()-t)
    print(bit_strings, parities)

    print(parities.shape)


if __name__ == "__main__":
    main()
