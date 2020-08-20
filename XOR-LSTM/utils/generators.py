import random
import numpy as np
import time


def check_parity(bit_string):
    ones = 0
    zeros = 0
    for i in range(len(bit_string)):
        if bit_string[i] == 0:
            zeros+=1
        else:
            ones+=1
    if ones % 2 == 0:
        return True

    else: return False

def check_parity_numpy(bit_string):
    s = np.sum(bit_string)
    if s%2 == 0:
        return True
    else:
        return False

# generate bit string of length N
def generate_bit_string(n):
    bit_string = []
    for i in range(n):
        bit_string.append(random.randint(0, 1))

    parity = check_parity(bit_string)

    return bit_string, parity

def generate_bit_strings(l, w):
    return np.random.randint(low=0, high=1, size=[l, w])


def generate_data(l, w):

    bit_strings = generate_bit_strings(l, w)
    parities = get_parities(bit_strings)

    assert bitstrings.shape[0] == parities.shape[0]

    return bit_strings, parities


def main():

    t = time.time()
    generate_bit_strings(100000, 50)
    print(time.time()-t)

if __name__ == "__main__":
    main()
