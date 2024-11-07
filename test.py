import torch
import numpy as np


def generate_random_fingerprints(fingerprint_size=100, batch_size=64):
    torch.manual_seed(64)
    #2 excluded, it creates a tensor of 0 and 1 with batch_size x fingerprint_size size
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2)
    return z


def main():

    fin = generate_random_fingerprints(100,1)
    fin = np.array(fin)
    print(fin.shape)
    for bit in fin:
        print(bit)
   


if __name__ == "__main__":
    main()
